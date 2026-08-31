// odin_tokenizer_fast.c
//
// Frozen C tokenizer for the Odin byte-level BPE vocabulary.
//
// The pipeline is bit-for-bit identical to `tokenizers` 0.22.2
// (PreTrainedTokenizerFast defaults for this tokenizer):
//
//   1. Added-vocabulary pass: literal leftmost-longest extraction of the 17
//      special tokens ([PAD] ... [am]); each match emits its id directly.
//   2. GPT-2 ByteLevel pre-tokenizer regex chunking on the remaining text:
//        's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//      (the \s+(?!\S) backtracking splits interior whitespace runs).
//   3. Byte-level BPE per chunk: symbols are the single-byte token ids, then
//      the iterative merge-list algorithm (min-heap ordered by
//      (rank, position), stale-entry validation) — as in tokenizers'
//      `Word::merge_all`. This is NOT greedy longest-match.
//
// Decode: ids -> piece bytes (specials rendered as their literal string
// unless skipped), concatenated and UTF-8 decoded with U+FFFD replacement
// (from_utf8_lossy semantics).
//
// Concurrency: `encode_many` / `encode_padded` fan out over a thread pool
// with the GIL released; a process-wide FNV-64 word cache (thread-safe via
// release/acquire publication) makes repeated surfaces ~10x cheaper.

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "odin_tokenizer_tables.h"

#define ID_BITS 14

// ---------------------------------------------------------------------------
// Unicode code-point helpers
// ---------------------------------------------------------------------------
static inline uint32_t utf8_next(const uint8_t *p, const uint8_t **p_next) {
    uint32_t c = p[0];
    if (c < 0x80) {
        *p_next = p + 1;
        return c;
    }
    if (c < 0xE0) {
        *p_next = p + 2;
        return ((c & 0x1Fu) << 6) | (p[1] & 0x3Fu);
    }
    if (c < 0xF0) {
        *p_next = p + 3;
        return ((c & 0x0Fu) << 12) | ((p[1] & 0x3Fu) << 6) | (p[2] & 0x3Fu);
    }
    *p_next = p + 4;
    return ((c & 0x07u) << 18) | ((p[1] & 0x3Fu) << 12) | ((p[2] & 0x3Fu) << 6) | (p[3] & 0x3Fu);
}

static inline int nb_in(const uint32_t *ranges, int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        uint32_t s = ranges[2 * mid], e = ranges[2 * mid + 1];
        if (cp < s)
            hi = mid - 1;
        else if (cp > e)
            lo = mid + 1;
        else
            return 1;
    }
    return 0;
}

// 0 = other, 1 = \p{L}, 2 = \p{N}, 3 = \s
static inline int cp_class(uint32_t cp) {
    if (cp < 0x10000) {
        uint64_t bit = 1ull << (cp & 63);
        int w = (int)(cp >> 6);
        return (CL_L[w] & bit) ? 1 : (CL_N[w] & bit) ? 2 : (CL_WS[w] & bit) ? 3 : 0;
    }
    if (nb_in(CL_L_NB, CL_L_NB_N, cp)) return 1;
    if (nb_in(CL_N_NB, CL_N_NB_N, cp)) return 2;
    return 0;
}

// ---------------------------------------------------------------------------
// BPE merge map lookup
// ---------------------------------------------------------------------------
static inline int merge_lookup(uint16_t a, uint16_t b, uint32_t *val_out) {
    uint32_t key = ((uint32_t)a << ID_BITS) | b;
    uint32_t stored = key + 1;
    int slot = (int)((key * 0x9E3779B9u) & (MERGE_SLOTS - 1));
    uint32_t k = MERGE_KEY[slot];
    while (k != 0 && k != stored) {
        slot = (slot + 1) & (MERGE_SLOTS - 1);
        k = MERGE_KEY[slot];
    }
    if (k != stored) return 0;
    *val_out = MERGE_VAL[slot];
    return 1;
}

// ---------------------------------------------------------------------------
// Min-heap ordered by (rank, pos)
// ---------------------------------------------------------------------------
typedef struct {
    uint16_t rank;
    int32_t pos;
    uint16_t new_id;
} HeapNode;

static inline int hlt(const HeapNode *a, const HeapNode *b) {
    if (a->rank != b->rank) return a->rank < b->rank;
    return a->pos < b->pos;
}

static void heap_push(HeapNode *h, int *n, uint16_t rank, int32_t pos, uint16_t new_id) {
    int i = (*n)++;
    HeapNode item;
    item.rank = rank;
    item.pos = pos;
    item.new_id = new_id;
    while (i > 0) {
        int par = (i - 1) / 2;
        if (!hlt(&item, &h[par])) break;
        h[i] = h[par];
        i = par;
    }
    h[i] = item;
}

static void heap_pop(HeapNode *h, int *n) {
    HeapNode item = h[--(*n)];
    int i = 0;
    for (;;) {
        int child = 2 * i + 1;
        if (child >= *n) break;
        if (child + 1 < *n && hlt(&h[child + 1], &h[child])) child++;
        if (!hlt(&h[child], &item)) break;
        h[i] = h[child];
        i = child;
    }
    h[i] = item;
}

// ---------------------------------------------------------------------------
// Per-thread scratch (symbol linked list + heap)
// ---------------------------------------------------------------------------
typedef struct {
    uint16_t *sym;
    int32_t *prev;
    int32_t *nxt;
    uint8_t *alive;
    HeapNode *heap;
    int cap;
} Scratch;

static void scratch_init(Scratch *ts) {
    ts->sym = NULL;
    ts->prev = NULL;
    ts->nxt = NULL;
    ts->alive = NULL;
    ts->heap = NULL;
    ts->cap = 0;
}

static void scratch_grow(Scratch *ts, int need) {
    if (need <= ts->cap) return;
    int ncap = ts->cap ? ts->cap : 256;
    while (ncap < need) ncap *= 2;
    ts->sym = realloc(ts->sym, (size_t)ncap * sizeof(uint16_t));
    ts->prev = realloc(ts->prev, (size_t)ncap * sizeof(int32_t));
    ts->nxt = realloc(ts->nxt, (size_t)ncap * sizeof(int32_t));
    ts->alive = realloc(ts->alive, (size_t)ncap);
    ts->heap = realloc(ts->heap, ((size_t)ncap * 3 + 8) * sizeof(HeapNode));
    ts->cap = ncap;
}

static void scratch_free(Scratch *ts) {
    free(ts->sym);
    free(ts->prev);
    free(ts->nxt);
    free(ts->alive);
    free(ts->heap);
    ts->cap = 0;
}

// Iterative BPE merge of one pre-tokenized chunk (bytes -> token ids).
static int bpe_chunk(const uint8_t *bytes, int m, uint16_t *out, Scratch *ts) {
    if (m <= 0) return 0;
    scratch_grow(ts, m);
    uint16_t *sym = ts->sym;
    int32_t *prev = ts->prev;
    int32_t *nxt = ts->nxt;
    uint8_t *alive = ts->alive;
    HeapNode *heap = ts->heap;
    int nheap = 0;

    for (int i = 0; i < m; i++) {
        sym[i] = BYTE_ID[bytes[i]];
        alive[i] = 1;
        prev[i] = i - 1;
        nxt[i] = (i + 1 < m) ? i + 1 : -1;
    }
    for (int i = 0; i + 1 < m; i++) {
        uint32_t v;
        if (merge_lookup(sym[i], sym[i + 1], &v))
            heap_push(heap, &nheap, (uint16_t)(v >> 16), i, (uint16_t)(v & 0xFFFFu));
    }
    while (nheap > 0) {
        HeapNode top = heap[0];
        heap_pop(heap, &nheap);
        int pos = top.pos;
        if (!alive[pos]) continue;
        int pnext = nxt[pos];
        if (pnext < 0 || !alive[pnext]) continue;
        uint32_t v;
        if (!merge_lookup(sym[pos], sym[pnext], &v) || (v & 0xFFFFu) != top.new_id) continue;
        // merge pos + pnext
        sym[pos] = top.new_id;
        alive[pnext] = 0;
        nxt[pos] = nxt[pnext];
        if (nxt[pos] >= 0) prev[nxt[pos]] = pos;
        int pprev = prev[pos];
        if (pprev >= 0) {
            uint32_t v2;
            if (merge_lookup(sym[pprev], sym[pos], &v2))
                heap_push(heap, &nheap, (uint16_t)(v2 >> 16), pprev, (uint16_t)(v2 & 0xFFFFu));
        }
        int nn = nxt[pos];
        if (nn >= 0) {
            uint32_t v2;
            if (merge_lookup(sym[pos], sym[nn], &v2))
                heap_push(heap, &nheap, (uint16_t)(v2 >> 16), pos, (uint16_t)(v2 & 0xFFFFu));
        }
    }
    int k = 0;
    for (int i = 0; i < m; i++)
        if (alive[i]) out[k++] = sym[i];
    return k;
}

// ---------------------------------------------------------------------------
// Regex chunking (GPT-2 ByteLevel pre-tokenizer)
// ---------------------------------------------------------------------------
static const uint8_t *run_end(const uint8_t *p, const uint8_t *end, int cls) {
    while (p < end) {
        const uint8_t *save = p;
        uint32_t c = utf8_next(save, &p);
        if (cp_class(c) != cls) {
            p = save;
            break;
        }
    }
    return p;
}

// Apostrophe alternatives: 's 't 're 've 'm 'll 'd (first match wins).
// Returns the chunk length in bytes (from p) or 0.
static inline int apostrophe_match(const uint8_t *p, const uint8_t *p1, const uint8_t *end) {
    if (p1 >= end) return 0;
    uint8_t d = p1[0];
    switch (d) {
        case 's':
        case 't':
        case 'm':
        case 'd':
            return 2;
        case 'r':
            return (p1 + 1 < end && p1[1] == 'e') ? 3 : 0;
        case 'v':
            return (p1 + 1 < end && p1[1] == 'e') ? 3 : 0;
        case 'l':
            return (p1 + 1 < end && p1[1] == 'l') ? 3 : 0;
        default:
            return 0;
    }
}

// ---------------------------------------------------------------------------
// Full encode of one string (specials + chunks + BPE)
// ---------------------------------------------------------------------------
static int encode_core(const uint8_t *s, int n, uint16_t *out, Scratch *ts) {
    int k = 0;
    const uint8_t *p = s;
    const uint8_t *end = s + n;
    while (p < end) {
        const uint8_t *cs = p;
        // 1. special-token literal extraction
        if (p[0] == '[' && (end - p) >= 4) {
            int si = SPECIAL_SEC[p[1]];
            if (si != 0xFF) {
                int cnt = SPECIAL_SEC_N[p[1]];
                for (int t = 0; t < cnt; t++) {
                    int e = si + t;
                    int L = SPECIAL_LEN[e];
                    if ((int)(end - p) < L) continue;
                    const uint16_t *tl = &SPECIAL_TAIL[e * 4];
                    int ok = 1;
                    for (int j = 1; j < L; j++) {
                        if (p[j] != tl[j - 1]) {
                            ok = 0;
                            break;
                        }
                    }
                    if (ok) {
                        out[k++] = SPECIAL_ID[e];
                        p += L;
                        goto next_iter;
                    }
                }
            }
        }
        // 2. regex chunk
        const uint8_t *p1;
        uint32_t c0 = utf8_next(p, &p1);
        int cl0 = cp_class(c0);
        const uint8_t *ce;
        if (c0 == 0x27) {
            int mlen = apostrophe_match(p, p1, end);
            ce = (mlen > 0) ? p + mlen : run_end(p1, end, 0);
        } else if (cl0 == 1) {
            ce = run_end(p1, end, 1);
        } else if (cl0 == 2) {
            ce = run_end(p1, end, 2);
        } else if (cl0 == 3) {
            if (p1 < end) {
                const uint8_t *p2;
                uint32_t c1 = utf8_next(p1, &p2);
                int cl1 = cp_class(c1);
                if (cl1 == 1) {
                    ce = run_end(p2, end, 1);
                } else if (cl1 == 2) {
                    ce = run_end(p2, end, 2);
                } else if (cl1 == 0) {
                    ce = run_end(p2, end, 0);
                } else {
                    // whitespace run: c0 (at p) plus the whitespace run
                    // starting at p1. \s+(?!\S) backtracks one code point
                    // short of the next non-whitespace char, so an interior
                    // run of k spaces emits k-1; a trailing run emits all k.
                    const uint8_t *q = p1;
                    const uint8_t *last = p1;
                    int interior = 0;
                    while (q < end) {
                        const uint8_t *save = q;
                        uint32_t cc = utf8_next(q, &q);
                        if (cp_class(cc) != 3) {
                            interior = 1;
                            break;
                        }
                        last = save;
                    }
                    ce = interior ? last : q;
                }
            } else {
                ce = p1;  // single trailing whitespace code point
            }
        } else {
            ce = run_end(p1, end, 0);
        }
        // 3. BPE this chunk
        k += bpe_chunk(cs, (int)(ce - cs), out + k, ts);
        p = ce;
    next_iter:
        ;
    }
    return k;
}

// ---------------------------------------------------------------------------
// Word cache (process-wide, thread-safe)
// ---------------------------------------------------------------------------
#define CACHE_SLOTS 4096
#define CACHE_MAX_IDS 128

typedef struct {
    _Atomic uint64_t hash;
    uint16_t len;
    uint8_t prefix[8];
    uint16_t ids[CACHE_MAX_IDS];
} CacheEntry;

static CacheEntry g_cache[CACHE_SLOTS];

static uint64_t fnv64(const uint8_t *s, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; i++) {
        h ^= s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int encode_one(const uint8_t *s, int n, uint16_t *out, Scratch *ts) {
    if (n == 0) return 0;
    uint64_t h = fnv64(s, n);
    CacheEntry *e = &g_cache[h & (CACHE_SLOTS - 1)];
    uint64_t eh = atomic_load_explicit(&e->hash, memory_order_acquire);
    if (eh == h && e->len > 0 && e->len <= CACHE_MAX_IDS) {
        uint16_t len = e->len;
        int pc = (len < 8) ? (int)len : 8;
        if (memcmp(e->prefix, s, pc) == 0) {
            memcpy(out, e->ids, (size_t)len * 2);
            return (int)len;
        }
    }
    int k = encode_core(s, n, out, ts);
    if (k <= CACHE_MAX_IDS) {
        memcpy(e->ids, out, (size_t)k * 2);
        int pc = (k < 8) ? k : 8;
        memcpy(e->prefix, s, (size_t)pc);
        if (pc < 8) memset(e->prefix + pc, 0, (size_t)(8 - pc));
        e->len = (uint16_t)k;
        atomic_store_explicit(&e->hash, h, memory_order_release);
    }
    return k;
}

// ---------------------------------------------------------------------------
// Thread pool for batch encode
// ---------------------------------------------------------------------------
static int g_nthreads = 8;

typedef struct {
    const uint8_t **data;
    const int *lens;
    uint16_t **outs;
    int *out_lens;
    int n;
    _Atomic int next;
} ParallelTask;

static void *worker(void *arg) {
    ParallelTask *t = (ParallelTask *)arg;
    Scratch ts;
    scratch_init(&ts);
    for (;;) {
        int i = atomic_fetch_add_explicit(&t->next, 1, memory_order_relaxed);
        if (i >= t->n) break;
        t->out_lens[i] = encode_one(t->data[i], t->lens[i], t->outs[i], &ts);
    }
    scratch_free(&ts);
    return NULL;
}

static void run_parallel(ParallelTask *t) {
    int nt = g_nthreads;
    if (nt > t->n) nt = t->n;
    if (nt <= 1) {
        Scratch ts;
        scratch_init(&ts);
        for (int i = 0; i < t->n; i++) t->out_lens[i] = encode_one(t->data[i], t->lens[i], t->outs[i], &ts);
        scratch_free(&ts);
        return;
    }
    atomic_init(&t->next, 0);
    pthread_t *ths = malloc((size_t)nt * sizeof(pthread_t));
    for (int i = 0; i < nt; i++) pthread_create(&ths[i], NULL, worker, t);
    for (int i = 0; i < nt; i++) pthread_join(ths[i], NULL);
    free(ths);
}

// ---------------------------------------------------------------------------
// Decode helpers
// ---------------------------------------------------------------------------
static int collect_ids(PyObject *ids_obj, const long **ids_out, Py_ssize_t *n_out) {
    PyObject *fast = PySequence_Fast(ids_obj, "ids must be a sequence of ints");
    if (!fast) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    long *ids = malloc((size_t)(n > 0 ? n : 1) * sizeof(long));
    for (Py_ssize_t i = 0; i < n; i++) {
        ids[i] = (long)PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, i));
        if (ids[i] == -1 && PyErr_Occurred()) {
            free(ids);
            Py_DECREF(fast);
            return -1;
        }
    }
    Py_DECREF(fast);
    *ids_out = ids;
    *n_out = n;
    return 0;
}

static PyObject *decode_one(const long *ids, Py_ssize_t n, int skip) {
    long total = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        long id = ids[i];
        if (id < 0 || id >= VOCAB_SIZE) {
            PyErr_Format(PyExc_ValueError, "id %ld out of range [0, %d)", id, VOCAB_SIZE);
            return NULL;
        }
        if (id < N_SPECIALS && skip) continue;
        total += PIECE_LEN[(int)id];
    }
    uint8_t *bytes = malloc((size_t)(total > 0 ? total : 1));
    long o = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        long id = ids[i];
        if (id < N_SPECIALS && skip) continue;
        memcpy(bytes + o, PIECE_BYTES + PIECE_OFF[(int)id], PIECE_LEN[(int)id]);
        o += PIECE_LEN[(int)id];
    }
    PyObject *s = PyUnicode_DecodeUTF8((const char *)bytes, (Py_ssize_t)total, "replace");
    free(bytes);
    return s;
}

// ---------------------------------------------------------------------------
// Python API
// ---------------------------------------------------------------------------
static PyObject *mod_encode(PyObject *Py_UNUSED(self), PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 1 || !PyUnicode_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "encode(text: str)");
        return NULL;
    }
    Py_ssize_t n;
    const char *s = PyUnicode_AsUTF8AndSize(args[0], &n);
    if (!s) return NULL;
    uint16_t *buf = malloc(((size_t)n > 0 ? (size_t)n : 1) * sizeof(uint16_t));
    Scratch ts;
    scratch_init(&ts);
    int k;
    Py_BEGIN_ALLOW_THREADS
    k = encode_one((const uint8_t *)s, (int)n, buf, &ts);
    Py_END_ALLOW_THREADS
    PyObject *lst = PyList_New(k);
    if (!lst) {
        free(buf);
        scratch_free(&ts);
        return NULL;
    }
    for (int i = 0; i < k; i++) PyList_SET_ITEM(lst, i, PyLong_FromLong((long)buf[i]));
    free(buf);
    scratch_free(&ts);
    return lst;
}

static PyObject *mod_encode_many(PyObject *Py_UNUSED(self), PyObject *seq) {
    if (!PyList_Check(seq) && !PyTuple_Check(seq)) {
        PyErr_SetString(PyExc_TypeError, "encode_many(texts: Sequence[str])");
        return NULL;
    }
    Py_ssize_t n = PySequence_Size(seq);
    if (n == 0) return PyList_New(0);
    const uint8_t **data = malloc((size_t)n * sizeof(char *));
    int *lens = malloc((size_t)n * sizeof(int));
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *o = PySequence_GetItem(seq, i);
        if (!PyUnicode_Check(o)) {
            Py_DECREF(o);
            free(data);
            free(lens);
            PyErr_SetString(PyExc_TypeError, "encode_many: all items must be str");
            return NULL;
        }
        Py_ssize_t l;
        const char *u = PyUnicode_AsUTF8AndSize(o, &l);
        Py_DECREF(o);
        if (!u) {
            free(data);
            free(lens);
            return NULL;
        }
        data[i] = (const uint8_t *)u;
        lens[i] = (int)l;
    }
    // single arena for all per-string output buffers (output can never
    // exceed the input byte count)
    uint16_t **outs = malloc((size_t)n * sizeof(uint16_t *));
    int *out_lens = malloc((size_t)n * sizeof(int));
    size_t total = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        outs[i] = NULL;
        total += (size_t)lens[i] > 0 ? (size_t)lens[i] : 1;
    }
    uint16_t *arena = malloc(total * sizeof(uint16_t));
    size_t off = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        outs[i] = arena + off;
        off += (size_t)lens[i] > 0 ? (size_t)lens[i] : 1;
    }

    ParallelTask pt = {data, lens, outs, out_lens, (int)n, 0};
    Py_BEGIN_ALLOW_THREADS
    run_parallel(&pt);
    Py_END_ALLOW_THREADS

    PyObject *res = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *lst = PyList_New(out_lens[i]);
        for (int j = 0; j < out_lens[i]; j++) PyList_SET_ITEM(lst, j, PyLong_FromLong((long)outs[i][j]));
        PyList_SET_ITEM(res, i, lst);
    }
    free(arena);
    free(outs);
    free(out_lens);
    free(data);
    free(lens);
    return res;
}

static PyObject *mod_encode_padded(PyObject *Py_UNUSED(self), PyObject *const *args, Py_ssize_t nargs) {
    // encode_padded(texts, max_len, pad_id) -> (ndarray (N, max_len) uint16, lengths (N,) uint8)
    if (nargs != 3 || !PyList_Check(args[0]) && !PyTuple_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "encode_padded(texts, max_len, pad_id)");
        return NULL;
    }
    long max_len = PyLong_AsLong(args[1]);
    long pad_id = PyLong_AsLong(args[2]);
    if (max_len < 0 || max_len > 255 || pad_id < 0 || pad_id >= VOCAB_SIZE) {
        PyErr_SetString(PyExc_ValueError, "max_len must be in [0, 255], pad_id in [0, VOCAB_SIZE)");
        return NULL;
    }
    PyObject *seq = args[0];
    Py_ssize_t n = PySequence_Size(seq);
    npy_intp dims[2] = {n, max_len};
    PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT16);
    PyArrayObject *lens_arr = (PyArrayObject *)PyArray_SimpleNew(1, (npy_intp[1]){n}, NPY_UINT16);
    if (!arr || !lens_arr) {
        Py_XDECREF(arr);
        Py_XDECREF(lens_arr);
        return NULL;
    }
    if (n == 0) return PyTuple_Pack(2, (PyObject *)arr, (PyObject *)lens_arr);

    const uint8_t **data = malloc((size_t)n * sizeof(char *));
    int *lens = malloc((size_t)n * sizeof(int));
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *o = PySequence_GetItem(seq, i);
        if (!PyUnicode_Check(o)) {
            Py_DECREF(o);
            free(data);
            free(lens);
            Py_DECREF(arr);
            Py_DECREF(lens_arr);
            PyErr_SetString(PyExc_TypeError, "encode_padded: all items must be str");
            return NULL;
        }
        Py_ssize_t l;
        const char *u = PyUnicode_AsUTF8AndSize(o, &l);
        Py_DECREF(o);
        if (!u) {
            free(data);
            free(lens);
            Py_DECREF(arr);
            Py_DECREF(lens_arr);
            return NULL;
        }
        data[i] = (const uint8_t *)u;
        lens[i] = (int)l;
    }
    uint16_t **outs = malloc((size_t)n * sizeof(uint16_t *));
    int *out_lens = malloc((size_t)n * sizeof(int));
    size_t total = 0;
    for (Py_ssize_t i = 0; i < n; i++) total += (size_t)lens[i] > 0 ? (size_t)lens[i] : 1;
    uint16_t *arena = malloc(total * sizeof(uint16_t));
    size_t off = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        outs[i] = arena + off;
        off += (size_t)lens[i] > 0 ? (size_t)lens[i] : 1;
    }

    ParallelTask pt = {data, lens, outs, out_lens, (int)n, 0};
    Py_BEGIN_ALLOW_THREADS
    run_parallel(&pt);
    Py_END_ALLOW_THREADS

    uint16_t *row = (uint16_t *)PyArray_DATA(arr);
    uint16_t *lens_data = (uint16_t *)PyArray_DATA(lens_arr);
    for (Py_ssize_t i = 0; i < n; i++) {
        int k = out_lens[i];
        int copy = (k < (int)max_len) ? k : (int)max_len;
        if (copy > 0) memcpy(row + (size_t)i * max_len, outs[i], (size_t)copy * 2);
        if (copy < (int)max_len)
            memset(row + (size_t)i * max_len + copy, 0, ((size_t)max_len - copy) * 2);
        for (int j = copy; j < (int)max_len; j++) row[(size_t)i * max_len + j] = (uint16_t)pad_id;
        lens_data[i] = (uint16_t)k;  // RAW (untruncated) token count
    }
    free(arena);
    free(outs);
    free(out_lens);
    free(data);
    free(lens);
    return PyTuple_Pack(2, (PyObject *)arr, (PyObject *)lens_arr);
}

static PyObject *mod_decode(PyObject *Py_UNUSED(self), PyObject *const *args, Py_ssize_t nargs,
                            PyObject *kwnames) {
    // decode(ids, skip_special_tokens=True)
    if (nargs < 1 || nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "decode(ids, skip_special_tokens=True)");
        return NULL;
    }
    PyObject *skip_obj = Py_True;
    if (nargs == 2) skip_obj = args[1];
    if (kwnames) {
        Py_ssize_t nk = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t i = 0; i < nk; i++) {
            PyObject *name = PyTuple_GET_ITEM(kwnames, i);
            if (PyUnicode_CompareWithASCIIString(name, "skip_special_tokens") == 0) {
                skip_obj = args[nargs + i];
            } else {
                PyErr_SetString(PyExc_TypeError, "unexpected keyword argument");
                return NULL;
            }
        }
    }
    int skip = PyObject_IsTrue(skip_obj);
    if (skip < 0) return NULL;
    const long *ids;
    Py_ssize_t n;
    if (collect_ids(args[0], &ids, &n) < 0) return NULL;
    PyObject *s = decode_one(ids, n, skip);
    free((void *)ids);
    return s;
}

static PyObject *mod_decode_many(PyObject *Py_UNUSED(self), PyObject *const *args, Py_ssize_t nargs,
                                 PyObject *kwnames) {
    // decode_many(id_lists, skip_special_tokens=True)
    if (nargs < 1 || nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "decode_many(id_lists, skip_special_tokens=True)");
        return NULL;
    }
    PyObject *skip_obj = Py_True;
    if (nargs == 2) skip_obj = args[1];
    if (kwnames) {
        Py_ssize_t nk = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t i = 0; i < nk; i++) {
            PyObject *name = PyTuple_GET_ITEM(kwnames, i);
            if (PyUnicode_CompareWithASCIIString(name, "skip_special_tokens") == 0) {
                skip_obj = args[nargs + i];
            } else {
                PyErr_SetString(PyExc_TypeError, "unexpected keyword argument");
                return NULL;
            }
        }
    }
    int skip = PyObject_IsTrue(skip_obj);
    if (skip < 0) return NULL;
    PyObject *seq = args[0];
    Py_ssize_t n = PySequence_Size(seq);
    PyObject *res = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *o = PySequence_GetItem(seq, i);
        const long *ids;
        Py_ssize_t m;
        if (collect_ids(o, &ids, &m) < 0) {
            Py_DECREF(o);
            Py_DECREF(res);
            return NULL;
        }
        PyObject *s = decode_one(ids, m, skip);
        free((void *)ids);
        Py_DECREF(o);
        if (!s) {
            Py_DECREF(res);
            return NULL;
        }
        PyList_SET_ITEM(res, i, s);
    }
    return res;
}

static uint32_t fnv32(const uint8_t *s, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) {
        h ^= s[i];
        h *= 16777619u;
    }
    return h;
}

static PyObject *mod_token_to_id(PyObject *Py_UNUSED(self), PyObject *arg) {
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "token_to_id(token: str)");
        return NULL;
    }
    Py_ssize_t n;
    const char *s = PyUnicode_AsUTF8AndSize(arg, &n);
    if (!s) return NULL;
    // Map the token's UTF-8 code points back to raw piece bytes (the table is
    // keyed on raw bytes, not UTF-8).
    uint8_t *raw = malloc((size_t)n > 0 ? (size_t)n : 1);
    const uint8_t *q = (const uint8_t *)s, *qend = q + n;
    int nr = 0, bad = 0;
    while (q < qend) {
        const uint8_t *qn;
        uint32_t cp = utf8_next(q, &qn);
        if (cp >= 352 || CHAR_TO_BYTE[cp] == 0xFF) {
            bad = 1;
            break;
        }
        raw[nr++] = CHAR_TO_BYTE[cp];
        q = qn;
    }
    if (bad) {
        free(raw);
        Py_RETURN_NONE;
    }
    uint32_t h = fnv32(raw, nr);
    int slot = (int)(h & (TID_SLOTS - 1));
    uint32_t k = TID_KEY[slot];
    while (k != 0) {
        int idx = (int)(k - 1);
        if (TID_HASH[slot] == h && PIECE_LEN[idx] == (uint8_t)nr &&
            memcmp(PIECE_BYTES + PIECE_OFF[idx], raw, (size_t)nr) == 0) {
            free(raw);
            return PyLong_FromLong((long)idx);
        }
        slot = (slot + 1) & (TID_SLOTS - 1);
        k = TID_KEY[slot];
    }
    free(raw);
    Py_RETURN_NONE;
}

static PyObject *mod_set_num_threads(PyObject *Py_UNUSED(self), PyObject *arg) {
    long v = PyLong_AsLong(arg);
    if (v < 1 || v > 64) {
        PyErr_SetString(PyExc_ValueError, "num_threads must be in [1, 64]");
        return NULL;
    }
    g_nthreads = (int)v;
    Py_RETURN_NONE;
}

static PyObject *mod_num_threads(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args)) {
    return PyLong_FromLong((long)g_nthreads);
}

static PyMethodDef Methods[] = {
    {"encode", (PyCFunction)(void (*)(void))mod_encode, METH_FASTCALL, "encode(text) -> list[int]"},
    {"encode_many", (PyCFunction)(void (*)(void))mod_encode_many, METH_O, "encode_many(texts) -> list[list[int]]"},
    {"encode_padded", (PyCFunction)(void (*)(void))mod_encode_padded, METH_FASTCALL,
     "encode_padded(texts, max_len, pad_id) -> (ndarray (N, max_len) uint16, raw_lengths (N,) uint16)"},
    {"decode", (PyCFunction)(void (*)(void))mod_decode, METH_FASTCALL | METH_KEYWORDS,
     "decode(ids, skip_special_tokens=True) -> str"},
    {"decode_many", (PyCFunction)(void (*)(void))mod_decode_many, METH_FASTCALL | METH_KEYWORDS,
     "decode_many(id_lists, skip_special_tokens=True) -> list[str]"},
    {"token_to_id", (PyCFunction)(void (*)(void))mod_token_to_id, METH_O, "token_to_id(token) -> int | None"},
    {"set_num_threads", (PyCFunction)(void (*)(void))mod_set_num_threads, METH_O, "set_num_threads(n)"},
    {"num_threads", (PyCFunction)(void (*)(void))mod_num_threads, METH_NOARGS, "num_threads() -> int"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,
    "fast",
    "Frozen C tokenizer for the Odin byte-level BPE vocabulary.",
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit_fast(void) {
    PyObject *m = PyModule_Create(&moddef);
    if (!m) return NULL;
    if (_import_array() < 0) {
        Py_DECREF(m);
        return NULL;
    }
    PyModule_AddIntConstant(m, "VOCAB_SIZE", VOCAB_SIZE);
    PyModule_AddStringConstant(m, "VOCAB_CHECKSUM", VOCAB_CHECKSUM);
    long nc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nc < 1) nc = 1;
    if (nc > 8) nc = 8;
    g_nthreads = (int)nc;
    return m;
}
