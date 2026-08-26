"""Grand WebDataset: resumable, DDP-safe webdataset stream over local shards.

This module provides resumable iterators over local webdataset shards that
can be restarted from a checkpoint without losing determinism. (Design
ported from the ranktriever-arena data pipeline; the "Ensemble" prefix was
dropped and the shards are local files rather than S3 objects.)

Background
~~~~~~~~~~

A DataLoader is a dataset and a collator. When the DataLoader is requested
to send data, an iterator is created with ``DataLoader.__iter__()``, which
calls ``dataset.__iter__()`` to create an iterator over the dataset in
order to pass items to the collator via ``next()``. The dataset iterator is
thus what determines the schedule of the data that reaches the collator.

It is important to control this iterator so that:

- items are dispatched according to the desired schedule,
- the flow of data can scale with available resources,
- the flow of data can be restarted if it is interrupted.

In the forking model for multiprocessing (used here), workers receive a
copy-on-write snapshot of the dataset object at fork time: plain Python
attributes are not shared between the main process and the workers, and
``DataLoader.__iter__()`` is called after forking. Therefore the epoch and
the number of processed shards are stored as ``multiprocessing.Value``
integers backed by a shared-memory segment that every process can access.
The main process is the sole writer: ``DataLoaderWithAutoCheckpoint`` has a
``progress_fn`` callable (provided by the data module, backed by the
Lightning Trainer) that it invokes in the main process at the beginning of
every ``__iter__()``, and it then calls ``_set_shared_progress_from_main()``
on the dataset. Workers read the shared values in ``GrandShardList.__iter__``.

Because ``GrandShardList.__iter__()`` is only (re)called when a worker is
created, the workers must be instantiated with ``persistent_workers=False``
so that the dataset iterator is refreshed at the start of every epoch (or,
alternatively, the number of shards must be a multiple of the number of
workers).

Shard order per epoch is a full permutation without replacement, seeded by
``seed + epoch``; every rank computes the identical permutation before
``split_by_node``/``split_by_worker`` take their strided subsequences, so
DDP shards are disjoint by construction. This design ensures that:

- the order of the shards is deterministic,
- resuming from a checkpoint is reproducible.
"""

from __future__ import annotations

import multiprocessing as mp
import random

import webdataset as wds


class GrandShardList(wds.shardlists.SimpleShardList):
    """Shard list with shared-memory training progress for resumable iteration."""

    def __init__(
        self,
        urls: str | list[str],
        seed: int | None = None,
        is_endless: bool = False,
    ):
        """Initialize the GrandShardList.

        Args:
            urls: URLs (local paths) to the webdataset shards; a brace
                pattern string is expanded.
            seed: Seed for the per-epoch shuffle (None for no shuffling).
            is_endless: Whether the dataset cycles over the shards forever
                (train) or stops after one pass (val/test).
        """
        super().__init__(urls, seed=seed)
        self.is_endless = is_endless
        self._processed_epochs = mp.Value("q", 0)
        self._processed_shards = mp.Value("q", 0)

    def _set_shared_progress_from_main(self, processed_epochs: int, processed_shards: int) -> None:
        """Write progress values in shared memory from the main process."""
        with self._processed_epochs.get_lock():
            self._processed_epochs.value = int(processed_epochs)
        with self._processed_shards.get_lock():
            self._processed_shards.value = int(processed_shards)

    def __iter__(self):
        """Iterate over the shards, honoring the shared progress values.

        The first pass starts at shard offset ``shards_processed`` (resume);
        for an endless list the epoch permutation then repeats forever and
        must be stopped externally (e.g. ``limit_train_batches``).

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        urls: list[str] = self.urls.copy()
        with self._processed_epochs.get_lock():
            epoch = int(self._processed_epochs.value)
        with self._processed_shards.get_lock():
            shards_processed = int(self._processed_shards.value)
        # Shuffle the shards if a seed is provided.
        if self.seed is not None:
            random.Random(self.seed + epoch).shuffle(urls)  # nosec: B311  # deterministic shard shuffle
        # Start from the next shard, then cycle forever from the beginning.
        for url in urls[shards_processed:]:
            yield {"url": url}
        if not self.is_endless:
            return
        while True:
            for url in urls:
                yield {"url": url}


class GrandWebDataset(wds.WebDataset):
    """WebDataset pipeline for local shards with reproducible sharding.

    Pipeline stages:

    - ``GrandShardList``: per-epoch shuffle (seeded) + resume offset.
    - ``wds.split_by_node``: strided shard assignment per DDP rank.
    - ``wds.split_by_worker``: strided shard assignment per DataLoader worker.
    - ``wds.cache.StreamingOpen``: opens shard files.
    - ``tar_file_expander``: extracts tar archives on the fly.
    - ``group_by_keys``: groups members into one sample per ``__key__``.
    - ``check_empty``: raises if no shards are found.
    """

    def __init__(
        self,
        urls: str | list[str],
        seed: int | None = None,
        is_endless: bool = False,
    ):
        """Initialize the GrandWebDataset.

        Args:
            urls: Local shard path(s); a brace pattern string (e.g.
                ``"train/shard-{000000..000019}.tar.gz"``) is accepted.
            seed: Seed for deterministic shard shuffling (None = no shuffle).
            is_endless: Whether the dataset cycles forever (train) or is
                finite (val/test).
        """
        super(wds.WebDataset, self).__init__()
        self.urls = urls if isinstance(urls, str) else list(urls)
        self.shardlist = GrandShardList(
            urls=self.urls,
            seed=seed,
            is_endless=is_endless,
        )
        self.opener = wds.cache.StreamingOpen()
        self.expander = wds.pipelinefilter(wds.tariterators.tar_file_expander)
        self.grouper = wds.pipelinefilter(wds.tariterators.group_by_keys)
        # Set up the pipeline.
        self.append(self.shardlist)
        self.append(wds.split_by_node)
        self.append(wds.split_by_worker)
        self.append(self.opener)
        self.append(self.expander())
        self.append(self.grouper())
        self.append(wds.compat.check_empty)

    def _set_shared_progress_from_main(self, processed_epochs: int, processed_shards: int) -> None:
        """Write worker-visible progress values from the main process."""
        self.shardlist._set_shared_progress_from_main(processed_epochs, processed_shards)

    def state_dict(self) -> dict:
        """Return state to be saved in a checkpoint (the seed)."""
        return {"seed": self.shardlist.seed}
