import gzip
import sys

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import Digits, Whitespace

# https://huggingface.co/docs/tokenizers/python/latest/pipeline.html

VOCAB_SIZE = 8192
LIMIT_ALPHABET = 4096


if __name__ == "__main__":
    data_path = sys.argv[1]

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            Whitespace(),
            Digits(individual_digits=True),
        ]
    )

    from tokenizers.trainers import WordPieceTrainer

    trainer = WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        limit_alphabet=LIMIT_ALPHABET,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    with gzip.open(data_path, "rt") as f:
        tokenizer.train_from_iterator(f, trainer=trainer)

    tokenizer.save("odin_tokenizer.json")
