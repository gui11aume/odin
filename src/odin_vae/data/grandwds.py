"""Grand WebDataset: resumable, DDP-safe webdataset stream over local shards.

This module provides resumable iterators over local webdataset shards that
can be restarted from a checkpoint without losing determinism.

Design
~~~~~~

The shard order of an epoch is a full permutation without replacement of the
shard list, seeded by `seed + epoch`. Every rank and worker computes the
identical permutation and then `split_by_node`/`split_by_worker` take
their strided subsequences, so DDP shards are disjoint by construction.
Endless streams (train) cycle the permutation forever and are stopped
externally (`limit_train_batches`); finite streams (val/test) stop after
one pass.

Resume state is a pair of integers: `(processed_epochs, processed_shards)`.
Iteration starts at shard offset `processed_shards` of permutation
`seed + processed_epochs`. The main process is the sole writer: it sets the
progress on the dataset before the DataLoader workers are forked, and the
workers read it from the copy-on-write snapshot of the dataset they inherit
at fork. This is why the pipeline requires `multiprocessing_context="fork"`
and `persistent_workers=False` (a persistent worker would keep iterating
its original snapshot).
"""

from __future__ import annotations

import random

import webdataset as wds


class GrandShardList(wds.shardlists.SimpleShardList):
    """Shard list with training progress for resumable iteration."""

    def __init__(
        self,
        urls: str | list[str],
        seed: int | None = None,
        loop_back: bool = False,
    ):
        """Initialize the GrandShardList.

        Args:
            urls: URLs (local paths) to the webdataset shards; a brace
                pattern string is expanded.
            seed: Seed for the per-epoch shuffle (None for no shuffling).
            loop_back: Whether the dataset cycles over the shards forever
                (train) or stops after one pass (val/test).
        """
        super().__init__(urls, seed=seed)
        self.loop_back = loop_back
        self.processed_epochs = 0
        self.processed_shards = 0

    def set_progress(self, processed_epochs: int, processed_shards: int) -> None:
        """Set the resume progress (call from the main process before forking)."""
        self.processed_epochs = int(processed_epochs)
        self.processed_shards = int(processed_shards)

    def __iter__(self):
        """Iterate over the shards, honoring the progress.

        The first pass starts at shard offset `processed_shards` (resume);
        for an endless list the epoch permutation then repeats forever and
        must be stopped externally (e.g. `limit_train_batches`).

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        urls: list[str] = self.urls.copy()
        epoch, shards_processed = self.processed_epochs, self.processed_shards
        # Shuffle the shards if a seed is provided.
        if self.seed is not None:
            random.Random(self.seed + epoch).shuffle(urls)  # nosec: B311  # deterministic shard shuffle
        # One pass through the shards (skip already processed shards).
        for url in urls[shards_processed:]:
            yield {"url": url}
        if not self.loop_back:
            return
        # Loop back: cycle forever, processing shards in the same order.
        while True:
            for url in urls:
                yield {"url": url}


class GrandWebDataset(wds.WebDataset):
    """WebDataset pipeline for local shards with reproducible sharding.

    Pipeline stages:

    - `GrandShardList`: per-epoch shuffle (seeded) + resume offset.
    - `wds.split_by_node`: strided shard assignment per DDP rank.
    - `wds.split_by_worker`: strided shard assignment per DataLoader worker.
    - `wds.cache.StreamingOpen`: opens shard files.
    - `tar_file_expander`: extracts tar archives on the fly.
    - `group_by_keys`: groups members into one sample per `__key__`.
    - `check_empty`: raises if no shards are found.
    """

    def __init__(
        self,
        urls: str | list[str],
        seed: int | None = None,
        loop_back: bool = False,
    ):
        """Initialize the GrandWebDataset.

        Args:
            urls: Local shard path(s); a brace pattern string (e.g.
                `"train/shard-{000000..000019}.tar.gz"`) is accepted.
            seed: Seed for deterministic shard shuffling (None = no shuffle).
            loop_back: Whether the dataset cycles forever (train) or is
                finite (val/test).
        """
        super(wds.WebDataset, self).__init__()
        self.urls = urls if isinstance(urls, str) else list(urls)
        self.shardlist = GrandShardList(
            urls=self.urls,
            seed=seed,
            loop_back=loop_back,
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

    @property
    def seed(self) -> int | None:
        """Seed of the shard permutation (restored/validated on resume)."""
        return self.shardlist.seed

    def set_progress(self, processed_epochs: int, processed_shards: int) -> None:
        """Set the resume progress on the shard list (main process only)."""
        self.shardlist.set_progress(processed_epochs, processed_shards)

    def progress(self) -> tuple[int, int]:
        """Current resume progress `(processed_epochs, processed_shards)`."""
        return self.shardlist.processed_epochs, self.shardlist.processed_shards
