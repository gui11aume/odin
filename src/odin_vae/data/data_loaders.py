"""Data loaders with shard-level checkpoint synchronization."""

from __future__ import annotations

from collections.abc import Callable

import torch


class DataLoaderWithAutoCheckpoint(torch.utils.data.DataLoader):
    """DataLoader with automatic checkpointing and epoch synchronization.

    Assumes that the dataset has:

    - a ``state_dict()`` method (as provided by ``GrandWebDataset``),
    - a ``_set_shared_progress_from_main(processed_epochs, processed_shards)``
      method (forwarded to ``GrandShardList``).

    With the provided ``progress_fn`` callable, the DataLoader resolves
    ``(processed_epochs, processed_shards)`` in the main process at the
    beginning of every ``__iter__()`` invocation and propagates these values
    to the dataset shared state before workers start fetching.
    """

    def __init__(
        self,
        *args,
        progress_fn: Callable[[], tuple[int, int]],
        **kwargs,
    ):
        """Initialize the DataLoaderWithAutoCheckpoint.

        Args:
            *args: Positional arguments forwarded to ``DataLoader``.
            progress_fn: Callable returning ``(processed_epochs,
                processed_shards)``. The dataset must implement
                ``_set_shared_progress_from_main(processed_epochs,
                processed_shards)``.
            **kwargs: Keyword arguments forwarded to ``DataLoader``.
        """
        super().__init__(*args, **kwargs)
        self._progress_fn = progress_fn

    def __iter__(self):
        """Create a new iterator, synchronizing progress first.

        This method runs in the main process. It calls the progress function
        to get the current progress and then calls
        ``_set_shared_progress_from_main()`` on the dataset to update the
        shared progress values. The workers have access to the shared values
        and use them to create the shard iterators.
        """
        sync_progress = getattr(self.dataset, "_set_shared_progress_from_main", None)
        if not callable(sync_progress):
            raise TypeError(
                "DataLoaderWithAutoCheckpoint requires a dataset with a `_set_shared_progress_from_main()` method."
            )
        processed_epochs, processed_shards = self._progress_fn()
        sync_progress(processed_epochs, processed_shards)
        return super().__iter__()

    def state_dict(self) -> dict:
        """Return state to be saved in a checkpoint."""
        state_dict = getattr(self.dataset, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("DataLoaderWithAutoCheckpoint requires a dataset with a state_dict() method.")
        return state_dict()

    def load_state_dict(self, checkpoint: dict) -> None:  # noqa: ARG002
        """Defining this method is sufficient for automatic checkpointing."""
        return
