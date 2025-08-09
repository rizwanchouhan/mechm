import time  # Importing time module for time-related functionalities
import random  # Importing random module for random sampling
import torch  # Importing torch module for PyTorch functionalities
from gpt.datasets.data_utils import move_to_cuda  # Importing move_to_cuda function from data_utils module
from torch.utils.data import DataLoader  # Importing DataLoader from torch.utils.data for loading data

class MultiIterLoader:
    """
    A simple wrapper for iterating over multiple iterators.
    """

    def __init__(self, loaders, ratios=None):
        """
        Initialize MultiIterLoader.

        Args:
            loaders (List[Loader]): List of Iterator loaders.
            ratios (List[float]): List of ratios to sample from each loader. If None, all loaders are sampled uniformly.
        """
        # assert all loaders has __next__ method
        for loader in loaders:
            assert hasattr(
                loader, "__next__"
            ), "Loader {} has no __next__ method.".format(loader)

        if ratios is None:
            ratios = [1.0] * len(loaders)
        else:
            assert len(ratios) == len(loaders)
            ratios = [float(ratio) / sum(ratios) for ratio in ratios]

        self.loaders = loaders
        self.ratios = ratios

    def __next__(self):
        """
        Get the next item from the loaders.

        Returns:
            Any: The next item.
        """
        # random sample from each loader by ratio
        loader_idx = random.choices(range(len(self.loaders)), self.ratios, k=1)[0]
        return next(self.loaders[loader_idx])


class PrefetchLoader:
    """
    Modified from https://github.com/ChenRocks/UNITER.

    Overlap compute and cuda data transfer.
    """

    def __init__(self, loader):
        """
        Initialize PrefetchLoader.

        Args:
            loader: DataLoader object.
        """
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        """
        Iterate over the DataLoader object.

        Returns:
            Any: The next item.
        """
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            is_tuple = isinstance(batch, tuple)
            if is_tuple:
                task, batch = batch

            if is_tuple:
                yield task, batch
            else:
                yield batch
            batch = self.next(loader_it)

    def __len__(self):
        """
        Get the length of the DataLoader object.

        Returns:
            int: Length of the DataLoader.
        """
        return len(self.loader)

    def preload(self, it):
        """
        Preload the data from DataLoader to CUDA.

        Args:
            it: Iterator object.
        """
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)

    def next(self, it):
        """
        Get the next batch of data.

        Args:
            it: Iterator object.

        Returns:
            Any: The next batch of data.
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        """
        Get an attribute of the DataLoader object.

        Args:
            name: Name of the attribute.

        Returns:
            Any: The attribute of the DataLoader.
        """
        method = self.loader.__getattribute__(name)
        return method


def record_cuda_stream(batch):
    """
    Record the CUDA stream for a batch.

    Args:
        batch: The batch of data.
    """
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass


class IterLoader:

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        """
        Initialize IterLoader.

        Args:
            dataloader (DataLoader): DataLoader object.
            use_distributed (bool): Whether to use distributed training.
        """
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        """
        Get the current epoch.

        Returns:
            int: Current epoch.
        """
        return self._epoch

    def __next__(self):
        """
        Get the next batch of data.

        Returns:
            Any: The next batch of data.
        """
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        """
        Iterate over the DataLoader object.

        Returns:
            IterLoader: The current instance.
        """
        return self

    def __len__(self):
        """
        Get the length of the DataLoader object.

        Returns:
            int: Length of the DataLoader.
        """
        return len(self._dataloader)
