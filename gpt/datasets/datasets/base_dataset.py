import json  # Importing JSON module for JSON handling
from typing import Iterable  # Importing Iterable for type hinting

from torch.utils.data import Dataset, ConcatDataset  # Importing Dataset and ConcatDataset from torch.utils.data
from torch.utils.data.dataloader import default_collate  # Importing default_collate from torch.utils.data.dataloader

class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        Base dataset class.

        Args:
            vis_processor: Visual processor for image processing.
            text_processor: Text processor for text processing.
            vis_root (str): Root directory of images.
            ann_paths (list): List of paths to annotation files.
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            ann = json.load(open(ann_path, "r"))
            if isinstance(ann, dict):
                self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
            else:
                self.annotation.extend(json.load(open(ann_path, "r")))
    
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        """
        Method to get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.annotation)

    def collater(self, samples):
        """
        Method to collate samples.

        Args:
            samples: List of samples to collate.

        Returns:
            dict: Collated samples.
        """
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        """
        Method to set processors.

        Args:
            vis_processor: Visual processor for image processing.
            text_processor: Text processor for text processing.
        """
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        """
        Method to add instance IDs to annotations.

        Args:
            key (str): Key to use for instance IDs.
        """
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        """
        Concatenated dataset class.

        Args:
            datasets (Iterable): Iterable of datasets to concatenate.
        """
        super().__init__(datasets)

    def collater(self, samples):
        """
        Method to collate samples.

        Args:
            samples: List of samples to collate.

        Returns:
            dict: Collated samples.
        """
        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
