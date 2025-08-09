import os  # Importing os module for operating system functionalities
from PIL import Image  # Importing Image from PIL for image processing
import webdataset as wds  # Importing webdataset as wds for handling web datasets
from gpt.datasets.datasets.base_dataset import BaseDataset  # Importing BaseDataset class from base_dataset module
from gpt.datasets.datasets.caption_datasets import CaptionDataset  # Importing CaptionDataset class from caption_datasets module

class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        """
        Dataset class for CCSBU dataset.

        Args:
            vis_processor: Visual processor for image processing.
            text_processor: Text processor for text processing.
            location (str): Location of the dataset.
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        # Creating inner dataset pipeline
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        """
        Method to convert a sample to dictionary format.

        Args:
            sample: Sample to convert.

        Returns:
            dict: Converted sample as a dictionary.
        """
        return {
            "image": sample[0],
            "answer": self.text_processor(sample[1]["caption"]),
        }

class CCSBUAlignDataset(CaptionDataset):
    def __getitem__(self, index):
        """
        Method to get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: Retrieved item.
        """
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["caption"]

        return {
            "image": image,
            "answer": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }
