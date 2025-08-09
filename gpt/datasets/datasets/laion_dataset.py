import webdataset as wds  # Importing webdataset module as wds for handling dataset
from gpt.datasets.datasets.base_dataset import BaseDataset  # Importing BaseDataset class from base_dataset module

class LaionDataset(BaseDataset):
    """
    Dataset class for Laion dataset.
    """

    def __init__(self, vis_processor, text_processor, location):
        """
        Initialize LaionDataset.

        Args:
            vis_processor: Processor for visual inputs.
            text_processor: Processor for text inputs.
            location (str): Location of the dataset.
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        # Define inner dataset using webdataset DataPipeline
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),  # Load resampled shards from the specified location
            wds.tarfile_to_samples(handler=wds.warn_and_continue),  # Convert tarfile to samples
            wds.shuffle(1000, handler=wds.warn_and_continue),  # Shuffle samples
            wds.decode("pilrgb", handler=wds.warn_and_continue),  # Decode images to PIL RGB format
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),  # Convert to tuple format
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),  # Apply visual processor
            wds.map(self.to_dict, handler=wds.warn_and_continue),  # Convert to dictionary format
        )

    def to_dict(self, sample):
        """
        Convert sample to dictionary format.

        Args:
            sample: Input sample.

        Returns:
            dict: Dictionary containing image and answer information.
        """
        return {
            "image": sample[0],  # Extract image from the sample
            "answer": self.text_processor(sample[1]["caption"]),  # Process and extract answer from the sample
        }
