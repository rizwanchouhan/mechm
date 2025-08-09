import os  # Importing os module for operating system related functionalities
import json  # Importing json module for JSON related operations
import pickle  # Importing pickle module for serialization and deserialization of Python objects
import random  # Importing random module for generating random numbers
import time  # Importing time module for time-related operations
import itertools  # Importing itertools module for creating iterators for efficient looping

import numpy as np  # Importing numpy module for numerical operations
from PIL import Image  # Importing Image class from PIL module for image processing
from torch.utils.data import Dataset  # Importing Dataset class from torch.utils.data module
from visual_genome import local  # Importing local module from visual_genome for accessing local Visual Genome data

class ReferVisualGenomeDataset(Dataset):
    """
    Dataset class for handling ReferIt Visual Genome dataset.
    """

    def __init__(self, vis_processor, text_processor, data_dir):
        """
        Initialize ReferVisualGenomeDataset.

        Args:
            vis_processor: Processor for visual inputs.
            text_processor: Processor for text inputs.
            data_dir (str): Directory containing Visual Genome data.
        """
        self.data_dir = data_dir  # Directory containing Visual Genome data

        self.vis_processor = vis_processor  # Processor for visual inputs
        self.text_processor = text_processor  # Processor for text inputs

        # Retrieve all region descriptions from the Visual Genome data
        all_regions = local.get_all_region_descriptions(self.data_dir)
        # Flatten the list of lists into a single list
        all_regions = [region for regions in all_regions for region in regions]

        # Filter regions based on a maximum size of 16384 pixels (following OFA practice)
        self.regions = [region for region in all_regions if region.width * region.height < 16384]

        # Define a pool of instruction templates for referring expressions
        self.instruction_pool = [
            "[refer] {}",
            "[refer] give me the location of {}",
            "[refer] where is {} ?",
            "[refer] from this image, tell me the location of {}",
            "[refer] the location of {} is",
            "[refer] could you tell me the location for {} ?",
            "[refer] where can I locate the {} ?",
        ]

    def __len__(self):
        """
        Get the total number of regions in the dataset.

        Returns:
            int: Total number of regions.
        """
        return len(self.regions)

    def preprocess(self, index):
        """
        Preprocess a region at the given index.

        Args:
            index (int): Index of the region to preprocess.

        Returns:
            dict: Preprocessed data for the region.
        """
        region = self.regions[index]  # Get information about the region
        image_file = region.image.url.split('/')[-2:]  # Extract image file path from the URL
        image_path = os.path.join(self.data_dir, *image_file)  # Construct full image path
        image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB format
        image_orig_size = image.size  # Get original image size
        image = self.vis_processor(image)  # Apply visual processor to the image
        image_new_size = [100, 100]  # Define new image size (for normalization)

        sample_sentence = region.phrase  # Get the phrase describing the region
        refer_sentence = self.text_processor(sample_sentence)  # Process the phrase with the text processor

        bbox = [region.x, region.y, region.width, region.height]  # Get bounding box coordinates
        bbox = [
            bbox[0] / image_orig_size[0] * image_new_size[0],  # Normalize bounding box coordinates
            bbox[1] / image_orig_size[1] * image_new_size[1],
            (bbox[0] + bbox[2]) / image_orig_size[0] * image_new_size[0],
            (bbox[1] + bbox[3]) / image_orig_size[1] * image_new_size[1]
        ]
        bbox = [int(x) for x in bbox]  # Convert bounding box coordinates to integers
        bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)  # Format bounding box as a string

        return {
            "image": image,  # Preprocessed image
            "refer_sentence": refer_sentence,  # Preprocessed reference sentence
            "bbox": bbox,  # Bounding box coordinates as a string
            "image_id": region.image.id,  # ID of the image containing the region
        }

    def __getitem__(self, index):
        """
        Get a sample from the dataset at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing the sample data.
        """
        data = self.preprocess(index)  # Preprocess the region at the given index
        instruction = random.choice(self.instruction_pool).format(data['refer_sentence'])  # Choose a random instruction template and format it with the reference sentence
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)  # Add HTML tags to the instruction

        return {
            "image": data['image'],  # Preprocessed image
            "instruction_input": instruction,  # Instruction for the sample
            "answer": data['bbox'],  # Bounding box coordinates as the answer
            "image_id": data['image_id'],  # ID of the image containing the region
        }
