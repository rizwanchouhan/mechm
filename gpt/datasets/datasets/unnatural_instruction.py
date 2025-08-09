import os  # Importing os module for operating system related functionalities
import json  # Importing json module for JSON related operations
import pickle  # Importing pickle module for serialization and deserialization of Python objects
import random  # Importing random module for generating random numbers
import time  # Importing time module for time-related operations
import itertools  # Importing itertools module for creating iterators for efficient looping

import numpy as np  # Importing numpy module for numerical operations
from PIL import Image  # Importing Image class from PIL module for image processing
import skimage.io as io  # Importing io module from skimage for image input/output
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot module for plotting
from matplotlib.collections import PatchCollection  # Importing PatchCollection class from matplotlib.collections module
from matplotlib.patches import Polygon, Rectangle  # Importing Polygon and Rectangle classes from matplotlib.patches module
from torch.utils.data import Dataset  # Importing Dataset class from torch.utils.data module
import webdataset as wds  # Importing webdataset module for handling dataset

from gpt.datasets.datasets.base_dataset import BaseDataset  # Importing BaseDataset class from base_dataset module
from gpt.datasets.datasets.caption_datasets import CaptionDataset  # Importing CaptionDataset class from caption_datasets module


class UnnaturalDataset(Dataset):
    """
    Dataset class for handling unnatural datasets.
    """

    def __init__(self, text_processor, ann_path):
        """
        Initialize UnnaturalDataset.

        Args:
            text_processor: Processor for text inputs.
            ann_path (str): Path to the annotation file.
        """
        self.text_processor = text_processor  # Processor for text inputs

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)  # Load data from JSON file

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.ann)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing the sample data.
        """
        info = self.ann[index]["instances"][0]  # Get information about the sample
        instruction = info["instruction_with_input"]  # Get instruction with input
        constraints = info["constraints"]  # Get constraints
        answer = info["output"]  # Get output answer

        if constraints is not None:
            instruction = instruction + " " + constraints  # Add constraints to the instruction if present

        return {
            "instruction_input": self.text_processor(instruction),  # Processed instruction
            "answer": self.text_processor(answer),  # Processed answer
        }
