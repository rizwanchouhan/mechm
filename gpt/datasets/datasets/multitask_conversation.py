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


class MultiTaskConversationDataset(Dataset):
    """
    Dataset class for multi-task conversation data.
    """

    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        Initialize MultiTaskConversationDataset.

        Args:
            vis_processor: Processor for visual inputs.
            text_processor: Processor for text inputs.
            vis_root (str): Root directory of images (e.g., coco/images/).
            ann_path (str): Path to the annotation file.
        """
        self.vis_root = vis_root  # Root directory of images
        self.vis_processor = vis_processor  # Processor for visual inputs
        self.text_processor = text_processor  # Processor for text inputs

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)  # Load annotations from JSON file

        self.connect_sym = "!@#"  # Symbol used to connect questions and answers in the conversation

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
        info = self.ann[index]  # Get information about the sample

        image_file = 'COCO_train2014_{}.jpg'.format(info['id'])  # Image file name
        image_path = os.path.join(self.vis_root, image_file)  # Full path to the image
        image = Image.open(image_path).convert("RGB")  # Open and convert the image to RGB format
        image = self.vis_processor(image)  # Process the image

        first_instruction = info['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()
        first_instruction = '<Img><ImageHere></Img> {} '.format(first_instruction)  # Format first instruction

        questions = [first_instruction]  # List to store questions
        answers = []  # List to store answers

        # Iterate through the conversations
        for i, item in enumerate(info["conversations"][1:]):
            if i % 2 == 0:  # Assistant's turn
                assistant_answer = item["value"]  # Get assistant's answer
                answers.append(assistant_answer)  # Add answer to the list
            else:  # Human's turn
                human_instruction = item["value"] + " "  # Get human's instruction
                questions.append(human_instruction)  # Add instruction to the list

        questions = self.connect_sym.join(questions)  # Join questions using connect_sym
        answers = self.connect_sym.join(answers)  # Join answers using connect_sym

        return {
            "image": image,  # Image data
            "conv_q": questions,  # Concatenated conversation questions
            'conv_a': answers,  # Concatenated conversation answers
            "image_id": info['id'],  # Image ID
            "connect_sym": self.connect_sym  # Symbol used to connect questions and answers
        }
