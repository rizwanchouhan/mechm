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


class OCRVQADataset(Dataset):
    """
    Dataset class for OCR-based Visual Question Answering.
    """

    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        Initialize OCRVQADataset.

        Args:
            vis_processor: Processor for visual inputs.
            text_processor: Processor for text inputs.
            vis_root (str): Root directory of images (e.g., coco/images/).
            ann_path (str): Path to the annotation file.
        """
        self.vis_root = vis_root  # Root directory of images
        self.vis_processor = vis_processor  # Processor for visual inputs
        self.text_processor = text_processor  # Processor for text inputs
        self.data = self.create_data(ann_path)  # Processed data
        self.instruction_pool = [
            "[vqa] {}",  # Instruction format for VQA
            "[vqa] Based on the image, respond to this question with a short answer: {}"  # Instruction format for VQA
        ]

    def create_data(self, ann_path):
        """
        Create processed data from the annotation file.

        Args:
            ann_path (str): Path to the annotation file.

        Returns:
            list: Processed data containing questions, answers, image paths, image IDs, titles, and genres.
        """
        processed_data = []  # List to store processed data
        with open(ann_path, 'r') as f:
            data = json.load(f)  # Load data from JSON file
        for k in data.keys():
            if data[k]['split'] != 1:
                continue  # Skip samples that are not for training
            ext = os.path.splitext(data[k]['imageURL'])[1]  # Get file extension
            imageFile = k + ext  # Image file name
            assert len(data[k]['questions']) == len(data[k]['answers'])  # Ensure questions and answers have same length
            for q, a in zip(data[k]['questions'], data[k]['answers']):
                processed_data.append(
                    {'question': q,  # Question
                     'answer': a,  # Answer
                     'image_path': imageFile,  # Image path
                     'image_id': k,  # Image ID
                     'title': data[k]['title'],  # Title
                     'genre': data[k]['genre'],  # Genre
                     }
                )
        return processed_data  # Return processed data

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing the sample data.
        """
        sample = self.data[index]  # Get sample data
        image = Image.open(os.path.join(self.vis_root, sample['image_path'])).convert("RGB")  # Open and convert image
        image = self.vis_processor(image)  # Process image
        question = self.text_processor(sample["question"])  # Process question
        answer = self.text_processor(sample["answer"])  # Process answer

        instruction = random.choice(self.instruction_pool).format(question)  # Choose random instruction format
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)  # Format instruction

        return {
            "image": image,  # Image data
            "instruction_input": instruction,  # Instruction for VQA
            "answer": answer,  # Answer for VQA
            "image_id": sample['image_id']  # Image ID
        }
