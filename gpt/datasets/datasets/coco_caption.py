import os  # Importing os module for operating system functionalities
import json  # Importing json module for JSON handling
import torch  # Importing torch for deep learning functionalities
import numpy as np  # Importing numpy for numerical computations

from PIL import Image  # Importing Image from PIL for image processing
from PIL import ImageFile  # Importing ImageFile from PIL for handling image file loading issues

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Setting PIL to load truncated images

from gpt.datasets.datasets.caption_datasets import COCOCaptionDataset, CaptionEvalDataset  # Importing COCOCaptionDataset and CaptionEvalDataset classes from caption_datasets module

COCOCapDataset = COCOCaptionDataset  # Assigning COCOCaptionDataset to COCOCapDataset

class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        Dataset class for evaluating COCO captions.

        Args:
            vis_processor: Visual processor for image processing.
            text_processor: Text processor for text processing.
            vis_root (string): Root directory of images.
            ann_paths (list): List of annotation paths.
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        """
        Method to get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: Retrieved item.
        """
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }

class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        Dataset class for evaluating captions when no captions are available.

        Args:
            vis_processor: Visual processor for image processing.
            text_processor: Text processor for text processing.
            vis_root (string): Root directory of images.
            ann_paths (list): List of annotation paths.
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        """
        Method to get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: Retrieved item.
        """
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["img_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }

class RefCOCOEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        """
        Dataset class for evaluating reference COCO data.

        Args:
            loaded_data: Loaded data for evaluation.
            vis_processor: Visual processor for image processing.
            root_path (string): Root directory path.
        """
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        """
        Method to get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        """
        Method to get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Retrieved item.
        """
        data = self.loaded_data[idx]
        img_id = data['img_id']
        sent = data['sents']
        image_path = os.path.join(self.root_path, f'{img_id[:27]}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[refer] give me the location of {sent}"
        return image, question, img_id

class EvalCaptionData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        """
        Dataset class for evaluating caption data.

        Args:
            loaded_data: Loaded data for evaluation.
            vis_processor: Visual processor for image processing.
            root_path (string): Root directory path.
        """
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        ann = dict()
        for item in self.loaded_data:
            image_id = item['image_id']
            ann[image_id] = item['image']
        self.ann = [{'image_id':image_id, 'image': ann[image_id]} for image_id in ann]

    def __len__(self):
        """
        Method to get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.ann)
    
    def __getitem__(self, idx):
        """
        Method to get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Retrieved item.
        """
        data = self.ann[idx]
        image_id = data['image_id']
        img_file = data['image'].split('/')[-1]
        image_path = os.path.join(self.root_path, img_file)
        image = Image.open(image_path).convert('RGB')
            
        image = self.vis_processor(image)
        question = f"[caption] please describe this image?"
        return image, question, image_id
