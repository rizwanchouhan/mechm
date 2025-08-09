import os  # Importing os module for operating system functionalities
import json  # Importing json module for JSON manipulation

from PIL import Image  # Importing Image module from PIL for image processing

from gpt.datasets.datasets.vqa_datasets import VQADataset  # Importing VQADataset from vqa_datasets module

from collections import OrderedDict  # Importing OrderedDict from collections module
import random  # Importing random module for random sampling

class __DisplMixin:
    """
    A mixin class for displaying item information.
    """

    def displ_item(self, index):
        """
        Display item information at the specified index.

        Args:
            index (int): Index of the item.

        Returns:
            OrderedDict: Ordered dictionary containing item information.
        """
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )

class GQADataset(VQADataset, __DisplMixin):
    """
    Dataset class for GQA (Visual Question Answering) tasks.
    """

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        Initialize GQADataset.

        Args:
            vis_processor: Processor for visual inputs.
            text_processor: Processor for text inputs.
            vis_root (str): Root directory of images.
            ann_paths (list): List of annotation file paths.
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.instruction_pool = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

    def __getitem__(self, index):
        """
        Get item at the specified index.

        Args:
            index (int): Index of the item.

        Returns:
            dict: Dictionary containing item information.
        """
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        answers = self.text_processor(ann["answer"])

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answers,
        }
