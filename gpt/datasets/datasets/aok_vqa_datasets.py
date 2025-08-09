from collections import OrderedDict  # Importing OrderedDict for ordered dictionaries
import json  # Importing JSON module for JSON handling
import os  # Importing os module for filesystem operations
import random  # Importing random module for random sampling
import torch  # Importing PyTorch library

from PIL import Image  # Importing Image module from PIL library

from gpt.datasets.datasets.vqa_datasets import VQADataset  # Importing VQADataset class for VQA dataset handling
# , VQAEvalDataset

class __DisplMixin:
    """
    Mixin class for display related methods.
    """
    def displ_item(self, index):
        """
        Method to display item information.

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
                "direct_answers": "; ".join(ann["direct_answers"]),
                "choices": "; ".join(ann["choices"]),
                "correct_choice": ann["choices"][ann["correct_choice_idx"]],
                "image": sample["image"],
            }
        )


class AOKVQADataset(VQADataset, __DisplMixin):
    """
    Dataset class for AOK VQA dataset.

    Args:
        vis_processor: Visual processor for image processing.
        text_processor: Text processor for text processing.
        vis_root (str): Path to the directory containing images.
        ann_paths (list): List of paths to annotation files.
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation

    def get_data(self, index):
        """
        Method to get data for a given index.

        Args:
            index (int): Index of the data.

        Returns:
            dict: Dictionary containing data for the given index.
        """
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_key = "direct_answers"

        answer_weight = {}
        for answer in ann[answer_key]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann[answer_key])
            else:
                answer_weight[answer] = 1 / len(ann[answer_key])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights

        return {
            "image": image,
            "question": question,
            "answer": answer,
        }

    def __getitem__(self, index):
        """
        Method to get item for a given index.

        Args:
            index (int): Index of the item.

        Returns:
            dict: Dictionary containing item for the given index.
        """
        data = self.get_data(index)
        question = self.text_processor(data["question"])
        instruction = random.choice(self.instruction_pool).format(question)

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        answer = self.text_processor(data['answer'])

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": answer,
        }


class AOKVQGDataset(AOKVQADataset):
    """
    Dataset class for AOK VQG dataset, inherited from AOKVQADataset.

    Args:
        vis_processor: Visual processor for image processing.
        text_processor: Text processor for text processing.
        vis_root (str): Path to the directory containing images.
        ann_paths (list): List of paths to annotation files.
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.instruction_pool = [
            'Given the image, generate a question whose answer is: {}',
            'Based on the image, provide a question with the answer: {}',
            'Given the visual representation, create a question for which the answer is "{}"',
            'From the image provided, craft a question that leads to the reply: {}',
            'Considering the picture, come up with a question where the answer is: {}',
            'Taking the image into account, generate an question that has the answer: {}'
        ]

    def __getitem__(self, index):
        """
        Method to get item for a given index.

        Args:
            index (int): Index of the item.

        Returns:
            dict: Dictionary containing item for the given index.
        """
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['answer'])

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": data['question'],
        }
