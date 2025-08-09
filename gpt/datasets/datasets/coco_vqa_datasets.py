import os  # Importing os module for operating system functionalities
import json  # Importing json module for JSON handling
import random  # Importing random module for random sampling

from PIL import Image  # Importing Image from PIL for image processing

from gpt.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset  # Importing VQADataset and VQAEvalDataset classes from vqa_datasets module

from collections import OrderedDict  # Importing OrderedDict from collections

class __DisplMixin:
    """
    Mixin class for displaying items.
    """
    def displ_item(self, index):
        """
        Method to display an item.

        Args:
            index (int): Index of the item to display.

        Returns:
            OrderedDict: Ordered dictionary representing the item.
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

class COCOVQADataset(VQADataset, __DisplMixin):
    """
    Dataset class for COCO VQA.
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        Initialize COCOVQADataset.

        Args:
            vis_processor: Visual processor for image processing.
            text_processor: Text processor for text processing.
            vis_root (string): Root directory of images.
            ann_paths (list): List of annotation paths.
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool = [
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
            dict: Data corresponding to the index.
        """
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["question_id"]

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = random.choices(answers, weights=weights, k=1)[0]  # Randomly sample an answer according to weights

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        """
        Method to get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: Retrieved item.
        """
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }

class COCOVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    """
    Dataset class for evaluating COCO VQA.
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        Initialize COCOVQAEvalDataset.

        Args:
            vis_processor: Visual processor for image processing.
            text_processor: Text processor for text processing.
            vis_root (string): Root directory of images.
            ann_paths (list): List of annotation paths.
        """
        self.instruction_pool = [
            'Question: {} Short answer:',
        ]
        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

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
        question = self.text_processor(ann["question"])

        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": image,
            'image_path': image_path,
            "question": question,
            "question_id": ann["question_id"],
            "instruction_input": instruction,
            "instance_id": ann["instance_id"],
        }
