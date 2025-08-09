import os  # Module provides a portable way of using operating system dependent functionality
import re  # Module provides support for regular expressions
import json  # Module to work with JSON data
import argparse  # Module for parsing command-line arguments
from collections import defaultdict  # Default dictionary implementation
import random  # Module to generate random numbers
import numpy as np  # Numerical computing library
from PIL import Image  # Python Imaging Library for image processing
from tqdm import tqdm  # Progress bar library
import torch  # PyTorch deep learning framework
from torch.utils.data import DataLoader  # DataLoader for managing datasets
from gpt.common.config import Config  # Configuration class
from gpt.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU  # Evaluation utilities
from gpt.conversation.conversation import CONV_VISION_minigptv2  # Conversation model
from gpt.common.registry import registry  # Registry for dynamically loading classes
from gpt.datasets.datasets.first_face import MECHMEmotionDataset  # Dataset class

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Metrics for evaluation

def list_of_str(arg):
    """Custom argument type for converting comma-separated strings to a list of strings."""
    return list(map(str, arg.split(',')))

parser = eval_parser()  # Initialize argument parser
parser.add_argument("--dataset", type=list_of_str, default='MECHM_emotion_caption', help="dataset to evaluate")  # Argument for dataset selection
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")  # Argument for resolution
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")  # Argument for resampling
args = parser.parse_args()  # Parse command line arguments

cfg = Config(args)  # Create configuration object

model, vis_processor = init_model(args)  # Initialize model and visual processor

model.eval()  # Set model to evaluation mode
CONV_VISION = CONV_VISION_minigptv2  # Conversation model
conv_temp = CONV_VISION.copy()  # Copy conversation model
conv_temp.system = ""  # Set system to empty string

save_path = cfg.run_cfg.save_path  # Save path from configuration

text_processor_cfg = cfg.datasets_cfg.MECHM_emotion_caption.text_processor.train  # Text processor configuration
text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)  # Initialize text processor
vis_processor_cfg = cfg.datasets_cfg.MECHM_emotion_caption.vis_processor.train  # Visual processor configuration
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)  # Initialize visual processor

print(args.dataset)  # Print selected dataset
if 'MECHM_emotion_caption' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["MECHM_emotion_caption"]["eval_file_path"]  # Evaluation file path
    img_path = cfg.evaluation_datasets_cfg["MECHM_emotion_caption"]["img_path"]  # Image path
    batch_size = cfg.evaluation_datasets_cfg["MECHM_emotion_caption"]["batch_size"]  # Batch size
    max_new_tokens = cfg.evaluation_datasets_cfg["MECHM_emotion_caption"]["max_new_tokens"]  # Maximum new tokens
    print(eval_file_path)  # Print evaluation file path
    print(img_path)  # Print image path
    print(batch_size)  # Print batch size
    print(max_new_tokens)  # Print maximum new tokens

    data = MECHMEmotionDataset(vis_processor, text_processor, img_path, eval_file_path)  # Initialize dataset
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)  # Create DataLoader

    targets_list = []  # List to store targets
    answers_list = []  # List to store model predictions
    for batch in eval_dataloader:
        images = batch['image']  # Get images from batch
        instruction_input = batch['instruction_input']  # Get instruction inputs from batch
        targets = batch['answer']  # Get targets from batch

        texts = prepare_texts(instruction_input, conv_temp)  # Prepare texts for input
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)  # Generate model predictions
        for i in range(len(answers)):
            if answers[i] not in ['neutral', 'sadness', 'fear', 'disgust', 'surprise', 'anger', 'joy']:
                answers[i] = 'neutral'  # If prediction is not in predefined emotions, set it to neutral
        targets_list.extend(targets)  # Add targets to list
        answers_list.extend(answers)  # Add predictions to list

    # Calculate evaluation metrics
    accuracy = accuracy_score(targets_list, answers_list)
    precision = precision_score(targets_list, answers_list, average='weighted')
    recall = recall_score(targets_list, answers_list, average='weighted')
    f1 = f1_score(targets_list, answers_list, average='weighted')

    print("Accuracy:", accuracy)  # Print accuracy
    print("Precision:", precision)  # Print precision
    print("Recall:", recall)  # Print recall
    print("F1 Score:", f1)  # Print F1 score

    confusion_mat = confusion_matrix(targets_list, answers_list)  # Compute confusion matrix
    print(confusion_mat)  # Print confusion matrix
