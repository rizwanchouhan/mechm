# Path to the file containing predicted causes
submit_cause_file_path = "outcomes/submit_all_cause_ck6_wd5_now-n_w-e.txt"
# Load and parse predicted causes into a dictionary
submit_cause_emo_list = [x.strip().split('\t') for x in open(submit_cause_file_path)]
submit_cause_emo_dict = {item[0]: item[1] for item in submit_cause_emo_list}

# Path to the file containing predicted emotions
submit_pred_file_path = "outcomes/submit_now-n_w-e.txt"
# Load and parse predicted emotions into a dictionary
submit_pred_emo_list = [x.strip().split(' ') for x in open(submit_pred_file_path)]
submit_pred_emo_dict = {item[0]: item[1] for item in submit_pred_emo_list}

import difflib  # Module for computing string similarity
# Function to calculate similarity between two strings
def calculate_similarity(str1, str2):
    similarity_ratio = difflib.SequenceMatcher(None, str1, str2).ratio()
    return similarity_ratio

# Function to find the most similar answer from a list given a test string
def find_most_similar_answer(answer_list, test_str):
    max_similarity = 0
    most_similar_answer = None

    max_similarity_idx = -1
    for i in range(len(answer_list)):
        similarity = calculate_similarity(answer_list[i], test_str)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_answer = answer_list[i]
            max_similarity_idx = i

    return most_similar_answer, max_similarity_idx + 1

import json  # JSON module for working with JSON data
submit_json_file_path = "Subtask_2_test.json"  # Path to the submission JSON file

# Load and parse submission data from JSON file
with open(submit_json_file_path, "r") as file:
    submit_data = json.load(file)

# Dictionary mapping emotion indices to emotion labels
idx_emotion = dict(zip(range(7), ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']))

# Iterate through each conversation in the submission data
for conv in submit_data:
    utt_submit_list = []  # List to store utterances
    conv_pred_pairs_list = []  # List to store predicted emotion-cause pairs
    # Iterate through each utterance in the conversation
    for utt in conv['conversation']:
        # Extract utterance information
        utterance_name = utt['video_name'].split('.')[0]
        utterance_ID = int(utt['utterance_ID'])
        utt_submit_list.append(utt['text'])  # Add utterance text to list
        
        # Get predicted emotion index
        pred_emotion_idx = int(submit_pred_emo_dict[utterance_name])
        pred_emotion = idx_emotion[pred_emotion_idx]  # Map index to emotion label
        if pred_emotion_idx == 0:  # Skip if emotion is neutral
            continue
        
        # Get predicted cause string
        pred_cause_str = submit_cause_emo_dict[utterance_name]
        # Find the most similar answer from the utterance list
        pred_cause, pred_cause_idx = find_most_similar_answer(utt_submit_list, pred_cause_str) 
        if pred_cause_idx == 0:  # If no similar answer found, use utterance ID as cause index
            pred_cause_idx = utterance_ID
        if utterance_ID != pred_cause_idx:  # If cause index is different from utterance ID, create emotion-cause pair
            utt_emo = str(utterance_ID) + '_' + pred_emotion
            utt_cause = str(utterance_ID) 
            pair = [utt_emo, utt_cause]
            conv_pred_pairs_list.append(pair)  # Add pair to list
            print(utterance_ID, pred_emotion, pred_cause_idx, pair)
        
        # Create emotion-cause pair
        utt_emo = str(utterance_ID) + '_' + pred_emotion
        utt_cause = str(pred_cause_idx) 
        pair = [utt_emo, utt_cause]
        conv_pred_pairs_list.append(pair)  # Add pair to list
    conv['emotion-cause_pairs'] = conv_pred_pairs_list  # Add emotion-cause pairs to conversation

submit_json_str = json.dumps(submit_data, indent=4)  # Convert submission data to JSON string

save_name = "outcomes/submit_all_cause_ck6_wd5_now-n_w-e.json"  # File path to save JSON file
# Write JSON string to file
with open(save_name, "w") as file:
    file.write(submit_json_str)
