import json
import re
from collections import defaultdict


def read_json_file(path):
    try:
        with open(path, "r") as json_file:
            json_data = json.load(json_file)
        
        return json_data
    
    except:
        return {}


def read_text_file(file_path):
    try:

        text_list = []
        with open(file_path, 'r') as file:
            for line in file:
                text_list.append(line.strip())
        
        return text_list
    
    except:
        return []


def make_target(keyword):
    target_candidate = [
        ['female', 'male'],
        ['under 44 years', 'over 44 years'],
        ]

    if keyword == 'persona_gender' or keyword == 'wino_gender':
        target_num = 0
    elif keyword == 'persona_age':
        target_num = 1
    
    group1 = target_candidate[target_num][0]
    group2 = target_candidate[target_num][1]

    return group1, group2


def get_target(keyword):
    target_candidate = [
        ['female', 'male'],
        ['under_44', 'over_44'],
        ]

    if keyword == 'persona_gender' or keyword == 'wino_gender':
        target_num = 0
    elif keyword == 'persona_age':
        target_num = 1
    
    group1 = target_candidate[target_num][0]
    group2 = target_candidate[target_num][1]

    return group1, group2


def get_model_list():
    return [
        'llama2_7b', 'llama2_7b_chat',
        'llama2_13b', 'llama2_13b_chat',
        'llama3_8b', 'llama3_8b_instruct',
        'llama3.1_8b', 'llama3.1_8b_instruct',
        'mistral_7b_v01', 'mistral_7b_plus',
        'qwen1.5_7b', 'qwen1.5_7b_chat',
        'qwen2_7b', 'qwen2_7b_instruct',        
        'gpt3.5', 'gpt4', 'gpt4o_mini'
        ]


def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries and sum their values.
    """
    merged = defaultdict(dict)
    keys = set(dict1.keys()).union(set(dict2.keys()))
    
    for key in keys:
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merged[key] = merge_dicts(dict1[key], dict2[key])
            else:
                merged[key] = dict1[key] + dict2[key]
        elif key in dict1:
            merged[key] = dict1[key]
        else:
            merged[key] = dict2[key]
    
    return dict(merged)


def divide_dict_values(data, divisor):
    """
    Recursively divide all numeric values in a dictionary (or list) by the given divisor.
    """
    if isinstance(data, dict):
        return {k: divide_dict_values(v, divisor) for k, v in data.items()}
    elif isinstance(data, list):
        return [divide_dict_values(v, divisor) for v in data]
    elif isinstance(data, (int, float)):
        return data / divisor
    else:
        return data


def extract_choice_names(choices, messages):
    if len(choices) == 3:
        pattern = r'\(1\): (.*?)\n\(2\): (.*?)\n\(3\): (.*?)\nAnswer:'
        matches = re.search(pattern, messages)

        if matches:
            val1 = matches.group(1).strip()
            val2 = matches.group(2).strip()
            val3 = matches.group(3).strip()

            return [val1, val2, val3]

    else:
        pattern = r'\(A\) (.*?)\n\(B\) (.*?)\n\(C\) (.*?)\n\(D\) (.*?)\nAnswer:'
        matches = re.search(pattern, messages)

        if matches:
            val1 = matches.group(1).strip()
            val2 = matches.group(2).strip()
            val3 = matches.group(3).strip()
            val4 = matches.group(4).strip()

            return [val1, val2, val3, val4]
        
        
def get_clean_sentences(file):

    sentences = []

    with open(file, 'r') as f:
        while True:
            line = f.readline()

            if not line:
                break

            line_clean = line.replace('[', '').replace(']', '').split(' ', 1)[1]

            sentences.append(line_clean)
    
    return sentences