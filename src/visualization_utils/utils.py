import os
import numpy as np
import pickle


def get_correct_wrong_ids(model_name, dataset_name):
    result_status_dict = {}
    result_status_dict[model_name] = {}
    print(f"\nModel: {model_name}\n")
    if "aime" in dataset_name:
        backlog_dir = f"/n/home04/yidachen/reasoning_characteristics/backlog/unfinished_thinking/aime_2024"
        unfinished_ids = set()
        unfinished_file = os.path.join(backlog_dir, f"unfinished_thoughts_{model_name}.pkl")
        if os.path.exists(unfinished_file):
            with open(unfinished_file, "rb") as f:
                unfinished_ids.update(pickle.load(f))

        result_status_dict[model_name]["unfinished"] = unfinished_ids

        wrong_ids = set()
        wrong_ids_file = os.path.join(backlog_dir, f"wrong_ids_{model_name}.pkl")
        if os.path.exists(wrong_ids_file):
            with open(wrong_ids_file, "rb") as f:
                wrong_ids.update(pickle.load(f))
                
        result_status_dict[model_name]["wrong_ids"] = wrong_ids

        correct_ids = set()
        correct_ids_file = os.path.join(backlog_dir, f"correct_ids_{model_name}.pkl")
        if os.path.exists(correct_ids_file):
            with open(correct_ids_file, "rb") as f:
                correct_ids.update(pickle.load(f))

        result_status_dict[model_name]["correct_ids"] = correct_ids

        backlog_dir = f"/n/home04/yidachen/reasoning_characteristics/backlog/unfinished_thinking/aime_2025"
        unfinished_file = os.path.join(backlog_dir, f"unfinished_thoughts_{model_name}.pkl")
        if os.path.exists(unfinished_file):
            with open(unfinished_file, "rb") as f:
                unfinished_ids.update(pickle.load(f))
        
        wrong_ids_file = os.path.join(backlog_dir, f"wrong_ids_{model_name}.pkl")
        if os.path.exists(wrong_ids_file):
            with open(wrong_ids_file, "rb") as f:
                wrong_ids.update(pickle.load(f))
                
        result_status_dict[model_name]["wrong_ids"] = wrong_ids
        
        correct_ids_file = os.path.join(backlog_dir, f"correct_ids_{model_name}.pkl")
        if os.path.exists(correct_ids_file):
            with open(correct_ids_file, "rb") as f:
                correct_ids.update(pickle.load(f))
                
        result_status_dict[model_name]["correct_ids"] = correct_ids
    else:
        backlog_dir = f"/n/home04/yidachen/reasoning_characteristics/backlog/unfinished_thinking/{dataset_name}"
        unfinished_ids = set()
        unfinished_file = os.path.join(backlog_dir, f"unfinished_thoughts_{model_name}.pkl")
        if os.path.exists(unfinished_file):
            with open(unfinished_file, "rb") as f:
                unfinished_ids.update(pickle.load(f))

        result_status_dict[model_name]["unfinished"] = unfinished_ids
        
        wrong_ids = set()
        wrong_ids_file = os.path.join(backlog_dir, f"wrong_ids_{model_name}.pkl")
        if os.path.exists(wrong_ids_file):
            with open(wrong_ids_file, "rb") as f:
                wrong_ids.update(pickle.load(f))
                
        result_status_dict[model_name]["wrong_ids"] = wrong_ids
        
        correct_ids = set()
        correct_ids_file = os.path.join(backlog_dir, f"correct_ids_{model_name}.pkl")
        if os.path.exists(correct_ids_file):
            with open(correct_ids_file, "rb") as f:
                correct_ids.update(pickle.load(f))

        result_status_dict[model_name]["correct_ids"] = correct_ids

    return correct_ids, wrong_ids, unfinished_ids, result_status_dict


def transform_parser_output(coder, type_of_code="appearance", dataset=None):
    """
    Transform the parser output into a dictionary with model names as keys and rule names as sub-keys.
    Args:
        coder (object): The coder object containing training logs and _parse_codes_occurence method.
    Returns:
        dict: A dictionary with model names as keys and dictionaries with rule names and idx values as values.
    """
    transformed_output = {}
    
    for idx in list(coder.training_logs['eval'].keys()):
        if type_of_code == "appearance":
            if 'data' in coder.training_logs['eval'][idx].keys():
                parser_output = coder._parse_codes_occurence(coder.training_logs['eval'][idx]['output_classification'], coder.training_logs['eval'][idx]['data']['labels'])
            else:
                parser_output = coder._parse_codes_occurence(coder.training_logs['eval'][idx]['output_classification'], match_label(dataset, idx))
        elif type_of_code == "decision":
            parser_output = coder._parse_decision_code(coder.training_logs['eval'][idx]['output_classification'])
            
        for rule_name, models in parser_output.items():
            if "Both model" in rule_name or "Both output" in rule_name or len(rule_name.split()) > 10 or "the" in rule_name:
                continue
                
            for model_name, exhibited in models.items():
                if exhibited:
                    if model_name not in transformed_output:
                        transformed_output[model_name] = {}
                    if rule_name not in transformed_output[model_name]:
                        transformed_output[model_name][rule_name] = []
                    transformed_output[model_name][rule_name].append(idx)
    return transformed_output



def calculate_correlation(coder, correct_idxs, model_name, rule_name, transformed_output, metric='correlation'):
    """
    Calculate the correlation between the correctness of response and the occurrence of a code.
    Args:
        coder (object): The coder object containing training logs.
        correct_idxs (list): A list of idxs where the model's response is correct.
        model_name (str): The name of the model.
        rule_name (str): The name of the rule.
    Returns:
        float: The correlation coefficient between the correctness of response and the occurrence of the code.
    """
    # Prune the list of correct idxs to only include those that are present in coder.training_logs['eval']
    all_idxs = list(coder.training_logs['eval'].keys())
    # Prune the list of correct idxs to only include those that are present in all_idxs
    valid_correct_idxs = [idx for idx in correct_idxs if idx in all_idxs]
    # print(len(valid_correct_idxs))
    if metric == 'correlation':
        # Create a binary vector indicating whether each idx is correct or not
        correct_vector = np.array([1 if idx in valid_correct_idxs else 0 for idx in all_idxs])
        # Create a binary vector indicating whether each idx exhibits the rule or not
        rule_vector = np.array([1 if idx in transformed_output[model_name][rule_name] else 0 for idx in all_idxs])
        # Calculate the correlation coefficient using Pearson's r
        result = np.corrcoef(correct_vector, rule_vector)[0, 1]
        cnt = rule_vector.sum()

    elif metric == 'jaccard':
        # Calculate the Jaccard index
        correct_set = set(valid_correct_idxs)
        rule_set = set(transformed_output[model_name][rule_name])
        intersection = len(correct_set.intersection(rule_set))
        union = len(correct_set.union(rule_set))
        result = intersection / union if union > 0 else 0
        cnt = union
    elif metric == 'percentage':
        # Count the number of times the rule occurs with a correct answer
        correct_count = sum(1 for idx in transformed_output[model_name][rule_name] if idx in valid_correct_idxs)
        # Count the total number of times the rule occurs
        total_count = len(transformed_output[model_name][rule_name])
        # Calculate the percentage of correct answers
        result = (correct_count / total_count) * 100 if total_count > 0 else 0
        cnt = total_count
    else:
        raise ValueError("Invalid metric. Must be 'correlation' or 'percentage_correct'.")
    return result, cnt


def match_label(dataset, idx):
    for data in dataset:
        if data["id"] == idx:
            return data["labels"]