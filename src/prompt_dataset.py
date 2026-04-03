from torch.utils.data import Dataset
import random
import os
import re
import json


# ### Load in the Reasoning Traces Dataset
def load_reasoning_traces(base_folder):
    reasoning_traces = {}
    # Iterate over each model folder in the base directory
    for model_name in os.listdir(base_folder):
        model_folder = os.path.join(base_folder, model_name)
        if not os.path.isdir(model_folder):
            continue
        # Initialize a dictionary to store reasoning traces for the current model
        model_traces = {}
        question_list = {}
        # Iterate over each file in the model folder
        for filename in os.listdir(model_folder):
            if not filename.endswith('.txt') or "raw_output" in filename:
                continue
            # Extract question ID from the filename
            if "ARC-AGI" in base_folder:
                # For ARC-AGI, extract any characters after question_id_
                question_id_match = re.search(r"question_id_([^_\s]+)", filename)
                if not question_id_match:
                    question_id_match = re.search(r"question_id_sample_([^_\s]+)", filename)
                    if not question_id_match:
                        continue
                question_id = question_id_match.group(1)  # Keep as string for ARC-AGI
            else:
                # For other models, extract digits only
                question_id_match = re.search(r"question_id_(\d+)", filename)
                if not question_id_match:
                    question_id_match = re.search(r"question_id_sample_(\d+)", filename)
                    if not question_id_match:
                        continue
                question_id = int(question_id_match.group(1))  # Convert to int for other models
            file_path = os.path.join(model_folder, filename)
            with open(file_path, 'r') as file:
                content = file.read()

                # Extract the question content
                question_match = re.search(r"### QUESTION:\s*(.*?)\n\n### THINKING:", content, re.DOTALL)
                if question_match:
                    question_content = question_match.group(1).strip()
                else:
                    question_content = ""

                # Extract the content between ### THINKING and ### ANSWER
                thinking_match = re.search(r"### THINKING:\s*(.*?)\n\n### ANSWER:", content, re.DOTALL)
                if thinking_match:
                    thinking_content = thinking_match.group(1).strip()
                else:
                    thinking_content = ""
                # Check if the extracted thinking content is empty after stripping whitespace
                if not thinking_content.strip():
                    answer_match = re.search(r"### ANSWER:\s*(.*)", content, re.DOTALL)
                    if answer_match:
                        thinking_content = answer_match.group(1).strip()
                        if "### CHOICES ORDER" in thinking_content:
                            thinking_content = thinking_content[:thinking_content.rfind("### CHOICES ORDER")]
                        answer = ""
                    else:
                        thinking_content = ""
                        answer = ""
                else:
                    # Extract the answer content after ### ANSWER
                    answer_match = re.search(r"### ANSWER:\s*(.*)", content, re.DOTALL)
                    if answer_match:
                        answer = answer_match.group(1).strip()
                        if "### CHOICES ORDER" in answer:
                            answer = answer[:answer.rfind("### CHOICES ORDER")]
                    else:
                        answer = ""

                # Check if the extracted thinking content is too short
                # This is a potential bug with code-seeder where </think> is not one token
                # If the thinking content is not empty but contain only a few words use answer content instead.
                if 10 > len(thinking_content) > 0:
                    thinking_content = answer_match.group(1).strip()
                        
                # Remove special thinking tokens
                thinking_content = thinking_content.replace("<think>", "").replace("</think>", "")
                thinking_content = thinking_content.strip("\n")

                # Store the reasoning trace in the model's dictionary with question ID as the key
                model_traces[question_id] = {
                    'question': question_content,
                    'thinking': thinking_content,
                    'answer': answer
                }
        # Store the model's reasoning traces in the main dictionary
        reasoning_traces[model_name] = model_traces
    return reasoning_traces


def make_dict_dataset(output_pairs, labels, ids=None, randomized=True):
    if randomized:
        output_pairs, labels, ids = shuffle_outputs_and_labels(output_pairs, labels, ids)

    return [{"outputs": single_outputs, 
             "labels": single_labels, 
             "id": single_id} for 
             single_outputs, 
             single_labels, 
             single_id in 
             zip(output_pairs, 
                 labels, 
                 ids)]

def shuffle_outputs_and_labels(outputs, labels, ids, questions, balance_order=False):
    if not (isinstance(outputs[0], list) or isinstance(outputs[0], tuple)):
        # If outputs is a list of single instances
        combined = list(zip(outputs, labels, ids, questions))
        random.shuffle(combined)
        shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions = zip(*combined)
        
        return list(shuffled_outputs), list(shuffled_labels), list(shuffled_ids), list(shuffled_questions)
    else:
        combined = list(zip(outputs, labels, ids, questions))
        random.shuffle(combined)
        shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions = [], [], [], []
        
        count = 0
        for output_pair, label_pair, single_id, single_question in combined:
            if balance_order:
                if count % 2 == 0:
                    shuffled_outputs.append(output_pair)
                    shuffled_labels.append(label_pair)
                else:
                    shuffled_outputs.append(output_pair[::-1])
                    shuffled_labels.append(label_pair[::-1])
                count += 1
            else:
                if random.choice([True, False]):
                    shuffled_outputs.append(output_pair)
                    shuffled_labels.append(label_pair)
                else:
                    shuffled_outputs.append(output_pair[::-1])
                    shuffled_labels.append(label_pair[::-1])
    
            shuffled_ids.append(single_id)
            shuffled_questions.append(single_question)
            
        return shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions


def create_dataset(reasoning_traces, model_options, random_seed):
    """
    Creates a dataset from reasoning traces based on provided model options.

    Args:
        random_seed (int): Seed for random number generation.
        model_options (list): List of two model options.
        reasoning_traces (dict): Dictionary containing reasoning traces for different models.

    Returns:
        list: A list of dictionaries representing the dataset.
    """

    # Set random seed
    random.seed(random_seed)

    # Get number of samples
    num_samples = len(reasoning_traces[model_options[0]])

    # Create outputs, questions, labels, and ids
    outputs = [
        [reasoning_traces[model_options[0]][idx]['thinking'], 
         reasoning_traces[model_options[1]][idx]['thinking']]
        for idx in range(num_samples)
    ]

    questions = [
        reasoning_traces[model_options[0]][idx]['question']
        for idx in range(num_samples)
    ]

    labels = [
        [model_options[0], model_options[1]]
        for idx in range(num_samples)
    ]

    ids = [idx for idx in range(num_samples)]

    # Shuffle outputs, labels, ids, and questions
    shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions = shuffle_outputs_and_labels(outputs, labels, ids, questions)

    # Create dataset
    dataset = [{"outputs": single_outputs, "labels": single_labels, "id": single_id, "question": single_question} 
               for single_outputs, single_labels, single_id, single_question in zip(shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions)]

    return dataset


class PromptDataset(Dataset):
    def __init__(self, output_pairs, labels, ids=None, randomized=True):
        self.output_pairs = output_pairs
        self.labels = labels
        self.ids = ids if ids else [idx for idx in range(len(self.output_pairs))]
        self.randomized = randomized
        self._create_dataset()
    
    def _create_dataset(self):
        if self.randomized:
            self.output_pairs, self.labels, self.ids = shuffle_outputs_and_labels(self.output_pairs, 
                                                                                  self.labels, 
                                                                                  self.ids)

    def __len__(self):
        return len(self.output_pairs)

    def __getitem__(self, idx):
        return {"within_instance_idx": idx,
                "id": self.ids[idx],
                "output_pair": self.output_pairs[idx],
                "labels": self.labels[idx]}

    

def load_arc_agi(folder_path):
    data_dict = {'data': [], 'ids': []}
    
    # Iterate over all files in the given folder
    for filename in os.listdir(folder_path):
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Open and load the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    json_content = json.load(file)
                    data_dict['data'].append(json_content)
                    # Remove the '.json' extension from the filename
                    data_dict['ids'].append(filename[:-5])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {filename}: {e}")
    
    return data_dict