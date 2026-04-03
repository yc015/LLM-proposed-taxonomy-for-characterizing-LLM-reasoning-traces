from src.coder_batch_trained import CoderBatchTrained
from src.coder_vllm import CoderVLLM
from copy import deepcopy
from src.prompts.few_shot_inst import FEW_SHOT_INST
import random
import os


class CoderFewShot(CoderBatchTrained):
    """
    A coder that uses few shot example and chain-of-thoughts to classify the reasoning traces
    
    Training Process:
    1. Select k-shot examples from training data
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize parent class
        super().__init__(*args, **kwargs)
        
        # Initialize training state variables
        self.few_shot_examples = []
        self.num_few_shot_examples = 3
        self.code_inst = FEW_SHOT_INST
        print("Few shot baseline")
        self.evaluation_method = "generative"
        print(f"Evaluation method must be {self.evaluation_method} for few shot baseline")

    def warm_start(self, dataset, ckpt_path=None):
        print("No warmup needed for few shot baseline")
        return 
    
    def train(self, dataset, verbosity=5, custom_logger=None, logger_kwargs=None, ckpt_path=None, 
              batch_size=32, accumulate_observation_training=False, accumulation_size=10, 
              sampling_training=False):
        """
        select few shot training examples from the dataset
        """
        print(f"Select {self.num_few_shot_examples} examples from the training dataset.")
        # select few shot training examples from the dataset without replacement
        self.few_shot_examples = random.sample(dataset, self.num_few_shot_examples)
    
    def _classify_prompt(self, outputs, question):
        coding_prompt = f"You will be given two reasoning outputs sampled from two different models, and your task is to classify which models do these reasoning outputs belong to based on the few example listed in the system message."

        option_prompt = f"Selecting from models: " + ", ".join(self.model_options)

        classification_prompt = """which model generated the OUTPUT A and which model generated the OUTPUT B? Think step by step. Make sure you reply in this specific format so we can parse your classification results:\n"Because of the characteristics [Characteristic Name] ... [Characteristic Name], the author model of OUTPUT A is [author model name]. Because of the characteristics [Characteristic Name] ... [Characteristic Name], The author model of OUTPUT B is [author model name]." """

        system_prompt = f"""{self.code_inst}\n"""

        if self.think_budget > 0:
            system_prompt += f"\nThink efficiently. Limit your thinking to {self.think_budget} words."

        sorted_few_shot_examples = []
        for example in self.few_shot_examples:
            sorted_example = {'labels': [], 'outputs': []}
            for model_option in self.model_options:
                model_example_index = example['labels'].index(model_option)
                sorted_example['labels'].append(model_option)
                sorted_example['outputs'].append(example['outputs'][model_example_index])
            sorted_few_shot_examples.append(sorted_example)

        for idx, example in enumerate(sorted_few_shot_examples):
            system_prompt += f"\nExample {idx + 1}:"
            system_prompt += f"\n###OUTPUT of model {example['labels'][0]}:\n{example['outputs'][0]}\n-----End of OUTPUT-----\n\n###OUTPUT of model {example['labels'][1]}:\n{example['outputs'][1]}\n-----End of OUTPUT-----\n\n"

        final_prompt = f"""{coding_prompt}\n\nGiven the ###OUTPUT A:\n{outputs[0]}\n-----End of OUTPUT A-----\n\nand ###OUTPUT B:\n{outputs[1]}\n-----End of OUTPUT B-----\n\n{option_prompt}, {classification_prompt}"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ]