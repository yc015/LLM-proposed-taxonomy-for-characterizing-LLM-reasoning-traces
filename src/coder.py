from . import CODE_INST, CORRECTION_INST, THINKING_MODEL_CODE_INST, THINKING_MODEL_CORRECTION_INST
from src.initial_code_examples import CRUXEVAL_EXAMPLES, GPQA_EXAMPLES

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from tqdm.auto import tqdm
import pickle

import numpy as np

from vllm import LLM, SamplingParams

from copy import deepcopy

import os

import re

from src.utils import fuzz_match, clean_code_name

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import random


class Coder:
    def __init__(
        self, 
        model_id, 
        model_options,
        
        cache_dir=None,
        
        use_vllm=False,
        compile=False,

        multi_gpus=4,
        torch_dtype="auto",
        
        sampling_parameters=None,

        max_input_tokens=73728,
        codebook=None, 
        code_inst=None, 
        
        evaluation_method="generative",
        train_evaluation_method="soft_vote",

        initial_code_example=None,
        correction_inst=None,

        think_budget=0,
        think_mode=False,

        templated_coding=False,

        model_name_translator=None,
        patience=2,
        global_patience=10,
        stop_criteria=["max_rules", "global_patience"],
        max_rule=40
        ):

        self.model_id = model_id
        self.use_vllm = use_vllm
        self.torch_dtype = torch.bfloat16 if "llama3.3" in self.model_id else torch_dtype
        self.compile = compile
        
        self.max_input_tokens = max_input_tokens
        self.init_model(cache_dir, multi_gpus)

        self.codebook = codebook if codebook else {}
        self.backtrack_codebook = deepcopy(self.codebook)
        self.failed_codebook = None
        self.failed_update = None
        self.current_annotation = None
        self.code_inst = code_inst if code_inst else (THINKING_MODEL_CODE_INST if think_mode else CODE_INST)
        self.correction_inst = correction_inst if correction_inst else (THINKING_MODEL_CORRECTION_INST if think_mode else CORRECTION_INST)
        self.judgement_ds = {}
        self.model_options = model_options
        self.training_logs = {"Updates": {},}
        self.code_occurrence_overall = {}
        self.decision_code_occurence_overall = {}

        self.run_type = "Normal"


        self.follow_codebook_nbc = True
        self.evaluation_method = evaluation_method
        # self.train_evaluation_method = train_evaluation_method
        # self.train_evaluation_method = evaluation_method

        self.initial_code_example = initial_code_example

        self.patience = patience
        self.global_patience = global_patience
        self.max_rule = max_rule
        self.num_consecutive_turns_without_update = 0
        self.think_mode = think_mode
        self.think_budget = think_budget
        self.templated_coding = templated_coding

        self.model_name_translator = model_name_translator
        self.stop_criteria = stop_criteria

        self._most_recent_code_occurrence_overall_train = None
        self._most_recent_decision_code_overall_train = None

        self.in_training = True
        self.embed_model = None

        self.batch_update = False
        self.batch_update_size = 4

        self.bor_training = False

        self.print_missing_code_during_parsing = False

        self.min_num_train_samples = 50
        # if self.think_mode:
        #     self.think_stop_token_id = self.tokenizer.vocab["</think>"]
        # self.generation_eos_id = self.config.eos_token_id
        if use_vllm:
            self.sampling_parameters = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=24576)
        else:
            self.sampling_parameters = sampling_parameters if sampling_parameters else {
                "do_sample": True,
                "max_new_tokens": 24576,
                "temperature": 0.6,
                "top_p": 0.95,
                }
        
        self.max_train_samples = 200

    def init_model(self, cache_dir, multi_gpus=8):
        if self.use_vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir)
            self.model = LLM(model=self.model_id, 
                             tensor_parallel_size=multi_gpus, 
                             download_dir=cache_dir,
                             dtype=self.torch_dtype)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                              torch_dtype=self.torch_dtype,
                                                              device_map="auto", 
                                                              cache_dir=cache_dir)
            if self.compile:
                self.model.generation_config.cache_implementation = "static"
                self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)

    def _parse_classification_codes(self, final_output, labels):
        """
        Parse code occurrence and decision code from processed classification output.
        
        Args:
            final_output (str): The processed classification output (after think mode processing if applicable)
            labels: The ground truth labels
            
        Returns:
            tuple: (code_occurrence, decision_code)
        """
        code_occurrence = self._parse_codes_occurence(final_output, labels)
        decision_code = self._parse_decision_code(final_output)
        return code_occurrence, decision_code

    def classify(self, outputs, labels, question, if_codebook_updated=0, train=True, qid=None, batched=False,  additional_outputs=None, additional_labels=None, additional_questions=None, no_eval=False):
        # if self.templated_coding:
        #     classification, classification_prompt = self._piece_meal_coding(outputs, question)
        # else:
        assert not (batched and train), "you could not use batched generation in training"
        if batched:
            # Your code here:
            classification_prompts = [self._classify_prompt(output, single_question) for output, single_question in zip(outputs, question)]
            classifications = self._generate_batch(classification_prompts)

            evaluations = []
            final_outputs = []
            for classification, label in zip(classifications, labels):
                if self.think_mode:
                    eot_token = "</think>"
                    eot_index = classification.rfind(eot_token)
                    if eot_index < 0:
                        print(f"Can't find the end of thinking token {eot_token}: . Potentially unfinished thought...", flush=True)
                    final_output = classification[eot_index + len(eot_token):]
                else:
                    final_output = classification
                
                if no_eval:
                    evaluation, pred = "do that later", "do that later"
                else:
                    evaluation, pred = self._evaluate_classification(final_output, label)

                evaluations.append(evaluation)
                final_outputs.append(final_output)
            return classification_prompts, classifications, evaluations, [if_codebook_updated] * len(labels), final_outputs
        else:
            classification_prompt = self._classify_prompt(outputs, question)
            classification = self._generate(classification_prompt)

            # print(classification)
            if self.think_mode:
                eot_token = "</think>"
                eot_index = classification.rfind(eot_token)
                if eot_index < 0:
                    print(f"Can't find the end of thinking token {eot_token}: . Potentially unfinished thought...", flush=True)
                final_output = classification[eot_index + len(eot_token):]
            else:
                final_output = classification

            self.current_annotation = final_output
                
            # Update the most recent code occurrence and decision code so classification can use the new code for evaluation
            code_occurrence, decision_code = self._parse_classification_codes(final_output, labels)
            self._most_recent_code_occurrence_overall_train = deepcopy(self.code_occurrence_overall["train"])
            self._most_recent_decision_code_overall_train = deepcopy(self.decision_code_occurence_overall["train"])
            self._update_code_occurence(self._most_recent_code_occurrence_overall_train, code_occurrence, is_decision_code=False, sample_id=qid)
            self._update_code_occurence(self._most_recent_decision_code_overall_train, decision_code, is_decision_code=True)

            evaluation, pred = self._evaluate_classification(final_output, labels)

            if evaluation == "correct":
                self._update_dataset(classification_prompt, classification)
                if if_codebook_updated > 0 and train:
                    self.num_consecutive_turns_without_update = 0
                else:
                    self.num_consecutive_turns_without_update += 1
                    # pass
            # However if the classification is incorrect, we will retract the update to the code occurrence and decision code overall
            elif train:
                if if_codebook_updated >= self.patience:
                    print(f"------\nOut of patience after {self.patience} update trails", flush=True)
                    if qid:
                        print(f"at sample {qid}", flush=True)
                    print("------", flush=True)
                    
                    # If the backtrack codebook is not empty and not VML, we will use it to backtrack
                    if len(self.backtrack_codebook.keys()) > 0 and self.run_type != "VML":
                        self.codebook = deepcopy(self.backtrack_codebook)
                    if hasattr(self, 'bor_training') and self.bor_training:
                        self._most_recent_code_occurrence_overall_train = deepcopy(self.code_occurrence_overall["train"])
                        self._most_recent_decision_code_overall_train = deepcopy(self.decision_code_occurence_overall["train"])
                    self.num_consecutive_turns_without_update += 1
                else:
                    # Update the codebook and redo the classification
                    # Retract the code occurrence and decision code overall to the previous update
                    self._most_recent_code_occurrence_overall_train = deepcopy(self.code_occurrence_overall["train"])
                    self._most_recent_decision_code_overall_train = deepcopy(self.decision_code_occurence_overall["train"])
                    # Backtrack the codebook if if_codebook_updated > 0 but still failed
                    # Then, we can go back to the previous codebook and start from there again
                    if if_codebook_updated > 0:
                        print("Backtracking", flush=True)
                        self.failed_codebook = deepcopy(self.codebook)
                        self.codebook = deepcopy(self.backtrack_codebook)
                    # If this is the first time update, we will keep a copy of the current codebook
                    else:
                        self.backtrack_codebook = deepcopy(self.codebook)
                    if self.batch_update and additional_outputs is not None:
                        return self._update_codebook_batch(outputs, labels, question, additional_outputs, additional_labels, additional_questions, qid=qid, if_codebook_updated=if_codebook_updated)
                    else:
                        return self._update_codebook(outputs, labels, question, qid=qid, if_codebook_updated=if_codebook_updated)

            # If this is a training sample and we reach to the end of the update process, either because of the patience or the update is successful
            # We will update the code occurrence and decision code overall for next update
            # We will also reset the backtrack codebook and failed codebook for next update
            if train:
                # Update the code occurrence and decision code overall for next update
                # Reset the backtrack codebook and failed codebook for next update
                self.backtrack_codebook = None
                self.failed_codebook = None
                self.failed_update = None

            return classification_prompt, classification, evaluation, if_codebook_updated, final_output

    def _classify_prompt(self, outputs, question):
        coding_prompt = f"You will be given two reasoning outputs sampled from two different models, and your task is to classify which models do these reasoning outputs belong to based on the distinguishing reasoning traits listed in the reasoning behavior taxonomy.\n\nThe system message provides a reasoning behavior taxonomy that illustrates the reasoning difference between two models. Follow the instruction and the reasoning behavior taxonomy in the system message and this message closely when making the classification."

        # coding_prompt = f"You will be given two reasoning outputs sampled from two different models. These are reasonings for the question:\n{question}\n\nYour task is to distinguish which models do these reasoning outputs belong to based on the distinguishing reasoning traits listed in the codebook."

        option_prompt = f"Selecting from models: " + ", ".join(self.model_options)

        classification_prompt = """which model generated the OUTPUT A and which model generated the OUTPUT B? Think step by step. Make sure you reply in this specific format so we can parse your classification results:\n"Because of the reasoning behaviors [Reasoning Behavior Name 1] ... [Reasoning Behavior Name X], the author model of OUTPUT A is [author model name]. Because of the reasoning behaviors [Reasoning Behavior Name 1] ... [Reasoning Behavior Name Y], the author model of OUTPUT B is [author model name]." Your final classification should not be biased by the order in which possible models appear in my prompt. Note that the given OUTPUTs are generated by two different models. If you determine that both OUTPUTs are likely generated by the same model, select the OUTPUT that most closely resembles that model's reasoning pattern."""

        system_prompt = f"""{self.code_inst}\n"""
        
        for code in self.codebook.keys():
            # code_definition = self.codebook[code]
            # if f"[Exhibited by {self.model_options[0]} and {self.model_options[1]}]" in code_definition or f"[Exhibited by {self.model_options[1]} and {self.model_options[0]}]" in code_definition:
            #     continue

            system_prompt += f"{code}: {self.codebook[code]}\n\n"

        if self.think_budget > 0:
            system_prompt += f"\nThink efficiently. Limit your thinking to {self.think_budget} words."

        final_prompt = f"""{coding_prompt}\n\nGiven the ###OUTPUT A:\n{outputs[0]}\n-----End of OUTPUT A-----\n\nand ###OUTPUT B:\n{outputs[1]}\n-----End of OUTPUT B-----\n\n{option_prompt}, {classification_prompt}"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ]

    def _generate(self, prompt, templated=False, max_tokens=None, temp_adjust=0):
        if not templated:
            text = self.tokenizer.apply_chat_template(prompt, 
                                                      tokenize=False, 
                                                      add_generation_prompt=True,
                                                      enable_thinking=self.think_mode)
        else:
            text = prompt

        if not self.use_vllm:
            # print(text)
            model_inputs = self.tokenizer(text, 
                                          return_tensors="pt", 
                                          padding=True, 
                                          padding_side="left").to(self.model.device)

            # conduct text completion
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens if max_tokens is not None else self.sampling_parameters["max_new_tokens"],
                do_sample=self.sampling_parameters["do_sample"],
                temperature=self.sampling_parameters["temperature"] + temp_adjust,
                top_p=self.sampling_parameters["top_p"],
                top_k=50,
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

            classification_result = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        else:
            classification_result = self.model.generate(text,
                                                        sampling_params=self.sampling_parameters,
                                                        use_tqdm=False)[0].outputs[0].text

        return classification_result

    def _generate_batch(self, prompts, templated=False, max_tokens=None, temp_adjust=0):
        if not templated:
            texts = [self.tokenizer.apply_chat_template(prompt, 
                                                        tokenize=False, 
                                                        add_generation_prompt=True,
                                                        enable_thinking=self.think_mode) for prompt in prompts]
        else:
            texts = prompts
        if not self.use_vllm:
            # print(texts)
            model_inputs = self.tokenizer(texts, 
                                        return_tensors="pt", 
                                        padding=True, 
                                        padding_side="left").to(self.model.device)
            # conduct text completion
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens if max_tokens is not None else self.sampling_parameters["max_new_tokens"],
                do_sample=self.sampling_parameters["do_sample"],
                temperature=self.sampling_parameters["temperature"] + temp_adjust,
                top_p=self.sampling_parameters["top_p"],
                top_k=50,
            )
            output_ids = [ids[len(model_inputs.input_ids[i]):] for i, ids in enumerate(generated_ids)]
            classification_results = [self.tokenizer.decode(ids, skip_special_tokens=True).strip("\n") for ids in output_ids]
        else:
            outputs = self.model.generate(texts,
                                                         sampling_params=self.sampling_parameters,
                                                         use_tqdm=False)
            classification_results = []
            for output in outputs:
                generated_text = output.outputs[0].text
                classification_results.append(generated_text)
            
        return classification_results
    
    def few_shot_warm_start(self, dataset, qid=0):
        labels = []
        outputs = []
        if "Few-shot Updates" not in self.training_logs.keys():
            self.training_logs["Few-shot Updates"] = {}
        for idx in tqdm(range(len(dataset))):
            outputs.append(dataset[idx]["outputs"])
            labels.append(dataset[idx]["labels"])
        correction_prompt = "You will be given multiple reasoning outputs and the names of their author models. Given the current reasoning behavior taxonomy, think step by step, your task is to check if any existing reasoning behavior needs to be updated or any new reasoning behaviors need to be added so you can make a correct classification on these outputs.\n\n"

        # correction_prompt += f"Below are the reasoning outputs for the question:\n{question}\n\n"
        for label, output in zip(labels, outputs): 
            correction_prompt += f"Below is the OUTPUT from {label[0]}:\n{output[0]}\n-----End of OUTPUT-----\n\n"
            correction_prompt += f"Below is the OUTPUT from {label[1]}:\n{output[1]}\n-----End of OUTPUT-----\n\n"

        correction_prompt += f"Follow the instruction in the system prompt and this message to update the reasoning behavior taxonomy. Since you have seen multiple example outputs from two models, try to find the patterns that are consistent across all outputs from each model.\n\nWhen giving the updated and added reasoning behaviors, follow the format specificed in the system message exactly (behavior name, reasoning behavior definition, and a quoted example)."

        system_prompt = self.correction_inst
        for code in self.codebook.keys():
            system_prompt += f"{code}: {self.codebook[code]}\n\n"
        if len(self.codebook.keys()) < 1:
            system_prompt += f"Current reasoning behavior taxonomy is empty. You should compare the reasoning traces and add new reasoning behavior into this reasoning behavior taxonomy!"
            if self.initial_code_example is not None and len(self.initial_code_example.keys()) > 0:
                system_prompt += "\n\nBelow are example codes that illustrate the reasoning behaviors of an imagined language model Alpha. Your should follow the format and style of these examples when generating the added and updated reasoning behaviors.\nExamples:\n"
                for code in list(self.initial_code_example.keys()):
                    system_prompt += f"{code}: {self.initial_code_example[code]}\n\n"

        if self.think_budget > 0:
            system_prompt += f"\n\nThink efficiently. Limit your thinking to {self.think_budget} words."

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": correction_prompt}]

        new_code = self._generate(messages)
        
        if self.think_mode:
            final_new_code = new_code[new_code.rfind("</think>") + len("</think>"):]
            self.codebook = self._parse_new_codes(final_new_code)
        else:
            self.codebook = self._parse_new_codes(new_code)
        # print(new_code)

        if qid:
            self.training_logs["Few-shot Updates"][qid] = {"input_prompt": messages,
                                                           "update": new_code,
                                                           "cb_ckpt": deepcopy(self.codebook),
                                                           "dataset": dataset}
            
        print(f"Warm start finished with {len(dataset)} examples")
        print("Current reasoning behavior taxonomy looks like")
        self.print_codebook()

        return new_code

    def _update_codebook(self, outputs, labels, question, qid=None, if_codebook_updated=0):
        messages = self._update_prompt(outputs, labels)

        new_code = self._generate(messages, temp_adjust=0.0)
        
        original_new_code = new_code
        if self.think_mode:
            final_new_code = new_code[new_code.rfind("</think>") + len("</think>"):]
            loc_failed_updates = final_new_code.rfind("Final output:")  + len("Final output:")
            if loc_failed_updates == -1:
                loc_failed_updates = final_new_code.lower().rfind("updated:")
            if loc_failed_updates == -1:
                loc_failed_updates = final_new_code.rfind("###")
            loc_end_failed_updates = final_new_code.lower().rfind("because of the reasoning behaviors")
            
            self.failed_update = final_new_code[loc_failed_updates:loc_end_failed_updates]

            self.codebook = self._parse_new_codes(final_new_code)
        else:
            loc_failed_updates = new_code.rfind("Final output:")  + len("Final output:")
            if loc_failed_updates == -1:
                loc_failed_updates = new_code.lower().rfind("updated:")
            if loc_failed_updates == -1:
                loc_failed_updates = new_code.rfind("###")
            loc_end_failed_updates = new_code.lower().rfind("because of the reasoning behaviors")

            self.failed_update = new_code[loc_failed_updates:loc_end_failed_updates]
            self.codebook = self._parse_new_codes(new_code)
        # print(new_code)

        if qid:
            self.training_logs["Updates"][qid] = {"input_prompt": messages,
                                                  "update": new_code,
                                                  "cb_ckpt": deepcopy(self.codebook),
                                                  "labels": labels,
                                                  "outputs": outputs,
                                                  "question": question,
                                                  "original_update": original_new_code}
        print("-" * 20 + " UPDATED CODEBOOK " + "-" * 20)
        self.print_codebook(common_code=False)
        print("-" * 50)
        # self.print_codebook(common_code=True)

        if hasattr(self, "no_check_after_update") and self.no_check_after_update:
            self.num_consecutive_turns_without_update = 0
            return messages, new_code, "update-no-check", if_codebook_updated + 1, new_code
        return self.classify(outputs, labels, question, if_codebook_updated=if_codebook_updated + 1, qid=qid)

    def _update_prompt(self, outputs, labels):
        correction_prompt = "You will be given two reasoning outputs and the names of their author models. Given the current reasoning behavior taxonomy, think step by step, your task is to check if any new reasoning behaviors need to be added so you can make a correct classification on these two outputs.\n\n"

        # correction_prompt += f"Below are the reasoning outputs for the question:\n{question}\n\n"

        correction_prompt += f"Below is the ###OUTPUT A from {labels[0]}:\n{outputs[0]}\n-----End of OUTPUT A-----\n\n"

        correction_prompt += f"Below is the ###OUTPUT B from {labels[1]}:\n{outputs[1]}\n-----End of OUTPUT B-----\n\n"

        if self.current_annotation is not None:
            correction_prompt += f"This is your annotation following the existing taxonomy:\n{self.current_annotation}\n\n"

        correction_prompt += f"Follow the instruction in the system prompt and this message to update the reasoning behavior taxonomy.\n\nGive detailed definition and example for the updated and added reasoning behaviors!\n\n"

        correction_prompt += f'At the end of your response, make sure you could confidently say that "Now, I will classify the author model based on the majority vote of appeared reasoning behaviors. The reasoning behaviors used in classifying each model should not be overlapped.\n\nBecause of the reasoning behaviors [Reasoning Behavior Name 1] ... [Reasoning Behavior Name m], I can confidently say that the author model of OUTPUT A is {labels[0]}. Because of the reasoning behaviors [Reasoning Behavior Name 1] ... [Reasoning Behavior Name m], the author model of OUTPUT B is {labels[1]}." with the updated reasoning behavior taxonomy. Make sure your final output follows the format in the system message exactly.'

        system_prompt = self.correction_inst
        included_code = 0
        for code in self.codebook.keys():
            system_prompt += f"{code}: {self.codebook[code]}\n\n"
            included_code += 1

        if included_code < 1:
            system_prompt += f"Current reasoning behavior taxonomy is empty. You should compare the reasoning traces and add new reasoning behavior into this reasoning behavior taxonomy!"

            if self.initial_code_example is not None and len(self.initial_code_example.keys()) > 0:
                system_prompt += "\n\nTo help you get started, below are example reasoning behaviors that illustrate the reasoning behaviors of an imagined language model Alpha. You should follow the format, style, and detailedness of these examples when generating the added and updated reasoning behaviors.\n\n"
                for code in list(self.initial_code_example.keys()):
                    system_prompt += f"{code}: {self.initial_code_example[code]}\n\n"
        
        if self.failed_codebook is not None:
            system_prompt += f"\n\nWe also know that the following updated reasoning behavior taxonomy cannot make the right classification, so your updates may want to avoid the changes in this reasoning behavior taxonomy below:\n\n"
            
            # system_prompt += f"Failed Updates:\n{self.failed_update}\n\nAvoid the updates above when generating your final output."
            # if len(self.failed_update) < 1 and self.failed_codebook is not None:
            for code in self.failed_codebook.keys():
                system_prompt += f"{code}: {self.failed_codebook[code]}\n\n"

        if self.think_budget > 0:
            system_prompt += f"\n\nThink efficiently. Limit your thinking to {self.think_budget} words."

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": correction_prompt}]
                    
        return messages

    def _update_codebook_batch(self, outputs, labels, question, additional_outputs, additional_labels, additional_questions, qid=None, if_codebook_updated=0):
        """
        Batch version of _update_codebook that uses multiple examples to inform the update but returns classification for the primary example.
        
        Args:
            outputs: Primary output pair [output_a, output_b] that needs to be reclassified
            labels: Primary label pair [label_a, label_b] that needs to be reclassified
            question: Primary question that needs to be reclassified
            additional_outputs: List of additional output pairs to help inform the codebook update
            additional_labels: List of additional label pairs to help inform the codebook update  
            additional_questions: List of additional questions to help inform the codebook update
            qid: Optional question ID for the primary example
            if_codebook_updated: Counter for update attempts
            
        Returns:
            Classification result for the primary example after codebook update
        """
        correction_prompt = "You will be given multiple reasoning outputs and the names of their author models. Given the current reasoning behavior taxonomy, think step by step, your task is to check if any new reasoning behaviors need to be added so you can make correct classifications on these outputs based on the occurrence of the new reasoning behaviors.\n\n"

        # Add the primary example first
        correction_prompt += f"### Example 1 ###\n"
        # correction_prompt += f"Question: {question}\n\n"
        correction_prompt += f"Below is the OUTPUT from {labels[0]}:\n{outputs[0]}\n-----End of OUTPUT from {labels[0]}-----\n\n"
        correction_prompt += f"Below is the OUTPUT from {labels[1]}:\n{outputs[1]}\n-----End of OUTPUT from {labels[1]}-----\n\n### Example 1 END ###\n\n\n\n"

        # Add all the additional examples to the correction prompt
        # Keep the label of output the same order as the given Example 1
        for i, (add_outputs, add_labels, add_question) in enumerate(zip(additional_outputs, additional_labels, additional_questions)):
            label_0_index = add_labels.index(labels[0])
            label_1_index = add_labels.index(labels[1])
            add_outputs = [add_outputs[label_0_index], add_outputs[label_1_index]]
            add_labels = [add_labels[label_0_index], add_labels[label_1_index]]
            correction_prompt += f"### Example {i+2} ###\n"
            # correction_prompt += f"Question: {add_question}\n\n"
            correction_prompt += f"Below is the OUTPUT from {add_labels[0]}:\n{add_outputs[0]}\n-----End of OUTPUT from {add_labels[0]}-----\n\n"
            correction_prompt += f"Below is the OUTPUT from {add_labels[1]}:\n{add_outputs[1]}\n-----End of OUTPUT from {add_labels[1]}-----\n\n### Example {i+2} END ###\n\n\n\n"

        correction_prompt += f"Follow the instruction in the system prompt and this message to update the reasoning behavior taxonomy based on all the examples above.\n\nGive detailed definition and example for the updated and added decision reasoning behaviors!\n\n"

        # Build the final validation instruction for all examples
        validation_examples = []
        # validation_examples.append(f'Make sure you could confidently say: "Because of the reasoning behavior [Reasoning Behavior Name], the author model of OUTPUT A in Example 1 is {labels[0]}. Because of the reasoning behavior [Reasoning Behavior Name], the author model of OUTPUT B in Example 1 is {labels[1]}." with the updated reasoning behavior taxonomy. Repeat this for all examples.')
        
        validation_text = "\n".join(validation_examples)
        # correction_prompt += f'At the end of your response, make sure you could confidently say:\n{validation_text}\n\nMake sure your final output follows the format in the system message exactly.'
        correction_prompt += f'Make sure your final output follows the format in the system message exactly.'

        # Build system prompt with current codebook
        system_prompt = self.correction_inst
        included_code = 0
        for code in self.codebook.keys():
            system_prompt += f"{code}: {self.codebook[code]}\n\n"
            included_code += 1

        if included_code < 1:
            system_prompt += f"Current reasoning behavior taxonomy is empty. You should compare the reasoning traces and add new decision reasoning behaviors into this reasoning behavior taxonomy!"
            if self.initial_code_example is not None and len(self.initial_code_example.keys()) > 0:
                system_prompt += "\n\nTo help you get started, below are example reasoning behaviors that illustrate the reasoning behaviors of an imagined language model Alpha. You should follow the format, style, and detailedness of these examples when generating the added and updated reasoning behaviors.\n\n"
                for code in list(self.initial_code_example.keys()):
                    system_prompt += f"{code}: {self.initial_code_example[code]}\n\n"
        
        # Include failed codebook information if available
        # if self.failed_codebook is not None:
        #     system_prompt += f"\n\nWe also know that the following updated reasoning behavior taxonomy cannot make the right classifications, so your updates may want to avoid the changes in this reasoning behavior taxonomy below:\n\n"
        #     for code in self.failed_codebook.keys():
        #         system_prompt += f"{code}: {self.failed_codebook[code]}\n\n"

        if self.think_budget > 0:
            system_prompt += f"\n\nThink efficiently. Limit your thinking to {self.think_budget} words."

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": correction_prompt}]

        # Generate the codebook update
        new_code = self._generate(messages, temp_adjust=0.0)
        
        # Parse the generated update
        if self.think_mode:
            final_new_code = new_code[new_code.rfind("</think>") + len("</think>"):]
            loc_failed_updates = final_new_code.rfind("Final output:")  + len("Final output:")
            if loc_failed_updates == -1:
                loc_failed_updates = final_new_code.lower().rfind("updated:")
            if loc_failed_updates == -1:
                loc_failed_updates = final_new_code.rfind("###")
            loc_end_failed_updates = final_new_code.lower().rfind("because of the reasoning behaviors")
            
            self.failed_update = final_new_code[loc_failed_updates:loc_end_failed_updates]
            self.codebook = self._parse_new_codes(final_new_code)
        else:
            loc_failed_updates = new_code.rfind("Final output:")  + len("Final output:")
            if loc_failed_updates == -1:
                loc_failed_updates = new_code.lower().rfind("updated:")
            if loc_failed_updates == -1:
                loc_failed_updates = new_code.rfind("###")
            loc_end_failed_updates = new_code.lower().rfind("because of the reasoning behaviors")

            self.failed_update = new_code[loc_failed_updates:loc_end_failed_updates]
            self.codebook = self._parse_new_codes(new_code)

        # Log the update
        if qid:
            self.training_logs["Updates"][qid] = {
                "input_prompt": messages,
                "update": new_code,
                "cb_ckpt": deepcopy(self.codebook),
                "labels": labels,
                "outputs": outputs,
                "question": question,
                "additional_labels": additional_labels,
                "additional_outputs": additional_outputs,
                "additional_questions": additional_questions
            }

        print("-" * 20 + " BATCH UPDATED CODEBOOK " + "-" * 20)
        self.print_codebook(common_code=False)
        print("-" * 50)

        # Test the updated codebook on the primary example only
        return self.classify(outputs, labels, question, if_codebook_updated=if_codebook_updated + 1, qid=qid, additional_outputs=additional_outputs, additional_labels=additional_labels, additional_questions=additional_questions)

    def _parse_new_codes(self, updates):
        """
        Update the codebook dictionary with the given updates.
        Parameters:
        - codebook (dict): The original codebook with reasoning behavior names as keys and definitions as values.
        - updates (str): The string containing updates in the specified format.
        Returns:
        - dict: The updated codebook.
        """
        # print(updates)
        updated_codebook = deepcopy(self.codebook)
        # Split the updates into lines
        if "final output:" in updates:
            updates = updates[updates.rfind("final output:"):]
        lines = updates.strip().split('\n')
        # Flags to track sections
        in_added_section = False
        in_updated_section = False
        # Process each line to find updates and additions
        number_of_add_codes = 0
        for line in lines:
            line = line.strip()
            if "Added:" in line:
                in_added_section = True
                in_updated_section = False
                continue
            if "Updated:" in line:
                in_added_section = False
                in_updated_section = True
                continue
            if in_added_section and ":" in line:
                if number_of_add_codes > 8:
                    print("Unusual: TOO MANY UPDATES. SKIPPED THE REST OF UPDATES")
                    continue
                try:
                    # Extract the added code name and definition
                    code_name, code_definition = line.split(":", 1)
                    code_name = code_name.strip("- ").strip().replace("*", "").replace("#", "")
                    code_definition = code_definition.strip()
                    if "final answer is" in code_name.lower() or len(code_definition) < 10:
                        print(f"PARSE ERROR: Meaningless reasoning behavior :{code_name}: {code_definition}", flush=True)
                        continue
                    if "example from output b" in code_name.strip().lower() or "example from output a" in code_name.strip().lower() or "example" == code_name.strip().lower() or "example of this behavior" in code_name.strip().lower():
                        print(f"PARSE ERROR: Meaningless reasoning behavior :{code_name}: {code_definition}", flush=True)
                        continue
                    if "definition" == code_name.strip().lower():
                        print(f"PARSE ERROR: Meaningless reasoning behavior :{code_name}: {code_definition}", flush=True)
                        continue
                    if "note" == code_name.strip().lower():
                        print(f"PARSE ERROR: Meaningless reasoning behavior :{code_name}: {code_definition}", flush=True)
                        continue
                    if "none" == code_name.strip().lower():
                        print(f"PARSE ERROR: Meaningless reasoning behavior :{code_name}: {code_definition}", flush=True)
                        continue
                    if "example of" in code_name.strip().lower():
                        print(f"PARSE ERROR: Meaningless reasoning behavior :{code_name}: {code_definition}", flush=True)
                        continue
                    if "count in output" in code_name.strip().lower():
                        print(f"PARSE ERROR: Meaningless reasoning behavior :{code_name}: {code_definition}", flush=True)
                        continue
                    if "OUTPUT A" in code_name or "OUTPUT B" in code_name:
                        print(f"PARSE ERROR: Meaningless reasoning behavior :{code_name}: {code_definition}", flush=True)
                        continue
                    # Add the new code to the codebook
                    if_matched = fuzz_match(code_name, list(updated_codebook.keys()), 90, method="character")
                    if if_matched:
                        print(f"SKIPPED: An added reasoning behavior seems to be too similar to the existing reasoning behavior {code_name} -> {if_matched}", flush=True)
                        # updated_codebook[if_matched] = code_definition
                    else:
                        # Cut short the updated definition if more than 1000 characters
                        if len(code_definition) > 2000:
                            code_definition = code_definition[:2000] + "..."
                        updated_codebook[code_name] = code_definition
                        number_of_add_codes += 1
                except Exception as e:
                    print(e)
                    print("Error when trying to parse the added code:", line)
            if in_updated_section and "->" in line:
                # Extract the original and updated code names and definitions
                try:
                    parts = line.split("->")
                    original_code = parts[0].strip("- ").strip().replace("*", "").replace("#", "")
                    updated_code, updated_definition = parts[1].split(":", 1)
                    updated_code = updated_code.strip().replace("*", "")
                    updated_definition = updated_definition.strip()
                    # Update the codebook
                    if_matched = fuzz_match(original_code, list(updated_codebook.keys()), 90, method="character")
                    # if original_code in updated_codebook:
                    if if_matched:
                        del updated_codebook[if_matched]
                        # Cut short the updated definition if more than 1000 characters
                        if len(updated_definition) > 2000:
                            updated_definition = updated_definition[:2000] + "..."
                        updated_codebook[original_code] = updated_definition
                except Exception as e:
                    print(e)
                    print("Error when trying to parse the updated code:", line)
        return updated_codebook

    def _parse_decision_code(self, classification):
        # Define a regular expression pattern to match the rules and model names
        pattern = r"Because of (?:the )?reasoning behaviors (.+?),\s*the author model of OUTPUT A is (\w+(?:-\w+)+)\.\s*Because of (?:the )?reasoning behaviors (.+?),\s*the author model of OUTPUT B is (\w+(?:-\w+)+)\."

        # Use re.search to find the pattern in the text
        classification = classification[classification.rfind("###"):]
        match = re.search(pattern, classification)
        if match:
            # Extract the rules and model names from the match groups
            rules_a = match.group(1).split(", ")
            model_a = match.group(2)
            rules_b = match.group(3).split(", ")
            model_b = match.group(4)
            all_rules = set(rules_a + rules_b)
            # Initialize the result dictionary
            result = {rule: {model_a: False, model_b: False} for rule in all_rules}
            # Update the dictionary based on rule occurrence
            for rule in rules_a:
                result[rule][model_a] = True
            for rule in rules_b:
                result[rule][model_b] = True
            return result
        else:
            return {}
    
    def _parse_codes_occurence(self, classification, labels=None):
        """
        Extract the codes and their existence in each output (A or B) as assessed by the coder.
        """
        # Split the text into lines
        lines = classification.split('\n')
        
        # Initialize a dictionary to store the results
        code_occurrences = {}
        
        # Initialize variables to keep track of the current code
        current_code = None
        
        if labels is None:
            model_a_name = "OUTPUT A"
            model_b_name = "OUTPUT B"
        else:
            model_a_name = labels[0]
            model_b_name = labels[1]

        # Iterate over each line
        for line in lines:
            line = line.strip()
            
            # Check if this is a rule name line (starts with ### but is not a definition, exhibited by, etc.)
            if line.startswith("###") and not any(keyword in line.lower() for keyword in [
                "definition of this reasoning behavior", "exhibited by", "in output", "output a shows", "output b shows", 
                "this reasoning behavior is", "because of this reasoning behavior", "output a", "output b", "reasoning behavior name", ":"
            ]):
                # This is likely a rule name
                rule_name = line.strip("#").strip()
                
                # Skip if it's too long (likely not a rule name) or too short
                if len(rule_name.split()) > 10 or len(rule_name.strip()) == 0:
                    # print("Rule name is too long. Maybe not a rule?\n", rule_name)
                    continue
                    
                current_code = rule_name
                code_occurrences[current_code] = {model_a_name: False, model_b_name: False}
            
            # Check for observation statements about the current rule
            elif current_code is not None and line.startswith("###") and "reasoning behavior is" in line.lower():
                # Handle positive observations
                if "is observed in output a" in line.lower():
                    code_occurrences[current_code][model_a_name] = True
                elif "is observed in output b" in line.lower():
                    code_occurrences[current_code][model_b_name] = True
                # Handle negative observations
                elif "is not observed in output a" in line.lower() or "not observed in output a" in line.lower():
                    code_occurrences[current_code][model_a_name] = False
                elif "is not observed in output b" in line.lower() or "not observed in output b" in line.lower():
                    code_occurrences[current_code][model_b_name] = False
            elif current_code is None and line.startswith("###") and "reasoning behavior is" in line.lower():
                # if self.print_missing_code_during_parsing:
                print("Missing a reasoning behavior!", line)
        
        return code_occurrences
        
    def _update_dataset(self, input_prompt, generation):
        if "INPUTS" not in self.judgement_ds.keys(): 
            self.judgement_ds["INPUTS"] = []
        self.judgement_ds["INPUTS"].append(input_prompt)
        if "OUTPUTS" not in self.judgement_ds.keys():
            self.judgement_ds["OUTPUTS"] = []
        self.judgement_ds["OUTPUTS"].append(generation)

    def _evaluate_classification(self, classification, labels, print_details=True):
        """
        Extracts the classification results from the given output, ignoring capitalization.
        Parameters:
        - output (str): The string containing the classification results.
        Returns:
        - dict: A dictionary with the classification results for outputs A and B.
        """
        if self.evaluation_method == "generative":
            model_a, model_b = self._generative_classification(classification, print_details=print_details)
        elif self.evaluation_method == "hard_vote":
            model_a, model_b = self._hard_voting_classification(classification, print_details=print_details)
        elif self.evaluation_method == "soft_vote":
            model_a, model_b = self._soft_voting_classification(classification, print_details=print_details)
        elif self.evaluation_method == "naive_bayes":
            model_a, model_b = self._naive_bayes_classification(classification, 
                                                                follow_code_occurrence_exactly=self.follow_codebook_nbc,
                                                                print_details=print_details)
        elif self.evaluation_method == "logistic_regression":
            model_a, model_b = self._logistic_regression_classification(classification, print_details=print_details)
        elif self.evaluation_method == "knn":
            model_a, model_b = self._knn_classification(classification, print_details=print_details, k=5)
        elif self.evaluation_method == "cosine_similarity":
            result = self._similarity_based_classification(classification, metric='cosine', print_details=print_details)
        elif self.evaluation_method == "euclidean_similarity":
            result = self._similarity_based_classification(classification, metric='euclidean', print_details=print_details)
            
        if print_details:
            print("Ground Truth:", flush=True)
            print(f"OUTPUT A: {labels[0]}; OUTPUT B: {labels[1]}", flush=True)
            print(f"Classification:", flush=True)
            print(f"OUTPUT A: {model_a}; OUTPUT B: {model_b}\n", flush=True)

        if model_a == labels[0] and model_b == labels[1]:
            return "correct", [model_a, model_b]
        elif model_a != labels[0] and model_b == labels[1]:
            return "Model A wrong", [model_a, model_b]
        elif model_a == labels[0] and model_b != labels[1]:
            return "Model B wrong", [model_a, model_b]
        else:
            return "Both wrong", [model_a, model_b]

    def _generative_classification(self, classification,print_details=False):
        """
        Extracts the classification results from the given output, ignoring capitalization.
        Parameters:
        - output (str): The string containing the classification results.
        Returns:
        - dict: A dictionary with the classification results for outputs A and B.
        """
        # Initialize a dictionary to store the classification results
        classification_results = {}
        # Convert the entire output to lowercase for case-insensitive processing
        output_lower = classification.lower()

        phrase_a = "the author model of output a is"
        phrase_b = "the author model of output b is"
        
        def extract_model_name(phrase, output_text, output_lower):
            """Helper function to extract model name after a given phrase"""
            index = output_lower.find(phrase)
            if index == -1:
                return "unknown"
            
            try:
                start_index = index + len(phrase)
                
                # Look for sentence endings that indicate the model name has ended
                # We'll look for patterns like ". " (period followed by space/capital)
                # or end of string, or newline
                remaining_text = output_text[start_index:]
                remaining_lower = output_lower[start_index:]
                
                # Find potential end markers
                end_markers = []
                
                # Look for period followed by space and capital letter (sentence end)
                period_match = re.search(r'\.\s+[A-Z]', remaining_text)
                if period_match:
                    end_markers.append(period_match.start())
                
                # Look for period at end of string
                if remaining_text.rstrip().endswith('.'):
                    end_markers.append(len(remaining_text.rstrip()) - 1)
                
                # Look for comma (as backup)
                comma_index = remaining_lower.find(",")
                if comma_index != -1:
                    end_markers.append(comma_index)
                
                # Look for newline
                newline_index = remaining_lower.find("\n")
                if newline_index != -1:
                    end_markers.append(newline_index)
                
                # Use the earliest valid end marker
                if end_markers:
                    end_index = min(end_markers)
                    model_text = remaining_text[:end_index].strip()
                else:
                    # No clear end marker found, take the rest of the text
                    model_text = remaining_text.strip()
                    # Remove trailing period if present
                    if model_text.endswith('.'):
                        model_text = model_text[:-1]
                
                # Extract the last word/token as the model name
                model_name = model_text.split()[-1] if model_text.split() else ""
                
                # Clean up the model name
                model_name = model_name.replace("*", "").replace("[", "").replace("]", "")
                
                return model_name if model_name else "unknown"
                
            except Exception as e:
                print(e)
                print(f"Error when parsing the classification results: {remaining_text.strip()}")
                return "unknown"
        
        # Extract model names for both outputs
        model_a = extract_model_name(phrase_a, classification, output_lower)
        model_b = extract_model_name(phrase_b, classification, output_lower)
        
        classification_results['Output A'] = model_a
        classification_results['Output B'] = model_b

        return model_a, model_b

    def _soft_voting_classification(self, classification, print_details=True, return_scores=False):
        """
        Classifies outputs using soft voting based on historical code occurrence probabilities.
        
        Parameters:
        - classification (str): The classification output text
        - labels (list): Ground truth labels [model_a_true, model_b_true]
        - phase (str): The phase to use for historical data ("train" or "eval")
        
        Returns:
        - tuple: (evaluation_result, predicted_labels)
        """
        # Parse code occurrences from the classification text using default OUTPUT A/B names
        code_occurrences = self._parse_codes_occurence(classification)
        
        # Get historical code occurrence data
        historical_data = self._most_recent_code_occurrence_overall_train if self._most_recent_code_occurrence_overall_train is not None else self.code_occurrence_overall["train"]
        
        # Initialize scores for each output and model
        score_A_model0 = 0.0  # Score for OUTPUT A being model_options[0]
        score_A_model1 = 0.0  # Score for OUTPUT A being model_options[1]
        score_B_model0 = 0.0  # Score for OUTPUT B being model_options[0]
        score_B_model1 = 0.0  # Score for OUTPUT B being model_options[1]
        
        valid_codes_count_A = 0
        valid_codes_count_B = 0
        
        # Calculate weighted scores for each code
        for code_name, occurrences in code_occurrences.items():
            if code_name in historical_data.keys():
                count_model0 = historical_data[code_name].get(self.model_options[0], 0)
                count_model1 = historical_data[code_name].get(self.model_options[1], 0)
                total_count = count_model0 + count_model1
                
                # Skip codes with no historical data
                if total_count == 0:
                    continue
                
                # Calculate probabilities
                prob_model0 = count_model0 / total_count
                prob_model1 = count_model1 / total_count
                
                # Update scores based on code occurrences
                if occurrences.get("OUTPUT A", False):  # Code occurs in OUTPUT A
                    score_A_model0 += prob_model0
                    score_A_model1 += prob_model1
                else:
                    score_A_model0 += (1 - prob_model0)
                    score_A_model1 += (1 - prob_model1)
                valid_codes_count_A += 1
                
                if occurrences.get("OUTPUT B", False):  # Code occurs in OUTPUT B
                    score_B_model0 += prob_model0
                    score_B_model1 += prob_model1
                else:
                    score_B_model0 += (1 - prob_model0)
                    score_B_model1 += (1 - prob_model1)
                valid_codes_count_B += 1

        # Normalize scores by the number of valid codes
        if valid_codes_count_A > 0:
            score_A_model0 /= valid_codes_count_A
            score_A_model1 /= valid_codes_count_A
        if valid_codes_count_B > 0:
            score_B_model0 /= valid_codes_count_B
            score_B_model1 /= valid_codes_count_B
        
        # Make predictions ensuring outputs are from different models
        # First, get the initial independent predictions
        # ori_score_A_model0 = score_A_model0
        # ori_score_A_model1 = score_A_model1
        # score_A_model0 = score_A_model0 + 1e-6 / (ori_score_A_model0 + ori_score_A_model1 + 2e-6)
        # score_A_model1 = score_A_model1 + 1e-6 / (ori_score_A_model0 + ori_score_A_model1 + 2e-6)

        # ori_score_B_model0 = score_B_model0
        # ori_score_B_model1 = score_B_model1
        # score_B_model0 = score_B_model0 + 1e-6 / (ori_score_B_model0 + ori_score_B_model1 + 2e-6)
        # score_B_model1 = score_B_model1 + 1e-6 / (ori_score_B_model0 + ori_score_B_model1 + 2e-6)

        initial_pred_A = self.model_options[0] if score_A_model0 > score_A_model1 else self.model_options[1]
        initial_pred_B = self.model_options[0] if score_B_model0 > score_B_model1 else self.model_options[1]
        
        # If both are predicted to be the same model, we need to reassign
        if initial_pred_A == initial_pred_B:
            if initial_pred_A == self.model_options[0]:
                # Both predicted to be model 0, assign based on which has higher score for model 0
                if score_A_model0 > score_B_model0:
                    pred_A = self.model_options[0]
                    pred_B = self.model_options[1]
                else:
                    pred_A = self.model_options[1]
                    pred_B = self.model_options[0]
            else:
                # Both predicted to be model 1, assign based on which has higher score for model 1
                if score_A_model1 > score_B_model1:
                    pred_A = self.model_options[1]
                    pred_B = self.model_options[0]
                else:
                    pred_A = self.model_options[0]
                    pred_B = self.model_options[1]
        else:
            # Different predictions, use them as is
            pred_A = initial_pred_A
            pred_B = initial_pred_B
        
        if print_details:
            # print("Soft Voting Classification:", flush=True)
            print(f"OUTPUT A scores - {self.model_options[0]}: {score_A_model0:.3f}, {self.model_options[1]}: {score_A_model1:.3f}", flush=True)
            print(f"OUTPUT B scores - {self.model_options[0]}: {score_B_model0:.3f}, {self.model_options[1]}: {score_B_model1:.3f}", flush=True)
        
        if return_scores:
            return pred_A, pred_B, [score_A_model0, score_A_model1], [score_A_model0, score_A_model1]
        return pred_A, pred_B
    
    def _hard_voting_classification(self, classification, print_details=True, return_scores=False):
        """
        Classifies outputs using hard voting by counting codes associated with each model.
        For each output, count how many codes that appear in it are more associated with model 0 vs model 1.
        
        Parameters:
        - classification (str): The classification output text
        - labels (list): Ground truth labels [model_a_true, model_b_true]
        - print_details (bool): Whether to print detailed voting information
        
        Returns:
        - tuple: (evaluation_result, predicted_labels)
        """
        # Parse code occurrences from the classification text using default OUTPUT A/B names
        code_occurrences = self._parse_codes_occurence(classification)
        
        # Get historical code occurrence data
        historical_data = self._most_recent_code_occurrence_overall_train if self._most_recent_code_occurrence_overall_train is not None else self.code_occurrence_overall["train"]
        
        # Initialize code counts for each output and model
        codes_A_model0 = 0  # Count of codes in OUTPUT A that are more associated with model_options[0]
        codes_A_model1 = 0  # Count of codes in OUTPUT A that are more associated with model_options[1]
        codes_B_model0 = 0  # Count of codes in OUTPUT B that are more associated with model_options[0]
        codes_B_model1 = 0  # Count of codes in OUTPUT B that are more associated with model_options[1]
        
        valid_codes_count = 0
        
        # Count codes for each output
        for code_name, occurrences in code_occurrences.items():
            # Get historical occurrence counts for this code
            if code_name in historical_data:
                count_model0 = historical_data[code_name].get(self.model_options[0], 0)
                count_model1 = historical_data[code_name].get(self.model_options[1], 0)
                total_count = count_model0 + count_model1
                
                # Skip codes with no historical data
                if total_count == 0:
                    continue
                
                # Determine which model the code is more associated with
                if count_model0 > count_model1:
                    # Code is more associated with model 0
                    model0_associated = True
                elif count_model1 > count_model0:
                    # Code is more associated with model 1
                    model0_associated = False
                else:
                    # Equal association, skip this code for hard voting
                    continue
                
                # Count codes
                if occurrences.get("OUTPUT A", False):  # Code occurs in OUTPUT A
                    if model0_associated:
                        codes_A_model0 += 1
                    else:
                        codes_A_model1 += 1
                
                if occurrences.get("OUTPUT B", False):  # Code occurs in OUTPUT B
                    if model0_associated:
                        codes_B_model0 += 1
                    else:
                        codes_B_model1 += 1
                
                valid_codes_count += 1
        
        # Make predictions based on code counts
        # First, get the initial independent predictions
        if codes_A_model0 > codes_A_model1:
            initial_pred_A = self.model_options[0]
        elif codes_A_model1 > codes_A_model0:
            initial_pred_A = self.model_options[1]
        else:
            # Tie, default to first model option
            initial_pred_A = self.model_options[0]
        
        if codes_B_model0 > codes_B_model1:
            initial_pred_B = self.model_options[0]
        elif codes_B_model1 > codes_B_model0:
            initial_pred_B = self.model_options[1]
        else:
            # Tie, default to first model option
            initial_pred_B = self.model_options[0]
        
        # If both are predicted to be the same model, we need to reassign
        if initial_pred_A == initial_pred_B:
            if initial_pred_A == self.model_options[0]:
                # Both predicted to be model 0, assign model 0 to the output with more codes for model 0
                if codes_A_model0 > codes_B_model0:
                    pred_A = self.model_options[0]
                    pred_B = self.model_options[1]
                else:
                    pred_A = self.model_options[1]
                    pred_B = self.model_options[0]
            else:
                # Both predicted to be model 1, assign model 1 to the output with more codes for model 1
                if codes_A_model1 > codes_B_model1:
                    pred_A = self.model_options[1]
                    pred_B = self.model_options[0]
                else:
                    pred_A = self.model_options[0]
                    pred_B = self.model_options[1]
        else:
            # Different predictions, use them as is
            pred_A = initial_pred_A
            pred_B = initial_pred_B
        
        if print_details:
            # print("Hard Voting Classification:", flush=True)
            print(f"OUTPUT A code counts - {self.model_options[0]}: {codes_A_model0}, {self.model_options[1]}: {codes_A_model1}", flush=True)
            print(f"OUTPUT B code counts - {self.model_options[0]}: {codes_B_model0}, {self.model_options[1]}: {codes_B_model1}", flush=True)
            print(f"Valid codes used: {valid_codes_count}", flush=True)
        
        if return_scores:
            return pred_A, pred_B, [codes_A_model0, codes_A_model1], [codes_B_model0, codes_B_model1]
        return pred_A, pred_B

    def _naive_bayes_classification(self, classification, smoothing=1e-6, 
                                    follow_code_occurrence_exactly=False, print_details=True, return_scores=False):
        """
        Classifies outputs using Bernoulli naive Bayes based on rule occurrences as binary features.
        
        Parameters:
        - classification (str): The classification output text
        - labels (list): Ground truth labels [model_a_true, model_b_true]
        - phase (str): The phase to use for historical data ("train" or "eval")
        - smoothing (float): Laplace smoothing parameter to handle zero probabilities
        
        Returns:
        - tuple: (evaluation_result, predicted_labels)
        """
        # Parse code occurrences from the classification text using default OUTPUT A/B names
        code_occurrences = self._parse_codes_occurence(classification)
        
        # Get historical code occurrence data
        historical_data = self._most_recent_code_occurrence_overall_train if self._most_recent_code_occurrence_overall_train is not None else self.code_occurrence_overall["train"]
        
        # Get total number of training samples processed
        # Since we have equal numbers of both models in training, each model appears in num_train_samples outputs
        total_outputs_per_model = getattr(self, 'num_train_samples', 0)
        
        # If we don't have training data yet, use small uniform priors
        if total_outputs_per_model == 0:
            total_outputs_per_model = 1
            
        total_outputs_model0 = total_outputs_per_model
        total_outputs_model1 = total_outputs_per_model
        
        # Calculate prior probabilities P(model)
        total_outputs = total_outputs_model0 + total_outputs_model1
        prior_model0 = total_outputs_model0 / total_outputs if total_outputs > 0 else 0.5
        prior_model1 = total_outputs_model1 / total_outputs if total_outputs > 0 else 0.5
        
        # Initialize log probabilities (using log to avoid numerical underflow)
        log_prob_A_model0 = np.log(prior_model0)
        log_prob_A_model1 = np.log(prior_model1)
        log_prob_B_model0 = np.log(prior_model0)
        log_prob_B_model1 = np.log(prior_model1)
        
        # For each rule in the codebook, calculate P(feature | model)
        for code_name in self.codebook.keys():
            # Get historical counts for this code
            if code_name in historical_data:
                count_model0 = historical_data[code_name].get(self.model_options[0], 0)
                count_model1 = historical_data[code_name].get(self.model_options[1], 0)
            else:
                count_model0 = 0
                count_model1 = 0
                # Right now, if we don't have historical data for a code, we skip it.
                continue
            
            # Calculate P(feature=1 | model) with Laplace smoothing
            prob_feature_given_model0 = (count_model0 + smoothing) / (total_outputs_model0 + 2 * smoothing)
            prob_feature_given_model1 = (count_model1 + smoothing) / (total_outputs_model1 + 2 * smoothing)
            
            # Calculate P(feature=0 | model) = 1 - P(feature=1 | model)
            prob_no_feature_given_model0 = 1 - prob_feature_given_model0
            prob_no_feature_given_model1 = 1 - prob_feature_given_model1
            
            # Check if this code occurs in OUTPUT A
            # Right now, let's only consider the codes that are in the annotations.
            if follow_code_occurrence_exactly and (code_name not in code_occurrences.keys()):
                continue
            feature_A = code_occurrences.get(code_name, {}).get("OUTPUT A", False)
            if feature_A:
                log_prob_A_model0 += np.log(prob_feature_given_model0)
                log_prob_A_model1 += np.log(prob_feature_given_model1)
            else:
                log_prob_A_model0 += np.log(prob_no_feature_given_model0)
                log_prob_A_model1 += np.log(prob_no_feature_given_model1)
        
            # Check if this code occurs in OUTPUT B
            # Right now, let's only consider the codes that are in the codebook.
            feature_B = code_occurrences.get(code_name, {}).get("OUTPUT B", False)
            if feature_B:
                log_prob_B_model0 += np.log(prob_feature_given_model0)
                log_prob_B_model1 += np.log(prob_feature_given_model1)
            else:
                log_prob_B_model0 += np.log(prob_no_feature_given_model0)
                log_prob_B_model1 += np.log(prob_no_feature_given_model1)
        
        # Convert back to probabilities (optional, for display purposes)
        prob_A_model0 = np.exp(log_prob_A_model0)
        prob_A_model1 = np.exp(log_prob_A_model1)
        prob_B_model0 = np.exp(log_prob_B_model0)
        prob_B_model1 = np.exp(log_prob_B_model1)
        
        # Normalize probabilities
        norm_A = prob_A_model0 + prob_A_model1
        norm_B = prob_B_model0 + prob_B_model1
        
        if norm_A > 0:
            prob_A_model0 /= norm_A
            prob_A_model1 /= norm_A
        
        if norm_B > 0:
            prob_B_model0 /= norm_B
            prob_B_model1 /= norm_B
        
        # Make predictions ensuring outputs are from different models
        # First, get the initial independent predictions
        initial_pred_A = self.model_options[0] if log_prob_A_model0 > log_prob_A_model1 else self.model_options[1]
        initial_pred_B = self.model_options[0] if log_prob_B_model0 > log_prob_B_model1 else self.model_options[1]
        
        # If both are predicted to be the same model, we need to reassign
        if initial_pred_A == initial_pred_B:
            if initial_pred_A == self.model_options[0]:
                # Both predicted to be model 0, assign based on which has higher confidence for model 0
                if log_prob_A_model0 > log_prob_B_model0:
                    pred_A = self.model_options[0]
                    pred_B = self.model_options[1]
                else:
                    pred_A = self.model_options[1]
                    pred_B = self.model_options[0]
            else:
                # Both predicted to be model 1, assign based on which has higher confidence for model 1
                if log_prob_A_model1 > log_prob_B_model1:
                    pred_A = self.model_options[1]
                    pred_B = self.model_options[0]
                else:
                    pred_A = self.model_options[0]
                    pred_B = self.model_options[1]
        else:
            # Different predictions, use them as is
            pred_A = initial_pred_A
            pred_B = initial_pred_B
        
        if print_details:
            # print("Naive Bayes Classification:", flush=True)
            print(f"Prior probabilities - {self.model_options[0]}: {prior_model0:.3f}, {self.model_options[1]}: {prior_model1:.3f}", flush=True)
            print(f"OUTPUT A probabilities - {self.model_options[0]}: {prob_A_model0:.3f}, {self.model_options[1]}: {prob_A_model1:.3f}", flush=True)
            print(f"OUTPUT B probabilities - {self.model_options[0]}: {prob_B_model0:.3f}, {self.model_options[1]}: {prob_B_model1:.3f}", flush=True)
            
        if return_scores:
            return pred_A, pred_B, [prob_A_model0, prob_A_model1], [prob_B_model0, prob_B_model1]
        return pred_A, pred_B

    def print_codebook(self, selected_codes=[], common_code=True):
        """
        Print the codebook in a nicely formatted way.
        Parameters:
        - codebook (dict): The codebook with rule names as keys and definitions as values.
        """
        print("Rulebook:", flush=True)
        print("=" * 40, flush=True)
        for code_name, definition in self.codebook.items():
            if not common_code:
                # exhibited_by = definition[definition.lower().rfind("exhibited by"):]
                # model_cnt = 0
                # for model_option in self.model_options:
                #     if model_option.lower() in exhibited_by.lower():
                #         model_cnt += 1

                # if model_cnt > 0:
                #     continue
                # code_definition = self.codebook[code]
                if f"[Exhibited by {self.model_options[0]} and {self.model_options[1]}]" in definition or f"[Exhibited by {self.model_options[1]} and {self.model_options[0]}]" in definition:
                    continue

            if len(selected_codes) > 0:
                if code_name not in selected_codes and code_name.strip("*") not in selected_codes and code_name.strip("[").strip("]") not in selected_codes:
                    continue
            print(f"RuleName: {code_name}", flush=True)
            print(f"Definition: {definition}", flush=True)
            print("\n" + "-" * 40, flush=True)

    def _update_code_occurence(self, results_dict, code_occurrence, sample_id=None, is_decision_code=False):
        for code in code_occurrence.keys():
            matched_code = fuzz_match(code, list(results_dict.keys()), 90, method="character")
            # if code not in self.code_occurrence_overall["eval"].keys():
            if not matched_code:
                results_dict[code] = {}
                for model_option in self.model_options:
                    results_dict[code][model_option] = 0
                matched_code = code

            for model_option in self.model_options:
                if model_option not in code_occurrence[code].keys():
                    continue
                if code_occurrence[code][model_option]: 
                    results_dict[matched_code][model_option] += 1

    def warm_start(self, dataset, ckpt_path=None):
        if hasattr(self, "no_check_after_update"):
            self.no_check_after_update = False
        self.training_logs["train"] = {}
        self.code_occurrence_overall["train"] = {}
        self.decision_code_occurence_overall["train"] = {}
        self.warm_start_count = 0

        if (ckpt_path is not None) and os.path.exists(ckpt_path):
            print(f"Loading from pretrained attr from {ckpt_path}")
            self.from_pretrained(filename=ckpt_path)
        else:
            print("WARMUP FROM SCRATCH. No existing pretrained data found.")

        for idx in tqdm(range(len(dataset))):
            if idx < self.warm_start_count:
                continue

            classification_prompt, classification, evaluation, if_codebook_updated, final_output = self._update_codebook(dataset[idx]["outputs"], dataset[idx]["labels"], dataset[idx]['question'], qid=f"{dataset[idx]['id']}", if_codebook_updated=-1)

            code_occurrence, decision_code = self._parse_classification_codes(final_output, dataset[idx]["labels"])
            
            self._update_code_occurence(self.code_occurrence_overall["train"], code_occurrence, is_decision_code=False, sample_id=dataset[idx]['id'])
            self._update_code_occurence(self.decision_code_occurence_overall["train"], decision_code, is_decision_code=True)

            self.training_logs["train"][dataset[idx]['id']] = {"input_prompt": classification_prompt,
                                                                "output_classification": classification,
                                                                "evaluation": evaluation,
                                                                "updated": if_codebook_updated,
                                                                "data": dataset[idx],
                                                                "code_occurence": code_occurrence,
                                                                "decision_code": decision_code}
        
            self.warm_start_count += 1

        if ckpt_path is not None:
            self.save(ckpt_path)

        print(f"Warm start finished with {self.warm_start_count} examples")
        print("Current reasoning behavior taxonomy looks like")
        self.print_codebook()

    def prune_codebook(self, method="top_k", **kwargs):
        """
        Prune the codebook based on the code occurrence.
        
        Args:
            method (str): Pruning method to use. Options:
                - "frequency": Original ratio-based pruning (default)
                - "top_k": Keep only the k most discriminative codes
            **kwargs: Additional parameters for specific pruning methods
                
                For frequency method:
                - min_occurrences (int): Minimum total occurrences to consider (default: 10)
                - equality_threshold (float): Threshold for equal distribution (default: 0.05)
                
                For top_k method:
                - k (int): Number of most discriminative codes to keep (required)
        """
        print("Base model pruning method")
        if method == "frequency":
            self._prune_codebook_frequency(**kwargs)
        elif method == "top_k":
            self._prune_codebook_top_k(**kwargs)
        else:
            print(f"Unknown pruning method: {method}. Using frequency method as fallback.")
            self._prune_codebook_frequency(**kwargs)
    
    def _prune_codebook_frequency(self, min_occurrences=10, equality_threshold=0.05):
        """
        Original frequency-based pruning method.
        """
        to_be_deleted_code = []
        for code in self.codebook.keys():
            # Remove code that occurs significant amount of time but equally occurs in both models
            if code not in self.code_occurrence_overall["train"].keys():
                continue
            total_count = self.code_occurrence_overall["train"][code][self.model_options[0]] + self.code_occurrence_overall["train"][code][self.model_options[1]]
            if total_count > min_occurrences:
                ratio = self.code_occurrence_overall["train"][code][self.model_options[0]] / total_count
                lower_bound = 0.5 - equality_threshold
                upper_bound = 0.5 + equality_threshold
                if lower_bound < ratio < upper_bound:
                    to_be_deleted_code.append(code)
        for code in to_be_deleted_code:
            del self.codebook[code]
            print(f"Prune out {code}")
        print(f"Now the codebook has {len(self.codebook.keys())} codes.")
    
    def _prune_codebook_top_k(self, k=None):
        """
        Keep only the top k most discriminative codes based on occurrence ratios.
        """
        if k is None:
            print("Error: k parameter is required for top_k pruning method.")
            k = self.max_rule
            print(f"Using max_rule as k: {k}")
            
        # Calculate discriminative scores for all codes
        code_scores = {}
        for code in self.codebook.keys():
            if code not in self.code_occurrence_overall["train"].keys():
                code_scores[code] = 0  # No data means low discriminative power
                continue
            
            count0 = self.code_occurrence_overall["train"][code][self.model_options[0]]
            count1 = self.code_occurrence_overall["train"][code][self.model_options[1]]
            total_count = count0 + count1
            
            if total_count == 0:
                code_scores[code] = 0
            else:
                # Calculate discriminative power as distance from equal distribution (0.5)
                ratio = count0 / total_count
                discriminative_score = abs(ratio - 0.5) * total_count  # Weight by frequency
                code_scores[code] = discriminative_score
        
        # Sort codes by discriminative power and keep top k
        sorted_codes = sorted(code_scores.items(), key=lambda x: x[1], reverse=True)
        
        current_size = len(self.codebook)
        k = min(k, current_size)
        
        if k >= current_size:
            print(f"Requested k={k} is >= current codebook size ({current_size}). No pruning needed.")
            return
        
        # Remove codes not in top k
        codes_to_keep = [code for code, score in sorted_codes[:k]]
        codes_to_remove = [code for code in self.codebook.keys() if code not in codes_to_keep]
        
        print(f"Keeping top {k} most discriminative codes out of {current_size}")
        print(f"Top 5 codes: {[code for code, score in sorted_codes[:5]]}")
        
        for code in codes_to_remove:
            del self.codebook[code]
            print(f"Prune out {code}")
        print(f"Now the codebook has {len(self.codebook.keys())} codes.")

    def check_stop_criteria(self, criteria=["max_rules"]):
        meet_criteria = 0
        # if self.min_num_train_samples > self.num_train_samples:
        #     return False
        batch_update_size = 1
        if hasattr(self, 'batch_update_size') and self.batch_update_size is not None:
            batch_update_size = self.batch_update_size

        if "max_rules" in criteria:
            if len(self.codebook.keys()) > self.max_rule:
                self.prune_codebook()
                self.num_consecutive_turns_without_update = 0
                
                if len(self.codebook.keys()) > self.max_rule and self.num_train_samples * batch_update_size > self.min_num_train_samples:
                    print(f"The reasoning behavior taxonomy now has {len(self.codebook.keys())} reasoning behaviors.\nExit the training earlier.")
                    meet_criteria += 1

        if "global_patience" in criteria:
            if self.num_consecutive_turns_without_update >= self.global_patience and self.num_train_samples * batch_update_size > self.min_num_train_samples:
                print(f"The reasoning behavior taxonomy has not been changed for {self.num_consecutive_turns_without_update} turns in a row.\nExit the training earlier.")
                meet_criteria += 1
            
        if "max_train_samples" in criteria:
            if self.num_train_samples > self.max_train_samples:
                print(f"The coder has been trained on {self.num_train_samples} exceed the limit {self.max_train_samples}. Exit the training early.")
                meet_criteria += 1

        return meet_criteria > 0


    def train(self, dataset, verbosity=5, custom_logger=None, logger_kwargs=None, ckpt_path=None, batch_size=32):
        # self.training_logs["train"] = {}
        # self.code_occurrence_overall["train"] = {}
        # self.decision_code_occurence_overall["train"] = {}
        self.num_train_samples = 0
        self.num_train_correct = 0
        self.train_correct_ids = []
        self.train_wrong_ids = []

        if (ckpt_path is not None) and os.path.exists(ckpt_path):
            print(f"Loading from pretrained attr from {ckpt_path}")
            self.from_pretrained(filename=ckpt_path)
        else:
            print("TRAIN FROM SCRATCH. No existing pretrained data found.")

        start_at = self.num_train_samples
        
        # Phase 1: Individual processing with potential codebook updates
        idx = start_at
        while idx < len(dataset) and self.in_training:
            if self.batch_update and self.batch_update_size > 1:
                batched_dataset = random.sample(dataset, self.batch_update_size - 1)
                batched_outputs = [item["outputs"] for item in batched_dataset]
                batched_labels = [item["labels"] for item in batched_dataset]
                batched_questions = [item["question"] for item in batched_dataset]
                batched_ids = [item["id"] for item in batched_dataset]
                classification_prompt, classification, evaluation, if_codebook_updated, final_output = self.classify(dataset[idx]["outputs"], dataset[idx]["labels"], dataset[idx]["question"], train=self.in_training and len(self.codebook) <= self.max_rule, qid=f"{dataset[idx]['id']}", batched=False, additional_outputs=batched_outputs, additional_labels=batched_labels, additional_questions=batched_questions)
                
            else:
                classification_prompt, classification, evaluation, if_codebook_updated, final_output = self.classify(dataset[idx]["outputs"], dataset[idx]["labels"], dataset[idx]["question"], train=self.in_training, qid=f"{dataset[idx]['id']}")

            code_occurrence, decision_code = self._parse_classification_codes(final_output, dataset[idx]["labels"])

            self.training_logs["train"][dataset[idx]['id']] = {"input_prompt": classification_prompt,
                                                               "output_classification": classification,
                                                               "evaluation": evaluation,
                                                               "updated": if_codebook_updated,
                                                               "code_occurrence": code_occurrence,
                                                               "decision_code": decision_code,
                                                               "data": dataset[idx],
                                                               "cb_ckpt": deepcopy(self.codebook),
                                                               }
            
            self.code_occurrence_overall["train"] = deepcopy(self._most_recent_code_occurrence_overall_train)
            self._most_recent_code_occurrence_overall_train = None
            self.decision_code_occurence_overall["train"] = deepcopy(self._most_recent_decision_code_overall_train)
            self._most_recent_decision_code_overall_train = None

            if self.batch_update and self.batch_update_size > 1:
                self._update_code_occurence(self.code_occurrence_overall["train"], code_occurrence, is_decision_code=False, sample_id=f"{dataset[idx]['id']}")
                self._update_code_occurence(self.decision_code_occurence_overall["train"], decision_code, is_decision_code=True)

            if if_codebook_updated > 0:
                self.train_wrong_ids.append(dataset[idx]["id"])
            elif evaluation == "correct":
                self.train_correct_ids.append(dataset[idx]["id"])
                self.num_train_correct += 1
            
            self.num_train_samples += 1

            if custom_logger:
                custom_logger(**logger_kwargs)

            self.training_logs["train"][dataset[idx]['id']]["current_train_acc"] = (self.num_train_correct / self.num_train_samples) * 100

            if verbosity > 0 and idx % verbosity == 0:
                accuracy = (self.num_train_correct / self.num_train_samples) * 100
                print(f"Training Progress: {idx+1}/{len(dataset)} ({(idx+1)/len(dataset)*100:.2f}%)", flush=True)
                print(f"Accuracy so far: {accuracy:.2f}% ({self.num_train_correct}/{self.num_train_samples})", flush=True)
                print("-" * 50, flush=True)
                if ckpt_path is not None:
                    self.save(ckpt_path)

            # Increment idx before checking stop criteria since current sample is already processed
            idx += 1

            if self.in_training and self.check_stop_criteria(criteria=self.stop_criteria):
                self.in_training = False
                print("STOP CRITERIA REACHED: Stop adding new distinguishing reasoning behaviors / updating distinguishing reasoning behaviors definition in the current codebook.")
                print("Apply evaluation on the rest training data using batch processing. Update code occurence.")
                break

        # Phase 2: Batch processing for remaining samples (if any)
        # if len(self.codebook.keys()) >= self.max_rule: 
        #     if len(dataset) - idx > 10:
        #         print("Run 10 more data points")
        #         dataset = dataset[:idx + 10]
        #     else:
        #         print(f"Run {len(dataset) - idx} more data points")
        # else:
        #     print("Early Exit the training")
        #     dataset = dataset[:50]
        # if idx  50:
        print(f"Early Exit the training. Run without codebook update until {self.min_num_train_samples} samples")
        
        dataset = dataset[:self.min_num_train_samples]

        if idx < len(dataset):
            print(f"Processing remaining {len(dataset) - idx} samples with batch generation...")
            
            for i in tqdm(range(idx, len(dataset), batch_size)):
                batch = dataset[i:i+batch_size]
                outputs = [item["outputs"] for item in batch]
                labels = [item["labels"] for item in batch]
                questions = [item["question"] for item in batch]
                ids = [item["id"] for item in batch]
                
                classification_prompts, classifications, evaluations, if_codebook_updated_list, final_outputs = self.classify(outputs, labels, questions, train=False, qid=ids, batched=True)
                
                for j, (classification_prompt, classification, evaluation) in enumerate(zip(classification_prompts, classifications, evaluations)):
                    code_occurrence, decision_code = self._parse_classification_codes(final_outputs[j], labels[j])
                    if_codebook_updated = if_codebook_updated_list[j]
                    
                    self.training_logs["train"][ids[j]] = {"input_prompt": classification_prompt,
                                                           "output_classification": classification,
                                                           "evaluation": evaluation,
                                                           "updated": if_codebook_updated,
                                                           "code_occurrence": code_occurrence,
                                                           "decision_code": decision_code,
                                                           "data": batch[j],
                                                           "cb_ckpt": deepcopy(self.codebook),
                                                           }
                    
                    self._update_code_occurence(self.code_occurrence_overall["train"], code_occurrence, is_decision_code=False, sample_id=ids[j])
                    self._update_code_occurence(self.decision_code_occurence_overall["train"], decision_code, is_decision_code=True)

                    if if_codebook_updated > 0:
                        self.train_wrong_ids.append(ids[j])
                    elif evaluation == "correct":
                        self.train_correct_ids.append(ids[j])
                        self.num_train_correct += 1
                    else:
                        self.train_wrong_ids.append(dataset[idx]["id"])
                    self.num_train_samples += 1

                    if custom_logger:
                        custom_logger(**logger_kwargs)

                    self.training_logs["train"][ids[j]]["current_train_acc"] = (self.num_train_correct / self.num_train_samples) * 100

                    if verbosity > 0 and self.num_train_samples % verbosity == 0:
                        accuracy = (self.num_train_correct / self.num_train_samples) * 100
                        print(f"Training Progress: {self.num_train_samples}/{len(dataset)} ({self.num_train_samples/len(dataset)*100:.2f}%)", flush=True)
                        print(f"Accuracy so far: {accuracy:.2f}% ({self.num_train_correct}/{self.num_train_samples})", flush=True)
                        print("-" * 50, flush=True)
                        if ckpt_path is not None:
                            self.save(ckpt_path)
                
        accuracy = (self.num_train_correct / self.num_train_samples) * 100
        print(f"Training Progress: {self.num_train_samples}/{len(dataset)} ({self.num_train_samples/len(dataset)*100:.2f}%)", flush=True)
        print(f"Overall Training Accuracy: {accuracy:.2f}% ({self.num_train_correct}/{self.num_train_samples})", flush=True)
        print("-" * 50, flush=True)

        self.training_logs["train_acc"] = accuracy

        self.print_codebook()

    def eval(self, dataset, verbosity=5, batched=False, batch_size=4, ckpt_path=None, freq_estimate=False):
        assert batched and batch_size > 0, "Current evaluation must be done using batch"
        if freq_estimate:
            split = "train_est"
        else:
            split = "eval"
        self.training_logs[split] = {}
        self.code_occurrence_overall[split] = {}
        self.decision_code_occurence_overall[split] = {}
        self._most_recent_code_occurrence_overall_train = None
        self._most_recent_decision_code_overall_train = None
        self.num_eval_samples = 0
        self.num_eval_correct = 0
        
        self.eval_correct_ids = []
        self.eval_wrong_ids = []
        if batch_size > 0:
            print("Report on every batch instead of verbosity")
            verbosity = batch_size

        if (ckpt_path is not None) and os.path.exists(ckpt_path):
            print(f"Loading from pretrained attr from {ckpt_path}")
            self.from_pretrained(filename=ckpt_path)
        else:
            print("EVAL FROM SCRATCH. No existing pretrained data found.")

        start_at = self.num_eval_samples
        if batched:
            for i in tqdm(range(start_at, len(dataset), batch_size)):
                # if self.num_eval_samples > i + batch_size:
                #     continue
                batch = dataset[i:i+batch_size]
                outputs = [item["outputs"] for item in batch]
                labels = [item["labels"] for item in batch]
                questions = [item["question"] for item in batch]
                ids = [item["id"] for item in batch]
                classification_prompts, classifications, evaluations, if_codebook_updated_list, final_outputs = self.classify(outputs, labels, questions, train=False, qid=ids, batched=True)
                for j, (classification_prompt, classification, evaluation) in enumerate(zip(classification_prompts, classifications, evaluations)):
                    code_occurrence, decision_code = self._parse_classification_codes(final_outputs[j], labels[j])
                    if_codebook_updated = if_codebook_updated_list[j]
                    self.training_logs[split][ids[j]] = {"input_prompt": classification_prompt,
                                                         "output_classification": classification,
                                                         "evaluation": evaluation,
                                                         "updated": if_codebook_updated,
                                                         "data": batch[j],
                                                         "code_occurence": code_occurrence,
                                                         "decision_code": decision_code,}
                    
                    self._update_code_occurence(self.code_occurrence_overall[split], code_occurrence, is_decision_code=False, sample_id=ids[j])
                    self._update_code_occurence(self.decision_code_occurence_overall[split], decision_code, is_decision_code=True)

                    if evaluation == "correct":
                        self.eval_correct_ids.append(ids[j])
                        self.num_eval_correct += 1
                    else:
                        self.eval_wrong_ids.append(ids[j])
                    self.num_eval_samples += 1
                
                    if verbosity > 0 and self.num_eval_samples % verbosity == 0:
                        accuracy = (self.num_eval_correct / self.num_eval_samples) * 100
                        print(f"Validation Progress: {self.num_eval_samples}/{len(dataset)} ({self.num_eval_samples/len(dataset)*100:.2f}%)", flush=True)
                        print(f"Accuracy so far: {accuracy:.2f}% ({self.num_eval_correct}/{self.num_eval_samples})", flush=True)
                        print("-" * 50, flush=True)

                        if ckpt_path is not None:
                            self.save(ckpt_path)
        else:
            for idx in tqdm(range(start_at, len(dataset))):
                # if self.num_eval_samples > idx:
                    # continue
                classification_prompt, classification, evaluation, if_codebook_updated, final_output = self.classify(dataset[idx]["outputs"], 
                                                                                                                        dataset[idx]["labels"], 
                                                                                                                        dataset[idx]["question"], 
                                                                                                                        train=False, 
                                                                                                                        qid=f"train {dataset[idx]['id']}")

                self.training_logs[split][dataset[idx]["id"]] = {"input_prompt": classification_prompt,
                                                                "output_classification": classification,
                                                                "evaluation": evaluation,
                                                                "updated": if_codebook_updated,
                                                                "data": dataset[idx],
                                                                "code_occurence": code_occurrence,
                                                                "decision_code": decision_code,}

                code_occurrence, decision_code = self._parse_classification_codes(final_output, dataset[idx]["labels"])

                self._update_code_occurence(self.code_occurrence_overall[split], code_occurrence, is_decision_code=False, sample_id=dataset[idx]['id'])
                self._update_code_occurence(self.decision_code_occurence_overall[split], decision_code, is_decision_code=True)

                if evaluation == "correct":
                    self.eval_correct_ids.append(dataset[idx]["id"])
                    self.num_eval_correct += 1
                else:
                    self.eval_wrong_ids.append(dataset[idx]["id"])
                self.num_eval_samples += 1

                if verbosity > 0 and idx % verbosity == 0:
                    accuracy = (self.num_eval_correct / self.num_eval_samples) * 100
                    print(f"Validation Progress: {idx+1}/{len(dataset)} ({(idx+1)/len(dataset)*100:.2f}%)", flush=True)
                    print(f"Accuracy so far: {accuracy:.2f}% ({self.num_eval_correct}/{self.num_eval_samples})", flush=True)
                    print("-" * 50, flush=True)

                    if ckpt_path is not None:
                        self.save(ckpt_path)

        if not freq_estimate:
            final_accuracy = (self.num_eval_correct / self.num_eval_samples) * 100
            print(f"\nFinal Validation Accuracy: {final_accuracy:.2f}% ({self.num_eval_correct}/{self.num_eval_samples})", flush=True)
            print(f"Correctly classified IDs: {self.eval_correct_ids}", flush=True)
            print(f"Incorrectly classified IDs: {self.eval_wrong_ids}", flush=True)
            self.training_logs["eval_acc"] = final_accuracy
            self.training_logs["eval_correct_ids"] = self.eval_correct_ids
            self.training_logs["eval_wrong_ids"] = self.eval_wrong_ids

        if ckpt_path is not None:
            self.save(ckpt_path)

    def save_coder(self, filename="coder.ckpt"):
        self.model = None
        self.tokenizer = None
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def save(self, filename="coder_attr.ckpt", exclude_attributes=["model", "tokenizer", "logistic_classifier", "knn_classifier", "scaler"]):
        attributes = {attr: value for attr, value in self.__dict__.items() if attr not in exclude_attributes}
        with open(filename, 'wb') as f:
            pickle.dump(attributes, f)

    def load_coder_attributes(self, filename="coder_attr.ckpt"):
        try:
            with open(filename, 'rb') as f:
                attributes = pickle.load(f)
                self.__dict__.update(attributes)
        except FileNotFoundError:
            print("File not found.")

    def from_pretrained(self, filename="coder_attr.ckpt"):
        self.load_coder_attributes(filename)
        if self.model == None or self.tokenizer == None:
            self.init_model()

    def merge_similar_codes(self, split="train", similarity_threshold=0.85, min_cluster_size=2, use_sentence_transformers=False):
        """
        Merge semantically similar codes in the codebook based on embedding similarity.
        
        Args:
            split (str): The split to use for updating code occurrences ("train", "eval", etc.)
            similarity_threshold (float): Cosine similarity threshold for merging codes (0.0 to 1.0)
            min_cluster_size (int): Minimum number of codes to form a cluster for merging
            use_sentence_transformers (bool): Whether to use sentence-transformers for better semantic embeddings
            
        Returns:
            dict: Dictionary tracking which codes have been merged to which code
        """
        if not self.codebook or len(self.codebook) <= 1:
            print("Codebook is empty or has only one code. No merging needed.")
            return {}
            
        if split not in self.code_occurrence_overall:
            print(f"Split '{split}' not found in code occurrence data. Cannot merge codes.")
            return {}
            
        print(f"Starting code merging process with {len(self.codebook)} codes...")
        
        # Step 1: Generate embeddings for all codes
        code_embeddings = self._generate_code_embeddings(use_sentence_transformers=use_sentence_transformers)
        
        if not code_embeddings:
            print("Failed to generate embeddings. Cannot merge codes.")
            return {}
            
        # Step 2: Find similar codes using clustering
        similar_code_groups = self._find_similar_code_groups(code_embeddings, similarity_threshold, min_cluster_size)
        
        if not similar_code_groups:
            print("No similar code groups found. No merging needed.")
            return {}
            
        # Step 3: Merge similar codes
        merge_tracking = {}
        codes_to_remove = set()
        
        for group in similar_code_groups:
            if len(group) < min_cluster_size:
                continue
                
            # Choose the code to keep (first one in the group, or could use other criteria)
            kept_code = group[0]
            codes_to_merge = group[1:]
            
            print(f"Merging codes into '{kept_code}': {codes_to_merge}")
            
            # Update merge tracking
            merge_tracking[kept_code] = codes_to_merge
            
            # Step 4: Merge code occurrences
            self._merge_code_occurrences(kept_code, codes_to_merge, split)
            
            # Mark codes for removal
            codes_to_remove.update(codes_to_merge)
        
        # Step 5: Remove merged codes from codebook
        for code_to_remove in codes_to_remove:
            if code_to_remove in self.codebook:
                del self.codebook[code_to_remove]
                print(f"Removed code: {code_to_remove}")
        
        print(f"Code merging completed. Codebook now has {len(self.codebook)} codes.")
        print(f"Merged {len(codes_to_remove)} codes into {len(merge_tracking)} groups.")
        
        return merge_tracking
    
    def _generate_code_embeddings(self, use_sentence_transformers=False):
        """
        Generate embeddings for all codes in the codebook.
        
        Args:
            use_sentence_transformers (bool): Whether to use sentence-transformers for better embeddings
        
        Returns:
            dict: Dictionary mapping code names to their embeddings
        """
        code_embeddings = {}
        code_texts = []
        code_names = []
        
        # Prepare texts for embedding (code name + definition)
        for code_name, definition in self.codebook.items():
            code_text = f"{code_name}: {definition}"
            code_texts.append(code_text)
            code_names.append(code_name)
        
        if not code_texts:
            return {}
        
        try:
            if use_sentence_transformers:
                embeddings = self._get_sentence_transformer_embeddings(code_texts)
            elif self.use_vllm:
                # For vLLM, use token-based embeddings
                embeddings = self._get_vllm_embeddings(code_texts)
            else:
                # For transformers model, try to get embeddings directly
                embeddings = self._get_transformers_embeddings(code_texts)
                
            # Create mapping from code names to embeddings
            for i, code_name in enumerate(code_names):
                code_embeddings[code_name] = embeddings[i]
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return {}
        
        return code_embeddings
    
    def _get_sentence_transformer_embeddings(self, texts):
        """
        Get embeddings using sentence-transformers (if available).
        This provides much better semantic similarity than token-based approaches.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight but effective model
            if self.embed_model is None:
                self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = self.embed_model.encode(texts)
            
            print("Using sentence-transformers for high-quality semantic embeddings")
            return embeddings
            
        except ImportError:
            print("sentence-transformers not available, falling back to token-based embeddings")
            return self._get_token_based_embeddings(texts)
        except Exception as e:
            print(f"Error with sentence-transformers, falling back to token-based: {e}")
            return self._get_token_based_embeddings(texts)
    
    def _get_vllm_embeddings(self, texts):
        """
        Get embeddings using vLLM model.
        For vLLM, we'll use a token-based similarity approach since 
        accessing the model internals for embeddings is complex and version-dependent.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        # Use the shared token-based embedding method
        return self._get_token_based_embeddings(texts)
    
    def _get_transformers_embeddings(self, texts):
        """
        Get embeddings using transformers model.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        embeddings = []
        
        try:
            for text in texts:
                # Tokenize text
                tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
                
                with torch.no_grad():
                    # Get hidden states
                    outputs = self.model(**tokens, output_hidden_states=True)
                    
                    # Use last layer hidden states
                    last_hidden_states = outputs.hidden_states[-1]
                    
                    # Use mean pooling for sentence embedding
                    attention_mask = tokens['attention_mask']
                    masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
                    embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                    
                    embeddings.append(embedding.cpu().numpy().flatten())
        
        except Exception as e:
            print(f"Error with transformers embeddings, falling back to token-based approach: {e}")
            # Fallback to token-based embeddings similar to vLLM approach
            return self._get_token_based_embeddings(texts)
        
        return np.array(embeddings)
    
    def _get_token_based_embeddings(self, texts):
        """
        Fallback method to create embeddings based on token distributions.
        This works when we can't access model hidden states.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        embeddings = []
        
        for text in texts:
            # Use tokenizer to create a simple bag-of-words style embedding
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Create a simple embedding based on token IDs
            # Truncate to a reasonable length
            max_tokens = 512
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
            
            # Create a fixed-size embedding by binning token IDs
            embedding_dim = 768
            embedding = np.zeros(embedding_dim)
            
            for token_id in tokens:
                # Map token ID to embedding dimension
                idx = token_id % embedding_dim
                embedding[idx] += 1.0
            
            # Normalize the embedding
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _find_similar_code_groups(self, code_embeddings, similarity_threshold, min_cluster_size):
        """
        Find groups of similar codes using clustering.
        
        Args:
            code_embeddings (dict): Dictionary mapping code names to embeddings
            similarity_threshold (float): Similarity threshold for clustering
            min_cluster_size (int): Minimum cluster size
            
        Returns:
            list: List of groups, where each group is a list of similar code names
        """
        if len(code_embeddings) < 2:
            return []
        
        code_names = list(code_embeddings.keys())
        embeddings = np.array([code_embeddings[name] for name in code_names])
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Convert similarity to distance for clustering
        distance_matrix = 1 - similarity_matrix
        
        # Use agglomerative clustering
        distance_threshold = 1 - similarity_threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average',
            metric='precomputed'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group codes by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(code_names[i])
        
        # Filter clusters by minimum size
        similar_groups = [group for group in clusters.values() if len(group) >= min_cluster_size]
        
        print(f"Found {len(similar_groups)} similar code groups:")
        for i, group in enumerate(similar_groups):
            print(f"  Group {i+1}: {group}")
        
        return similar_groups
    
    def _merge_code_occurrences(self, kept_code, codes_to_merge, split):
        """
        Merge code occurrences from similar codes into the kept code by re-parsing all training samples.
        This correctly handles cases where multiple codes to be merged appear in the same classification.
        
        Args:
            kept_code (str): The code to keep
            codes_to_merge (list): List of codes to merge into the kept code
            split (str): The split to update occurrences for
        """
        if split not in self.training_logs:
            print(f"Split '{split}' not found in training logs. Cannot merge occurrences.")
            return
        
        # Get all codes that need to be considered as one (kept + codes to merge)
        all_merged_codes = [kept_code] + codes_to_merge
        
        # Initialize counters for the kept code
        merged_occurrences = {}
        merged_decision_occurrences = {}
         
        for model_option in self.model_options:
            merged_occurrences[model_option] = 0
            merged_decision_occurrences[model_option] = 0
        
        print(f"Re-parsing {len(self.training_logs[split])} training samples to merge codes: {all_merged_codes}")
        
        # Re-parse all training samples to count merged occurrences
        for sample_id, sample_data in self.training_logs[split].items():
            if not isinstance(sample_data, dict) or "output_classification" not in sample_data:
                continue
                
            # Get the classification and labels for this sample
            classification = sample_data["output_classification"]
            labels = sample_data["data"]["labels"]
            
            # Parse code occurrences for this sample
            if self.think_mode:
                eot_token = "</think>"
                eot_index = classification.rfind(eot_token)
                if eot_index >= 0:
                    final_output = classification[eot_index + len(eot_token):]
                else:
                    final_output = classification
            else:
                final_output = classification
            
            # Parse code and decision code occurrences
            code_occurrence = self._parse_codes_occurence(final_output, labels)
            decision_code = self._parse_decision_code(final_output)
            
                         # Check if any of the codes to be merged appear in this sample
             # Track which models had any merged code appear in this sample
            models_with_regular_code = set()
            models_with_decision_code = set()
             
            for code_name in all_merged_codes:
                # Check regular code occurrences
                if code_name in code_occurrence:
                    for model_option in self.model_options:
                        if code_occurrence[code_name].get(model_option, False):
                            models_with_regular_code.add(model_option)
                 
                # Check decision code occurrences
                if code_name in decision_code:
                    for model_option in self.model_options:
                        if model_option in decision_code[code_name] and decision_code[code_name][model_option]:
                            models_with_decision_code.add(model_option)
             
            # Increment count for each model that had any merged code appear (once per sample)
            for model_option in models_with_regular_code:
                merged_occurrences[model_option] += 1
             
            for model_option in models_with_decision_code:
                merged_decision_occurrences[model_option] += 1
        
        # Update the code occurrence overall for the kept code
        self.code_occurrence_overall[split][kept_code] = merged_occurrences
        
        # Update decision code occurrences if they exist
        if hasattr(self, 'decision_code_occurence_overall') and split in self.decision_code_occurence_overall:
            self.decision_code_occurence_overall[split][kept_code] = merged_decision_occurrences
        
        # Remove the merged codes from occurrence tracking
        for code_to_merge in codes_to_merge:
            if code_to_merge in self.code_occurrence_overall[split]:
                del self.code_occurrence_overall[split][code_to_merge]
                print(f"Removed '{code_to_merge}' from code occurrences")
            
            if hasattr(self, 'decision_code_occurence_overall') and split in self.decision_code_occurence_overall:
                if code_to_merge in self.decision_code_occurence_overall[split]:
                    del self.decision_code_occurence_overall[split][code_to_merge]
                    print(f"Removed '{code_to_merge}' from decision code occurrences")
        
        print(f"Merged code '{kept_code}' final occurrences: {merged_occurrences}")
        print(f"Merged codes removed: {codes_to_merge}")

    def filter_equally_occurring_codes(self, split="train", equality_threshold=1):
        """
        Filter out codes that occur equally across all models from the codebook.
        
        Args:
            split (str): The split to check occurrences in (default: "train")
            equality_threshold (float): Threshold for determining equality (default: 0.1)
                                      If the relative difference between model occurrences is below this,
                                      the code is considered equally occurring
        
        Returns:
            dict: Dictionary with information about filtered codes
        """
        if split not in self.code_occurrence_overall:
            print(f"Split '{split}' not found in code occurrence data. No filtering performed.")
            return {}
        
        if not self.codebook:
            print("Codebook is empty. No filtering needed.")
            return {}
        
        # Get current filter count for naming
        filter_count = 0
        for key in self.training_logs.keys():
            if key.startswith("codebook_before_") and key.endswith("_th_filter"):
                filter_count += 1
        filter_count += 1
        
        # Save original codebook before filtering
        original_codebook_key = f"codebook_before_{filter_count}_th_filter"
        self.training_logs[original_codebook_key] = deepcopy(self.codebook)
        
        codes_to_remove = []
        filtering_info = {
            "filtered_codes": [],
            "filtering_stats": {},
            "original_codebook_size": len(self.codebook),
            "filter_count": filter_count,
            "split_used": split,
            "equality_threshold": equality_threshold,
        }
        
        print(f"Starting code filtering process for split '{split}'...")
        print(f"Original codebook size: {len(self.codebook)}")
        
        # Check each code in the codebook
        for code_name in list(self.codebook.keys()):
            if code_name not in self.code_occurrence_overall[split]:
                # Code has no occurrences in this split, skip
                continue
            
            code_occurrences = self.code_occurrence_overall[split][code_name]
            
            # Get occurrence counts for all models
            model_counts = []
            total_occurrences = 0
            
            for model_option in self.model_options:
                count = code_occurrences.get(model_option, 0)
                model_counts.append(count)
                total_occurrences += count
            
            # Check if occurrences are equal across models
            if self._are_occurrences_equal(model_counts, equality_threshold):
                codes_to_remove.append(code_name)
                
                # Store detailed info about this code
                filtering_info["filtering_stats"][code_name] = {
                    "model_counts": {model: count for model, count in zip(self.model_options, model_counts)},
                    "total_occurrences": total_occurrences
                }
                
                print(f"Marking code '{code_name}' for removal - occurrences: {dict(zip(self.model_options, model_counts))}")
        
        # Remove the codes from codebook
        for code_name in codes_to_remove:
            del self.codebook[code_name]
            filtering_info["filtered_codes"].append(code_name)
        
        filtering_info["filtered_codebook_size"] = len(self.codebook)
        
        # Store filtering information in training logs
        self.training_logs[f"filtering_info_{filter_count}"] = filtering_info
        
        print(f"Filtering completed:")
        print(f"  - Removed {len(codes_to_remove)} codes")
        print(f"  - Codebook size: {filtering_info['original_codebook_size']} -> {filtering_info['filtered_codebook_size']}")
        print(f"  - Original codebook saved as '{original_codebook_key}'")
        
        if codes_to_remove:
            print(f"  - Removed codes: {codes_to_remove}")
        
        return filtering_info
    
    def _are_occurrences_equal(self, model_counts, equality_threshold):
        """
        Check if occurrence counts are equal between two models within the threshold.
        Equal means the percentage of any model is within (50-threshold, 50+threshold).
        
        Args:
            model_counts (list): List of occurrence counts for each model (assumes 2 models)
            equality_threshold (float): Threshold in percentage points (e.g., 10 for 10%)
            
        Returns:
            bool: True if occurrences are considered equal, False otherwise
        """
        if len(model_counts) != 2:
            return False
        
        count1, count2 = model_counts[0], model_counts[1]
        
        # Both are zero - considered equal
        if count1 == 0 and count2 == 0:
            return True
        
        # One is zero, the other is not - not equal
        if count1 == 0 or count2 == 0:
            return False
        
        # Calculate total occurrences
        total = count1 + count2
        
        # Calculate percentage for first model
        percentage1 = (count1 / total) * 100
        
        # Check if percentage is within (50-threshold, 50+threshold)
        lower_bound = 50 - equality_threshold
        upper_bound = 50 + equality_threshold
        
        return lower_bound < percentage1 < upper_bound
    

