from src.coder_vllm import CoderVLLM
from copy import deepcopy
import random
import os


class CoderBatchTrained(CoderVLLM):
    """
    A coder that combines the efficiency of CoderVLLM with batch training using 
    accumulation-update cycles. This class inherits from CoderVLLM and uses 
    codebook-based reasoning code parsing while implementing the training logic 
    from coder_bor.py.
    
    Training Process:
    1. Accumulation Phase: Process samples in batches without updating codebook
    2. Update Phase: Process samples individually with codebook updates until changes occur
    3. Repeat until stopping criteria are met
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize parent class
        super().__init__(*args, **kwargs)
        
        # Initialize training state variables
        self.num_update_samples = 0
        self.num_update_correct = 0
        self.max_train_samples = 1000  # Maximum training samples to prevent infinite loops
        
        print("CoderBatchTrained initialized with accumulation-update training logic")
    
    def train(self, dataset, verbosity=5, custom_logger=None, logger_kwargs=None, ckpt_path=None, 
              batch_size=32, accumulate_observation_training=False, accumulation_size=10, 
              sampling_training=False):
        """
        Override train method to add accumulate_observation_training feature.
        
        Args:
            dataset: Training dataset
            verbosity: Verbosity level  
            custom_logger: Custom logging function
            logger_kwargs: Logger keyword arguments
            ckpt_path: Checkpoint path
            batch_size: Batch size for processing
            accumulate_observation_training (bool): Whether to accumulate observations before updating
            accumulation_size (int): Number of samples to accumulate before attempting updates
            sampling_training (bool): Whether to use stochastic sampling instead of sequential iteration
        """
        # Initialize training state
        self.num_train_samples = 0
        self.num_train_correct = 0
        self.num_update_samples = 0
        self.num_update_correct = 0
        self.train_correct_ids = []
        self.train_wrong_ids = []
        self.in_training = True
        
        # Initialize training logs and code occurrence tracking
        if "train" not in self.training_logs:
            self.training_logs["train"] = {}
        if "train" not in self.code_occurrence_overall:
            self.code_occurrence_overall["train"] = {}
        if "train" not in self.decision_code_occurence_overall:
            self.decision_code_occurence_overall["train"] = {}
        
        # Load from checkpoint if available
        if (ckpt_path is not None) and os.path.exists(ckpt_path):
            print(f"Loading from pretrained attr from {ckpt_path}")
            self.from_pretrained(filename=ckpt_path)
        else:
            print("TRAIN FROM SCRATCH. No existing pretrained data found.")

        start_at = self.num_train_samples
        
        if accumulate_observation_training:
            if sampling_training:
                print(f"Using stochastic sampling accumulate observation training with accumulation size: {accumulation_size}")
                self._train_with_accumulation_stochastic_sampling(
                    dataset, start_at, verbosity, custom_logger, logger_kwargs, 
                    ckpt_path, batch_size, accumulation_size
                )
            else:
                print(f"Using sequential accumulate observation training with accumulation size: {accumulation_size}")
                self._train_with_accumulation(
                    dataset, start_at, verbosity, custom_logger, logger_kwargs, 
                    ckpt_path, batch_size, accumulation_size
                )
        else:
            # Use parent's training method
            super().train(dataset, verbosity, custom_logger, logger_kwargs, ckpt_path, batch_size)
            
        # Clear temporary variables
        self._most_recent_code_occurrence_overall_train = None
        self._most_recent_decision_code_overall_train = None
    
    def _train_with_accumulation(self, dataset, start_at, verbosity, custom_logger, logger_kwargs, 
                               ckpt_path, batch_size, accumulation_size):
        """
        Training with clean accumulation → update cycle:
        1. Accumulation Phase: Process accumulation_size samples with train=False (batched for speed)
        2. Update Phase: Process new samples with train=True until codebook changes
        3. Repeat until stop criteria met
        """
        idx = start_at
        in_accumulation_mode = True
        accumulation_count = 0
        samples_accumulated_this_round = 0
        accumulation_batch = []
        # Store classification results for proper evaluation after codebook updates
        pending_samples = []

        if self._extend_with_nan:
            print("Extend the vector with np.nan. Thus:")
            self.no_check_after_update = True
            print("No check after update:", self.no_check_after_update)
        
        while idx < len(dataset) and self.in_training:
            sample = dataset[idx]
            
            if in_accumulation_mode:
                # Accumulation Phase: Collect samples for batched processing (train=False)
                accumulation_batch.append(sample)
                samples_accumulated_this_round += 1
                
                # Process batch when we reach batch_size or complete accumulation_size
                should_process_batch = (
                    len(accumulation_batch) >= batch_size or 
                    samples_accumulated_this_round >= accumulation_size
                )
                
                if should_process_batch:
                    print(f"Processing accumulation batch: {len(accumulation_batch)} samples")
                    
                    # Prepare batch data
                    outputs = [s["outputs"] for s in accumulation_batch]
                    labels = [s["labels"] for s in accumulation_batch]
                    questions = [s["question"] for s in accumulation_batch]
                    ids = [s["id"] for s in accumulation_batch]
                    
                    # Batch classify with train=False and no_eval=True to get outputs first
                    classification_prompts, classifications, evaluations, if_codebook_updated_list, final_outputs = self.classify(
                        outputs, labels, questions, train=False, qid=ids, batched=True, no_eval=True,
                    )
                    
                    # Store results for later evaluation (after codebook updates)
                    for i, batch_sample in enumerate(accumulation_batch):
                        pending_samples.append({
                            'sample': batch_sample,
                            'classification_prompt': classification_prompts[i],
                            'classification': classifications[i],
                            'final_output': final_outputs[i],
                            'if_codebook_updated': 0  # Always 0 for accumulation
                        })
                    
                    # Clear the batch
                    accumulation_batch = []
                
                # Check if we've accumulated enough samples for this round
                if samples_accumulated_this_round >= accumulation_size:
                    accumulation_count += 1
                    print(f"Accumulation round {accumulation_count} complete: {samples_accumulated_this_round} samples processed")
                    
                    # Now process all pending samples for evaluation
                    self._process_pending_accumulation_samples(pending_samples, custom_logger, logger_kwargs)
                    pending_samples = []  # Clear after processing
                    
                    accuracy = (self.num_train_correct / self.num_train_samples) * 100 if self.num_train_samples > 0 else 0
                    print(f"Overall accuracy so far: {accuracy:.2f}% ({self.num_train_correct}/{self.num_train_samples})")
                    
                    print("Switching to UPDATE MODE")
                    samples_accumulated_this_round = 0
                    in_accumulation_mode = False
            
            else:
                # Update Phase: Process new samples with training enabled until codebook changes
                print(f"Update mode: Processing new sample {sample['id']}")
                self.num_update_samples += 1
                
                # Get current codebook size to detect changes
                current_codebook_size = len(self.codebook.keys()) if self.codebook else 0
                
                classification_prompt, classification, evaluation, if_codebook_updated, final_output = self.classify(
                    sample["outputs"], sample["labels"], sample["question"], 
                    train=True, qid=f"train {sample['id']}"
                )
                
                # Update statistics for the new sample
                self._update_sample_statistics(
                    sample, classification_prompt, classification, 
                    evaluation, if_codebook_updated, final_output, 
                    custom_logger, logger_kwargs
                )
                
                # Check if codebook was updated
                if if_codebook_updated > 0:
                    pass  # Codebook was updated
                elif evaluation == "correct":
                    self.num_update_correct += 1
                
                # Check if codebook size changed
                new_codebook_size = len(self.codebook.keys()) if self.codebook else 0
                
                if new_codebook_size != current_codebook_size:
                    print(f"Codebook updated! Size: {current_codebook_size} → {new_codebook_size}")
                    print("Returning to ACCUMULATION MODE")
                    in_accumulation_mode = True
                
                # Check stop criteria after each training sample in update mode
                if self.in_training and self.check_stop_criteria(criteria=self.stop_criteria):
                    self.in_training = False
                    print("STOP CRITERIA REACHED")
                    break
            
            idx += 1
            
            # Progress reporting
            if verbosity > 0 and idx % verbosity == 0:
                accuracy = (self.num_train_correct / self.num_train_samples) * 100 if self.num_train_samples > 0 else 0
                update_accuracy = (self.num_update_correct / self.num_update_samples) * 100 if self.num_update_samples > 0 else 0
                print(f"Training Progress: {idx}/{len(dataset)} ({idx/len(dataset)*100:.2f}%)", flush=True)
                print(f"Overall Accuracy: {accuracy:.2f}% ({self.num_train_correct}/{self.num_train_samples})", flush=True)
                print(f"Update Accuracy: {update_accuracy:.2f}% ({self.num_update_correct}/{self.num_update_samples})", flush=True)
                print(f"Mode: {'Accumulation' if in_accumulation_mode else 'Update'}", flush=True)
                if in_accumulation_mode:
                    print(f"Samples in current accumulation round: {samples_accumulated_this_round}/{accumulation_size}", flush=True)
                    print(f"Samples in current batch: {len(accumulation_batch)}/{batch_size}", flush=True)
                    print(f"Pending samples awaiting evaluation: {len(pending_samples)}", flush=True)
                print(f"Accumulation rounds completed: {accumulation_count}", flush=True)
                print("-" * 50, flush=True)
                if ckpt_path is not None:
                    self.save(ckpt_path)
        
        # Process any remaining samples in accumulation batch and pending samples
        if accumulation_batch:
            print(f"Processing final accumulation batch: {len(accumulation_batch)} samples")
            
            # Prepare batch data
            outputs = [s["outputs"] for s in accumulation_batch]
            labels = [s["labels"] for s in accumulation_batch]
            questions = [s["question"] for s in accumulation_batch]
            ids = [s["id"] for s in accumulation_batch]
            
            # Batch classify with train=False and no_eval=True
            classification_prompts, classifications, evaluations, if_codebook_updated_list, final_outputs = self.classify(
                outputs, labels, questions, train=False, qid=ids, batched=True, no_eval=True
            )
            
            # Add to pending samples
            for i, batch_sample in enumerate(accumulation_batch):
                pending_samples.append({
                    'sample': batch_sample,
                    'classification_prompt': classification_prompts[i],
                    'classification': classifications[i],
                    'final_output': final_outputs[i],
                    'if_codebook_updated': 0
                })
        
        # Process any remaining pending samples
        if pending_samples:
            print(f"Processing final pending samples: {len(pending_samples)} samples")
            self._process_pending_accumulation_samples(pending_samples, custom_logger, logger_kwargs)
        
        accuracy = (self.num_train_correct / self.num_train_samples) * 100 if self.num_train_samples > 0 else 0
        print(f"Overall Training Accuracy: {accuracy:.2f}% ({self.num_train_correct}/{self.num_train_samples})", flush=True)
        if "train" not in self.training_logs:
            self.training_logs["train"] = {}
        self.training_logs["train"]["final_accuracy"] = accuracy
        self.print_codebook()
    
    def _train_with_accumulation_stochastic_sampling(self, dataset, start_at, verbosity, custom_logger, 
                                                   logger_kwargs, ckpt_path, batch_size, accumulation_size):
        """
        Training with stochastic sampling accumulation → update cycle:
        1. Accumulation Phase: Sample accumulation_size samples WITHOUT replacement (batched for speed)
        2. Update Phase: Sample new samples WITH replacement until codebook changes
        3. Repeat until stop criteria met
        4. When no available indices remain, start a new epoch by resetting the available indices
        """
        if self._extend_with_nan:
            print("Extend the vector with np.nan. Thus:")
            self.no_check_after_update = True
            print("No check after update:", self.no_check_after_update)

        # Create indices for sampling (excluding samples before start_at)
        available_indices = list(range(start_at, len(dataset)))
        in_accumulation_mode = True
        accumulation_count = 0
        total_samples_processed = start_at
        current_epoch = 1
        
        while self.in_training or in_accumulation_mode:
            
            if in_accumulation_mode:
                # Accumulation Phase: Sample without replacement
                # Determine how many samples to take
                samples_to_take = min(accumulation_size, len(available_indices))
                
                if samples_to_take == 0:
                    # Start a new epoch by resetting available indices
                    current_epoch += 1
                    available_indices = list(range(len(dataset)))
                    samples_to_take = min(accumulation_size, len(available_indices))
                    print(f"Starting new epoch {current_epoch}: Reset available indices ({len(available_indices)} samples)")
                    
                    if samples_to_take == 0:
                        print("Dataset is empty, cannot continue training")
                        break
                
                # Sample indices without replacement
                sampled_indices = random.sample(available_indices, samples_to_take)
                # Remove sampled indices from available pool
                for idx in sampled_indices:
                    available_indices.remove(idx)
                
                # Get sampled data
                sampled_data = [dataset[idx] for idx in sampled_indices]
                
                print(f"Accumulation mode: Sampled {len(sampled_data)} samples for processing")
                
                # Process in batches for efficiency - store results for later evaluation
                accumulation_batch = []
                samples_processed_this_round = 0
                pending_samples = []
                
                for sample in sampled_data:
                    accumulation_batch.append(sample)
                    samples_processed_this_round += 1
                    
                    # Process batch when we reach batch_size or finish all samples
                    should_process_batch = (
                        len(accumulation_batch) >= batch_size or 
                        samples_processed_this_round >= len(sampled_data)
                    )
                    
                    if should_process_batch:
                        print(f"Processing accumulation batch: {len(accumulation_batch)} samples")
                        
                        # Prepare batch data
                        outputs = [s["outputs"] for s in accumulation_batch]
                        labels = [s["labels"] for s in accumulation_batch]
                        questions = [s["question"] for s in accumulation_batch]
                        ids = [s["id"] for s in accumulation_batch]
                        
                        # Batch classify with train=False and no_eval=True to get outputs first
                        classification_prompts, classifications, evaluations, if_codebook_updated_list, final_outputs = self.classify(
                            outputs, labels, questions, train=False, qid=ids, batched=True, no_eval=True,
                        )
                        
                        # Store results for later evaluation (after codebook updates)
                        for i, batch_sample in enumerate(accumulation_batch):
                            pending_samples.append({
                                'sample': batch_sample,
                                'classification_prompt': classification_prompts[i],
                                'classification': classifications[i],
                                'final_output': final_outputs[i],
                                'if_codebook_updated': 0  # Always 0 for accumulation
                            })
                            total_samples_processed += 1
                        
                        # Clear the batch
                        accumulation_batch = []
                
                # Complete accumulation round - now process all pending samples
                accumulation_count += 1
                print(f"Accumulation round {accumulation_count} complete: {samples_processed_this_round} samples processed")
                print(f"Available samples remaining: {len(available_indices)}")
                
                # Process all pending samples
                self._process_pending_accumulation_samples(pending_samples, custom_logger, logger_kwargs)
                
                accuracy = (self.num_train_correct / self.num_train_samples) * 100 if self.num_train_samples > 0 else 0
                print(f"Overall accuracy so far: {accuracy:.2f}% ({self.num_train_correct}/{self.num_train_samples})")

                if self.in_training:
                    print("Switching to UPDATE MODE")
                    in_accumulation_mode = False
                else:
                    in_accumulation_mode = False
                    print("Finish updating with the last update to the codebook. Quit Here")
                    break

                # Check stop criteria after each training sample in update mode
                if hasattr(self, "no_update_to_codebook") and self.no_update_to_codebook:
                    in_accumulation_mode = True
                    print("No Update to Codebook is one. Continue accumulation.\nAlways in accumulation model until reach to the maximum number of training samples.")
                    if self.num_train_samples >= self.max_train_samples:
                        print("Maximum number of training samples reached")
                        break

                if self.in_training and self.check_stop_criteria(criteria=self.stop_criteria):
                    self.in_training = False
                    print("STOP CRITERIA REACHED")
                    if in_accumulation_mode:
                        print("There was an update to the codebook")
                        print("Do one more round of accumulation before quiting")
                    else:
                        break

            else:
                # Update Phase: Sample with replacement until codebook changes
                print("Update mode:")
                self.num_update_samples += 1
                
                # Get current codebook size to detect changes
                current_codebook_size = len(self.codebook.keys()) if self.codebook else 0
                
                # Sample one sample with replacement from entire dataset
                sample_idx = random.choice(range(len(dataset)))
                sample = dataset[sample_idx]
                
                print(f"Update mode: Processing sample {sample['id']} (index {sample_idx})")
                
                classification_prompt, classification, evaluation, if_codebook_updated, final_output = self.classify(
                    sample["outputs"], sample["labels"], sample["question"], 
                    train=True, qid=f"{sample['id']}"
                )

                if evaluation != "update-no-check":
                    # Update statistics for the new sample
                    self._update_sample_statistics(
                        sample,
                        classification_prompt, 
                        classification, 
                        evaluation, 
                        if_codebook_updated, 
                        final_output, 
                        custom_logger, 
                        logger_kwargs)
                elif evaluation == "update-no-check":
                    self.num_train_samples += 1
                    print("An update to the codebook occurs... check if the codebook size changed")                        
                    in_accumulation_mode = len(self.codebook.keys()) != current_codebook_size
                    if in_accumulation_mode:
                        print("Codebook expands! Switch to accumulation.")
                
                # Check if codebook was updated
                if if_codebook_updated > 0:
                    pass  # Codebook was updated
                elif evaluation == "correct":
                    self.num_update_correct += 1
                
                total_samples_processed += 1
                
                # Check if codebook size changed
                new_codebook_size = len(self.codebook.keys()) if self.codebook else 0
                
                if new_codebook_size != current_codebook_size:
                    print(f"Codebook updated! Size: {current_codebook_size} → {new_codebook_size}")
                    print("Returning to ACCUMULATION MODE")
                    in_accumulation_mode = True
                
                # Check stop criteria after each training sample in update mode
                if (not in_accumulation_mode) and self.check_stop_criteria(criteria=["global_patience", "max_train_samples",]):
                    self.in_training = False
                    print("STOP CRITERIA REACHED")
                    if in_accumulation_mode:
                        print("There was an update to the codebook")
                        print("Do one more round of accumulation before quiting")
                    else:
                        break
            
            # Progress reporting
            if verbosity > 0 and total_samples_processed % verbosity == 0:
                accuracy = (self.num_train_correct / self.num_train_samples) * 100 if self.num_train_samples > 0 else 0
                update_accuracy = (self.num_update_correct / self.num_update_samples) * 100 if self.num_update_samples > 0 else 0
                print(f"Training Progress: {total_samples_processed} samples processed", flush=True)
                print(f"Current Epoch: {current_epoch}", flush=True)
                print(f"Overall Accuracy: {accuracy:.2f}% ({self.num_train_correct}/{self.num_train_samples})", flush=True)
                print(f"Update Accuracy: {update_accuracy:.2f}% ({self.num_update_correct}/{self.num_update_samples})", flush=True)
                print(f"Mode: {'Accumulation' if in_accumulation_mode else 'Update'}", flush=True)
                print(f"Available samples for accumulation: {len(available_indices)}", flush=True)
                print(f"Accumulation rounds completed: {accumulation_count}", flush=True)
                print("-" * 50, flush=True)
                if ckpt_path is not None:
                    self.save(ckpt_path)
        
        accuracy = (self.num_train_correct / self.num_train_samples) * 100 if self.num_train_samples > 0 else 0
        update_accuracy = (self.num_update_correct / self.num_update_samples) * 100 if self.num_update_samples > 0 else 0
        print(f"Stochastic Sampling Training Complete!")
        print(f"Total epochs completed: {current_epoch}")
        print(f"Total samples processed: {total_samples_processed}")
        print(f"Overall Training Accuracy: {accuracy:.2f}% ({self.num_train_correct}/{self.num_train_samples})", flush=True)
        print(f"Overall Update Accuracy: {update_accuracy:.2f}% ({self.num_update_correct}/{self.num_update_samples})", flush=True)
        if "train" not in self.training_logs:
            self.training_logs["train"] = {}
        self.training_logs["train"]["final_accuracy"] = accuracy
        self.print_codebook()
    
    def _process_pending_accumulation_samples(self, pending_samples, custom_logger, logger_kwargs):
        """
        Process pending accumulation samples: update code occurrences, then evaluate.
        Follows the same 3-step logic as coder_bor.py but uses base class methods.
        
        Args:
            pending_samples (list): List of pending sample data with classification results
            custom_logger: Custom logging function
            logger_kwargs: Logger keyword arguments
        """
        if not pending_samples:
            return
        
        print(f"Processing {len(pending_samples)} pending samples...")
        
        # Step 1: Update all code occurrences from pending samples
        print("Step 1: Updating code occurrences...")
        for pending_sample in pending_samples:
            sample = pending_sample['sample']
            final_output = pending_sample['final_output']
            
            # Parse codes from the classification output
            code_occurrence, decision_code = self._parse_classification_codes(final_output, sample["labels"])
            
            # Update code occurrences in the training data
            self._update_code_occurence(self.code_occurrence_overall["train"], code_occurrence, is_decision_code=False, sample_id=sample["id"])
            self._update_code_occurence(self.decision_code_occurence_overall["train"], decision_code, is_decision_code=True)
        
        # Step 2: Perform imputation if available (base class doesn't have this, but we can check)
        if self._extend_with_nan and self._impute_after_accumulation:
                print("Step 2: Performing imputation...")
                self._impute_missing_values(self.code_occurrence_overall["train"])
        
        # Clear the temporary storage
        self._most_recent_code_occurrence_overall_train = None
        self._most_recent_decision_code_overall_train = None
        
        # Step 3: Now evaluate each sample using the updated code occurrences
        print("Step 3: Evaluating samples...")
        for pending_sample in pending_samples:
            sample = pending_sample['sample']
            classification_prompt = pending_sample['classification_prompt']
            classification = pending_sample['classification']
            final_output = pending_sample['final_output']
            if_codebook_updated = pending_sample['if_codebook_updated']
            
            # Evaluate the classification using the base class method
            evaluation, pred = self._evaluate_classification(final_output, sample["labels"])
            
            # Update statistics with the evaluation result (skip code occurrence updates)
            self._update_sample_statistics(
                sample, 
                classification_prompt, 
                classification, 
                evaluation, 
                if_codebook_updated, 
                final_output, 
                custom_logger, 
                logger_kwargs
            )
        
        print(f"Completed processing {len(pending_samples)} pending samples")
    
    def _update_sample_statistics(self, sample, classification_prompt, classification, evaluation, 
                                        if_codebook_updated, final_output, custom_logger, logger_kwargs):
        """
        Update training statistics for a single sample with direct evaluation result.
        This version skips the code occurrence updates since they were already done.
        
        Args:
            sample: The sample data
            classification_prompt: The classification prompt
            classification: The classification output
            evaluation: The evaluation result (already computed)
            if_codebook_updated: Whether codebook was updated
            final_output: The final output
            custom_logger: Custom logging function
            logger_kwargs: Logger keyword arguments
        """
        # Parse codes for logging purposes (already updated in code occurrences)
        code_occurrence, decision_code = self._parse_classification_codes(final_output, sample["labels"])
        
        # Update training logs
        if "train" not in self.training_logs:
            self.training_logs["train"] = {}
            
        self.training_logs["train"][sample['id']] = {
            "input_prompt": classification_prompt,
            "output_classification": classification,
            "evaluation": evaluation,
            "updated": if_codebook_updated,
            "code_occurrence": code_occurrence,
            "decision_code": decision_code,
            "data": sample,
            "cb_ckpt": deepcopy(self.codebook),
        }
        
        # Note: Code occurrences were already updated in _process_pending_accumulation_samples
        # So we don't need to update them again here
        
        # Update training statistics
        if if_codebook_updated > 0:
            self.train_wrong_ids.append(sample["id"])
        elif evaluation == "correct":
            self.train_correct_ids.append(sample["id"])
            self.num_train_correct += 1
        else:
            self.train_wrong_ids.append(sample["id"])
        
        self.num_train_samples += 1
        
        if custom_logger:
            custom_logger(**logger_kwargs)
        
        self.training_logs["train"][sample['id']]["current_train_acc"] = (self.num_train_correct / self.num_train_samples) * 100

    # def _update_sample_statistics(self, sample, classification_prompt, classification, evaluation, 
    #                             if_codebook_updated, final_output, custom_logger, logger_kwargs):
    #     """
    #     Update training statistics for a single sample.
    #     Uses the codebook-based approach from the base Coder class.
        
    #     Args:
    #         sample: The sample data
    #         classification_prompt: The classification prompt
    #         classification: The classification output
    #         evaluation: The evaluation result
    #         if_codebook_updated: Whether codebook was updated
    #         final_output: The final output
    #         custom_logger: Custom logging function
    #         logger_kwargs: Logger keyword arguments
    #     """
    #     # Parse codes using the base class method (codebook-based)
    #     code_occurrence, decision_code = self._parse_classification_codes(final_output, sample["labels"])
        
    #     # Update training logs
    #     self.training_logs["train"][sample['id']] = {
    #         "input_prompt": classification_prompt,
    #         "output_classification": classification,
    #         "evaluation": evaluation,
    #         "updated": if_codebook_updated,
    #         "code_occurrence": code_occurrence,
    #         "decision_code": decision_code,
    #         "data": sample,
    #         "cb_ckpt": deepcopy(self.codebook),
    #     }
        
    #     # Update code occurrence tracking using base class method
    #     self.code_occurrence_overall["train"] = deepcopy(self._most_recent_code_occurrence_overall_train)
    #     self._most_recent_code_occurrence_overall_train = None
    #     self.decision_code_occurence_overall["train"] = deepcopy(self._most_recent_decision_code_overall_train)
    #     self._most_recent_decision_code_overall_train = None
        
    #     # Update training statistics
    #     if if_codebook_updated > 0:
    #         self.train_wrong_ids.append(sample["id"])
    #     elif evaluation == "correct":
    #         self.train_correct_ids.append(sample["id"])
    #         self.num_train_correct += 1
    #     else:
    #         self.train_wrong_ids.append(sample["id"])
        
    #     self.num_train_samples += 1
        
    #     if custom_logger:
    #         custom_logger(**logger_kwargs)
        
    #     self.training_logs["train"][sample['id']]["current_train_acc"] = (self.num_train_correct / self.num_train_samples) * 100 