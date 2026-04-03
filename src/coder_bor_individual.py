from .coder_bor import CoderBOR
import re
from copy import deepcopy
from src.prompts.bag_of_reasoning_annotation_inst import CODE_INST, CORRECTION_INST
from vllm import LLM, SamplingParams


class CoderBORIndividual(CoderBOR):
    """
    A variant of CoderBOR that annotates each output individually in batch,
    then combines the results for evaluation.
    
    This approach processes each OUTPUT separately using the same system message,
    which may lead to more consistent annotations per output.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configuration for splitting long outputs based on word count
        self.max_output_words = 250  # Maximum words per part
        self.search_window_words = 50  # Word window to search back for good break points
        self.sampling_parameters = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192)
        self.code_inst = CODE_INST
        self.correction_inst = CORRECTION_INST

    def classify(self, outputs, labels, question, if_codebook_updated=0, train=True, qid=None, batched=False, additional_outputs=None, additional_labels=None, additional_questions=None, no_eval=False):
        """
        Override classify method to annotate each output individually then combine results.
        Now supports splitting long outputs into parts for more efficient processing.
        """
        assert not (batched and train), "you could not use batched generation in training"
        
        if batched:
            # Process each output part individually in batch
            part_prompts = []
            part_mappings = []  # Track which prompt corresponds to which sample, output, and part
            
            # Create prompts for each output part
            for sample_idx, (sample_outputs, single_question) in enumerate(zip(outputs, question)):
                for output_idx, single_output in enumerate(sample_outputs):
                    # Split ONLY the reasoning output into parts (instructions/system message stay the same)
                    output_parts = self._split_output_into_parts(single_output)
                    
                    for part_idx, part_text in enumerate(output_parts):
                        prompt = self._classify_prompt_part(part_text, single_question, output_idx, part_idx, len(output_parts))
                        part_prompts.append(prompt)
                        part_mappings.append((sample_idx, output_idx, part_idx))
            
            # Generate annotations for all output parts in batch
            part_annotations = self._generate_batch(part_prompts, max_tokens=int(self.max_output_words * 4), temp_adjust=0)
            
            # Process thinking mode for part annotations
            processed_part_annotations = []
            for annotation in part_annotations:
                if self.think_mode:
                    eot_token = "</think>"
                    eot_index = annotation.rfind(eot_token)
                    if eot_index >= 0:
                        annotation = annotation[eot_index + len(eot_token):]
                processed_part_annotations.append(annotation)
            
            # Combine individual annotations back into paired format
            classification_prompts = []
            classifications = []
            evaluations = []
            final_outputs = []
            
            for sample_idx, (sample_outputs, sample_labels) in enumerate(zip(outputs, labels)):
                # Get annotations for this sample's outputs
                sample_annotations = []
                for output_idx in range(len(sample_outputs)):
                    # Find all part annotations for this output
                    part_annotations_for_output = []
                    for i, (s_idx, o_idx, p_idx) in enumerate(part_mappings):
                        if s_idx == sample_idx and o_idx == output_idx:
                            part_annotations_for_output.append((p_idx, processed_part_annotations[i]))
                    
                    # Sort by part index and combine parts into full output annotation
                    part_annotations_for_output.sort(key=lambda x: x[0])
                    part_texts = [part_ann for _, part_ann in part_annotations_for_output]
                    
                    if part_texts:
                        # Combine part annotations into a single output annotation
                        combined_output_annotation = self._combine_part_annotations(part_texts)
                        sample_annotations.append(combined_output_annotation)
                    else:
                        sample_annotations.append("")
                
                # Combine individual output annotations into paired format
                combined_annotation = self._combine_individual_annotations(sample_annotations, sample_labels)
                
                # Create a dummy classification prompt (for logging purposes)
                classification_prompt = self._classify_prompt(sample_outputs, question[sample_idx])
                
                classification_prompts.append(classification_prompt)
                classifications.append(combined_annotation)  # Use combined as the "full classification"
                
                if no_eval:
                    evaluation, pred = "do that later", "do that later"
                else:
                    evaluation, pred = self._evaluate_classification(combined_annotation, sample_labels)
                
                evaluations.append(evaluation)
                final_outputs.append(combined_annotation)
            return classification_prompts, classifications, evaluations, [if_codebook_updated] * len(labels), final_outputs
        
        else:
            # For non-batched processing, still use batch generation when there are multiple prompts
            all_part_prompts = []
            part_mappings = []  # Track which part belongs to which output
            
            # Collect all parts from all outputs
            for output_idx, single_output in enumerate(outputs):
                # Split ONLY the reasoning output into parts (instructions/system message stay the same)
                output_parts = self._split_output_into_parts(single_output)
                
                # Create prompts for each part
                for part_idx, part_text in enumerate(output_parts):
                    part_prompt = self._classify_prompt_part(part_text, question, output_idx, part_idx, len(output_parts))
                    all_part_prompts.append(part_prompt)
                    part_mappings.append((output_idx, part_idx))
            
            # Always use batch generation when there are multiple prompts (from multiple outputs or splitting)
            if len(all_part_prompts) > 1:
                all_part_annotations = self._generate_batch(all_part_prompts, max_tokens=int(self.max_output_words * 4), temp_adjust=0)
            elif len(all_part_prompts) == 1:
                # Only use individual generate for single prompt
                part_annotation = self._generate(all_part_prompts[0])
                all_part_annotations = [part_annotation]
            else:
                all_part_annotations = []
            
            # Process thinking mode for all part annotations
            processed_part_annotations = []
            for annotation in all_part_annotations:
                if self.think_mode:
                    eot_token = "</think>"
                    eot_index = annotation.rfind(eot_token)
                    if eot_index >= 0:
                        annotation = annotation[eot_index + len(eot_token):]
                processed_part_annotations.append(annotation)
            
            # Reconstruct annotations by output
            individual_annotations = []
            for output_idx in range(len(outputs)):
                # Find all part annotations for this output
                part_annotations_for_output = []
                for i, (o_idx, p_idx) in enumerate(part_mappings):
                    if o_idx == output_idx:
                        part_annotations_for_output.append((p_idx, processed_part_annotations[i]))
                
                # Sort by part index and combine
                part_annotations_for_output.sort(key=lambda x: x[0])
                part_texts = [part_ann for _, part_ann in part_annotations_for_output]
                
                if part_texts:
                    # Combine part annotations into full output annotation
                    combined_output_annotation = self._combine_part_annotations(part_texts)
                    individual_annotations.append(combined_output_annotation)
                else:
                    individual_annotations.append("")
            
            # Combine individual annotations
            combined_annotation = self._combine_individual_annotations(individual_annotations, labels)
            
            # Create classification prompt for logging
            classification_prompt = self._classify_prompt(outputs, question)
            
            # Truncate annotation if it's too long
            self.current_annotation = self._truncate_annotation_if_needed(combined_annotation)
            
            # Update the most recent code occurrence and decision code so classification can use the new code for evaluation
            code_occurrence, decision_code = self._parse_classification_codes(combined_annotation, labels)
            self._most_recent_code_occurrence_overall_train = deepcopy(self.code_occurrence_overall["train"])
            self._most_recent_decision_code_overall_train = deepcopy(self.decision_code_occurence_overall["train"])
            self._update_code_occurence(self._most_recent_code_occurrence_overall_train, code_occurrence, is_decision_code=False, sample_id=qid)
            self._update_code_occurence(self._most_recent_decision_code_overall_train, decision_code, is_decision_code=True)

            evaluation, pred = self._evaluate_classification(combined_annotation, labels)

            if evaluation == "correct":
                self._update_dataset(classification_prompt, combined_annotation)
                if if_codebook_updated > 0 and train:
                    self.num_consecutive_turns_without_update = 0
                else:
                    self.num_consecutive_turns_without_update += 1
            elif train:
                if if_codebook_updated >= self.patience:
                    print(f"------\nOut of patience after {self.patience} update trails", flush=True)
                    if qid:
                        print(f"at sample {qid}", flush=True)
                    print("------", flush=True)
                    self.codebook = deepcopy(self.backtrack_codebook)
                    if hasattr(self, 'bor_training') and self.bor_training:
                        self._most_recent_code_occurrence_overall_train = deepcopy(self.code_occurrence_overall["train"])
                        self._most_recent_decision_code_overall_train = deepcopy(self.decision_code_occurence_overall["train"])
                    self.num_consecutive_turns_without_update += 1
                else:
                    # Update the codebook and redo the classification
                    self._most_recent_code_occurrence_overall_train = deepcopy(self.code_occurrence_overall["train"])
                    self._most_recent_decision_code_overall_train = deepcopy(self.decision_code_occurence_overall["train"])
                    
                    if if_codebook_updated > 0:
                        print("Backtracking", flush=True)
                        self.failed_codebook = deepcopy(self.codebook)
                        self.codebook = deepcopy(self.backtrack_codebook)
                    else:
                        self.backtrack_codebook = deepcopy(self.codebook)
                    
                    if self.batch_update and additional_outputs is not None:
                        return self._update_codebook_batch(outputs, labels, question, additional_outputs, additional_labels, additional_questions, qid=qid, if_codebook_updated=if_codebook_updated)
                    else:
                        return self._update_codebook(outputs, labels, question, qid=qid, if_codebook_updated=if_codebook_updated)

            if train:
                self.backtrack_codebook = None
                self.failed_codebook = None
                self.failed_update = None

            return classification_prompt, combined_annotation, evaluation, if_codebook_updated, combined_annotation
    
    def _split_output_into_parts(self, output):
        """
        Split a long reasoning output into smaller parts based on word count.
        Prioritizes newlines for natural breaks, then sentence boundaries.
        
        This only splits the reasoning output text - NOT the system message, 
        instructions, or any other parts of the prompt.
        
        Args:
            output (str): The reasoning output text to split (e.g., model's reasoning response)
            
        Returns:
            list: List of output parts (strings) - each containing a portion of the original reasoning
        """
        # Split into words while preserving original text structure
        words = output.split()
        if len(words) <= self.max_output_words:
            return [output]
        
        parts = []
        start_word_idx = 0
        
        while start_word_idx < len(words):
            # Determine the end word index for this part
            end_word_idx = min(start_word_idx + self.max_output_words, len(words))
            
            # If this is not the last part, try to find a better break point
            if end_word_idx < len(words):
                best_break_idx = end_word_idx
                search_start_idx = max(start_word_idx, end_word_idx - self.search_window_words)
                
                # Reconstruct text to find break points
                current_text = " ".join(words[start_word_idx:end_word_idx])
                search_text = " ".join(words[search_start_idx:end_word_idx])
                
                # Find the position in the original text where our search window starts
                if search_start_idx > start_word_idx:
                    prefix_text = " ".join(words[start_word_idx:search_start_idx])
                    search_offset = len(prefix_text) + 1  # +1 for the space
                else:
                    search_offset = 0
                
                # Look for newlines first (highest priority)
                newline_positions = []
                for i, char in enumerate(search_text):
                    if char == '\n':
                        # Calculate word position of this newline
                        text_before_newline = current_text[:search_offset + i]
                        words_before_newline = len(text_before_newline.split())
                        if words_before_newline > 0:  # Ensure we have at least some content
                            newline_positions.append(start_word_idx + words_before_newline)
                
                if newline_positions:
                    # Use the last newline within our search window
                    best_break_idx = newline_positions[-1]
                else:
                    # Look for sentence endings
                    sentence_positions = []
                    for i, char in enumerate(search_text):
                        if char in '.!?' and i + 1 < len(search_text) and search_text[i + 1].isspace():
                            # Calculate word position of this sentence end
                            text_before_sentence = current_text[:search_offset + i + 1]
                            words_before_sentence = len(text_before_sentence.split())
                            if words_before_sentence > 0:
                                sentence_positions.append(start_word_idx + words_before_sentence)
                    
                    if sentence_positions:
                        # Use the last sentence end within our search window
                        best_break_idx = sentence_positions[-1]
                
                end_word_idx = best_break_idx
            
            # Create the part text
            part_words = words[start_word_idx:end_word_idx]
            if part_words:  # Only add non-empty parts
                part_text = " ".join(part_words)
                parts.append(part_text)
            
            start_word_idx = end_word_idx
            
            # Avoid infinite loop
            if start_word_idx >= len(words):
                break
        
        return parts if parts else [output]
    
    def _classify_prompt_part(self, part_text, question, output_idx, part_idx, total_parts):
        """
        Create a classification prompt for a part of an output.
        
        Only the reasoning output (part_text) is split. The system message 
        and instructions remain exactly the same across all parts.
        
        Args:
            part_text (str): The SPLIT part of the reasoning output to annotate
            question (str): The question/task description
            output_idx (int): Index of the output (0 for OUTPUT A, 1 for OUTPUT B)
            part_idx (int): Index of this part within the output
            total_parts (int): Total number of parts for this output
            
        Returns:
            list: Messages for the LLM
        """
        output_label = "OUTPUT A" if output_idx == 0 else "OUTPUT B"
        
        if total_parts == 1:
            part_info = ""
        else:
            part_info = f" (Part {part_idx + 1} of {total_parts})"
        
        # SAME instruction for all parts - only the reasoning output changes
        coding_prompt = f"You will be given a part of a reasoning output and your task is to annotate the occurrence of reasoning behaviors in the given part based on the definitions and examples of reasoning behaviors in a taxonomy.\n\nThe system message provides a reasoning taxonomy that illustrates the reasoning behaviors. Follow the instruction and reasoning taxonomy in the system message and this message closely when making annotations. Annotate the reasoning OUTPUT part sentence by sentence."

        if hasattr(self, "no_update_to_codebook") and self.no_update_to_codebook:
            coding_prompt += " It's possible that a sentence does contain any reasoning behaviors listed in the given taxonomy. If so, use [Not in Taxonomy] to annotate that sentence."

        # SAME system message for all parts - taxonomy and instructions unchanged
        system_prompt = f"""{self.code_inst}\n"""
        
        # Handle case where codebook might be None
        if self.codebook:
            for code in self.codebook.keys():
                system_prompt += f"{code}: {self.codebook[code]}\n\n"

        if self.think_budget > 0:
            system_prompt += f"\nThink efficiently. Limit your thinking to {self.think_budget} words."

        # ONLY the reasoning output (part_text) changes - everything else is identical
        final_prompt = f"""{coding_prompt}\n\nGiven the ### {output_label}{part_info}:\n{part_text}\n-----End of {output_label}{part_info}-----\n\nAnnotate this part following the provided reasoning taxonomy in the system instruction. Your output should just contain an annotated reasoning part."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ]
    
    def _combine_part_annotations(self, part_annotations):
        """
        Combine annotations from multiple parts of the same output.
        
        Args:
            part_annotations (list): List of annotation strings for parts of the same output
            
        Returns:
            str: Combined annotation for the full output
        """
        if not part_annotations:
            return ""
        
        if len(part_annotations) == 1:
            return part_annotations[0]
        
        # Parse each part annotation to get counts
        part_counts = []
        for part_annotation in part_annotations:
            part_count = self._parse_individual_annotation(part_annotation)
            part_counts.append(part_count)
        
        # Combine counts across parts
        combined_counts = {}
        for part_count in part_counts:
            for behavior_name, count in part_count.items():
                if behavior_name not in combined_counts:
                    combined_counts[behavior_name] = 0
                combined_counts[behavior_name] += count
        
        # Create a combined annotation text
        combined_parts = []
        # combined_parts.append("Combined annotation from multiple parts:")
        # combined_parts.append("")
        
        # for behavior_name, total_count in combined_counts.items():
        #     if total_count > 0:
        #         combined_parts.append(f"{behavior_name}: {total_count} occurrences")
        
        combined_parts.append("")
        combined_parts.append("\n--- Original part annotations ---")
        for i, part_annotation in enumerate(part_annotations):
            combined_parts.append(f"Part {i + 1}:")
            combined_parts.append(part_annotation)
            combined_parts.append("")
        
        return "\n".join(combined_parts)
    
    def configure_splitting(self, max_output_words=400, search_window_words=50):
        """
        Configure the parameters for splitting long outputs based on word count.
        
        Args:
            max_output_words (int): Maximum words per part
            search_window_words (int): Word window to search back for good break points (newlines, sentences)
        """
        self.max_output_words = max_output_words
        self.search_window_words = search_window_words
        print(f"Output splitting configured: max_words={max_output_words}, search_window={search_window_words}")
    
    def _combine_individual_annotations(self, individual_annotations, labels):
        """
        Literally concatenate the partial outputs and add count summary.
        Preserves all annotation content plus adds structured counts at the end.
        
        Args:
            individual_annotations (list): List of annotations for each output
            labels (list): Labels for each output
            
        Returns:
            str: Literally combined annotation content plus count summary
        """
        if len(individual_annotations) < 2:
            # Pad with empty annotation if needed
            while len(individual_annotations) < 2:
                individual_annotations.append("")
        
        # Step 1: Concatenate annotation content with output labels
        combined_content_parts = []
        
        # Add annotation content with clear output labels
        for i, annotation in enumerate(individual_annotations):
            if annotation.strip():  # Only add non-empty annotations
                output_label = "OUTPUT A" if i == 0 else "OUTPUT B"
                combined_content_parts.append(f"=== Annotation for {output_label} ===")
                combined_content_parts.append(annotation.strip())
                combined_content_parts.append("")  # Add spacing between outputs
        
        # Join all content with newlines
        literal_combined_content = "\n".join(combined_content_parts)
        
        # Step 2: Parse individual annotations to extract code counts 
        parsed_annotations = []
        for annotation in individual_annotations:
            parsed_annotations.append(self._parse_individual_annotation(annotation))
        
        # Get ONLY the codes that actually appear in the partial outputs
        all_codes = []
        for parsed in parsed_annotations:
            for code in parsed.keys():
                if code not in all_codes:
                    all_codes.append(code)
        
        # Step 3: Build the final output: literal content + count summary
        final_parts = []
        
        # Add the literal combined content first
        if literal_combined_content:
            final_parts.append(literal_combined_content)
            final_parts.append("")  # Separator
        
        # Add count summary for each behavior found
        for code in all_codes:
            final_parts.append(f"### {code}")
            final_parts.append("")
            
            # Get counts for each output
            count_a = parsed_annotations[0].get(code, 0) if len(parsed_annotations) > 0 else 0
            count_b = parsed_annotations[1].get(code, 0) if len(parsed_annotations) > 1 else 0
            
            # Add occurrence statements before counts
            if count_a > 0:
                final_parts.append("This reasoning behavior is observed in OUTPUT A.")
            else:
                final_parts.append("This reasoning behavior is not observed in OUTPUT A.")
            
            final_parts.append(f"### Count in OUTPUT A: {count_a}")
            final_parts.append("")
            
            if count_b > 0:
                final_parts.append("This reasoning behavior is observed in OUTPUT B.")
            else:
                final_parts.append("This reasoning behavior is not observed in OUTPUT B.")
            
            final_parts.append(f"### Count in OUTPUT B: {count_b}")
            final_parts.append("")
        
        return "\n".join(final_parts)
    
    def _parse_individual_annotation(self, annotation):
        """
        Parse an individual annotation to extract code counts by counting occurrences of behavior names.
        
        Args:
            annotation (str): The annotation text for a single output
            
        Returns:
            dict: Mapping of code names to counts
        """
        code_counts = {}
        
        # Get all reasoning behavior names from the codebook
        if not self.codebook:
            return code_counts
            
        behavior_names = list(self.codebook.keys())
        
        # Count occurrences of each behavior name in the annotation
        annotation_lower = annotation.lower()
        
        for behavior_name in behavior_names:
            # Count direct mentions of the behavior name (case-insensitive)
            behavior_lower = behavior_name.lower()
            
            # Count occurrences, but be careful about partial matches
            # We'll look for the behavior name as a whole word or phrase
            count = 0
            
            # Split the behavior name into words for more flexible matching
            behavior_words = behavior_lower.split()
            
            if len(behavior_words) == 1:
                # Single word behavior - count whole word occurrences
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(behavior_lower) + r'\b'
                matches = re.findall(pattern, annotation_lower)
                count = len(matches)
            else:
                # Multi-word behavior - count phrase occurrences
                # Look for the full phrase
                count = annotation_lower.count(behavior_lower)
            
            code_counts[behavior_name] = count
        
        return code_counts 

    def _truncate_annotation_if_needed(self, annotation):
        """
        Always remove count summary from annotation.
        If annotation exceeds 20,000 words, also truncate content under each output label to 10,000 words max.
        
        Args:
            annotation (str): The annotation to process
            
        Returns:
            str: Processed annotation without count summary, and truncated if needed
        """
        lines = annotation.split('\n')
        
        # Step 1: Remove count summaries (everything from first "### " line onward)
        for i, line in enumerate(lines):
            if line.startswith("### Count in "):
                lines = lines[:i]  # Keep only lines before count summaries
                break
        
        # Step 2: If too long, truncate each output section to 10,000 words
        total_words = len(' '.join(lines).split())
        if total_words > 20000:
            result_lines = []
            current_section = []
            current_header = None
            
            for line in lines:
                if line.startswith("=== Annotation for OUTPUT"):
                    # Process previous section if exists
                    if current_header and current_section:
                        section_text = '\n'.join(current_section)
                        words = section_text.split()
                        if len(words) > 10000:
                            section_text = ' '.join(words[:10000])
                        
                        result_lines.append(current_header)
                        result_lines.append(section_text)
                        result_lines.append("")  # spacing
                    
                    # Start new section
                    current_header = line
                    current_section = []
                else:
                    if current_header:
                        current_section.append(line)
                    else:
                        result_lines.append(line)  # Content before output sections
            
            # Handle last section
            if current_header and current_section:
                section_text = '\n'.join(current_section)
                words = section_text.split()
                if len(words) > 10000:
                    section_text = ' '.join(words[:10000])
                
                result_lines.append(current_header)
                result_lines.append(section_text)
            
            return '\n'.join(result_lines)
        
        # No truncation needed, just return without count summaries
        return '\n'.join(lines) 