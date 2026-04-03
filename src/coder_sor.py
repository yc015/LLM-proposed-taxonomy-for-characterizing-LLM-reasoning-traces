import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Any
from .coder_bor_individual import CoderBORIndividual
from .classifiers.lstm_classifier import classify_with_lstm


class CoderSor(CoderBORIndividual):
    """
    A Sequential-Order-Reasoning (SOR) variant of CoderBORIndividual that preserves
    the sequential ordering of reasoning behaviors in annotated traces.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Configuration for sequence representation
        self.max_sequence_length = 50
        self.behavior_to_index = {}
        self.index_to_behavior = {}
    
    def parse_reasoning_trace_to_vector(self, annotation_text: str, code_order: Optional[List[str]] = None) -> np.ndarray:
        """
        Parse an annotated reasoning trace and convert it to a vector representing sequence of behaviors.
        
        Args:
            annotation_text (str): The annotated reasoning trace
            code_order (List[str], optional): Ordered list of behavior codes. If None, uses self.codebook
            
        Returns:
            np.ndarray: Vector where dimensions represent position/temporal order and values represent 
                       behavior indices according to codebook or code_order
        """
        if code_order is not None:
            # Use provided code order
            behavior_to_index = {behavior: idx for idx, behavior in enumerate(code_order)}
        elif self.codebook:
            # Use codebook order
            behavior_to_index = {behavior: idx for idx, behavior in enumerate(self.codebook.keys())}
        elif hasattr(self, 'behavior_to_index') and self.behavior_to_index:
            # Use stored behavior mapping
            behavior_to_index = self.behavior_to_index
        else:
            raise ValueError("No code order or codebook available for parsing")
        
        # Parse sequences from annotation
        sequences_a, sequences_b = self._parse_sequences_from_annotation(annotation_text, behavior_to_index)
        
        # Convert to vectors with fixed length
        vector_a = self._sequence_to_vector(sequences_a)
        vector_b = self._sequence_to_vector(sequences_b)
        
        return vector_a, vector_b
    
    def _sequence_to_vector(self, sequence_indices: List[int]) -> np.ndarray:
        """
        Convert a sequence of behavior indices to a fixed-length vector.
        
        Args:
            sequence_indices (List[int]): List of behavior indices in sequence order
            
        Returns:
            np.ndarray: Fixed-length vector representing the sequence
        """
        vector = np.full(self.max_sequence_length, -1, dtype=int)  # -1 for padding
        
        # Fill vector with sequence values up to max length
        for i, idx in enumerate(sequence_indices[:self.max_sequence_length]):
            vector[i] = idx
        
        return vector
    
    def _update_reasoning_sequence_vectors(self, results_dict: Dict, code_occurrence: Dict, sample_id: Optional[str] = None):
        """Update reasoning vectors with temporal sequence data."""
        # Initialize structure (core BOR components + behavior mappings for tokenization)
        if 'vectors' not in results_dict:
            results_dict['vectors'] = {}
            results_dict['code_order'] = []
            results_dict['sample_ids'] = []
            results_dict['labels'] = []
            results_dict['behavior_to_index'] = {}
            results_dict['index_to_behavior'] = {}
        
        # Get current codebook order
        current_codes = list(self.codebook.keys()) if self.codebook else []
        
        # If codebook has grown, reindex existing vectors
        if len(current_codes) > len(results_dict['code_order']):
            self._reindex_existing_vectors(results_dict, current_codes)
        
        # Update mappings
        results_dict['code_order'] = current_codes[:]
        results_dict['behavior_to_index'] = {behavior: idx for idx, behavior in enumerate(current_codes)}
        results_dict['index_to_behavior'] = {idx: behavior for idx, behavior in enumerate(current_codes)}
        self.behavior_to_index = results_dict['behavior_to_index']
        self.index_to_behavior = results_dict['index_to_behavior']
        
        # Extract model names
        model_names = []
        if code_occurrence:
            first_code_data = next(iter(code_occurrence.values()))
            if isinstance(first_code_data, dict):
                model_names = list(first_code_data.keys())
        
        model_a_name = model_names[0] if len(model_names) >= 1 else "OUTPUT A"
        model_b_name = model_names[1] if len(model_names) >= 2 else "OUTPUT B"
        
        # Parse sequences and convert to vectors
        sequences_a, sequences_b = self._parse_sequences_from_annotation(
            self.current_annotation if hasattr(self, 'current_annotation') else "",
            self.behavior_to_index
        )
        
        vector_a = self._sequence_to_vector(sequences_a)
        vector_b = self._sequence_to_vector(sequences_b)
        
        # Store vectors
        if sample_id:
            model_a_key = f"{sample_id}_{model_a_name}"
            model_b_key = f"{sample_id}_{model_b_name}"
            if model_a_key not in results_dict['vectors']:
                results_dict['sample_ids'].extend([model_a_key, model_b_key])
                results_dict['labels'].extend([model_a_name, model_b_name])
            results_dict['vectors'][model_a_key] = vector_a
            results_dict['vectors'][model_b_key] = vector_b
        else:
            print("Sample ID not provided: can potentially cause errors")
    
    def _parse_sequences_from_annotation(self, annotation_text: str, behavior_to_index: Dict[str, int]) -> Tuple[List[int], List[int]]:
        """
        Parse the annotated reasoning trace to extract sequential behavior patterns.
        
        Args:
            annotation_text (str): The full annotation text
            behavior_to_index (dict): Mapping from behavior names to indices
            
        Returns:
            Tuple[List[int], List[int]]: Sequences for OUTPUT A and OUTPUT B as behavior indices
        """
        sequences_a = []
        sequences_b = []
        
        if not annotation_text or not behavior_to_index:
            return sequences_a, sequences_b
        
        # Split annotation into sections for OUTPUT A and OUTPUT B
        sections = self._split_annotation_by_output(annotation_text)
        
        # Extract sequences from each section
        if "OUTPUT A" in sections:
            sequences_a = self._extract_sequence_from_text(sections["OUTPUT A"], behavior_to_index)
        
        if "OUTPUT B" in sections:
            sequences_b = self._extract_sequence_from_text(sections["OUTPUT B"], behavior_to_index)
        
        return sequences_a, sequences_b
    
    def _split_annotation_by_output(self, annotation_text: str) -> Dict[str, str]:
        """Split the annotation text into sections for different outputs."""
        sections = {}
        
        # Look for patterns like "Part 1:" or "OUTPUT A:" etc.
        current_section = None
        current_content = []
        
        lines = annotation_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if this line indicates a new section
            if 'Part' in line and ':' in line:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = line
                current_content = []
            elif line:
                current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        # If no clear sections found, treat the entire text as both A and B
        if not sections:
            sections["OUTPUT A"] = annotation_text
            sections["OUTPUT B"] = annotation_text
        
        return sections
    
    def _extract_sequence_from_text(self, text: str, behavior_to_index: Dict[str, int]) -> List[int]:
        """
        Extract a sequence of behavior indices from annotation text.
        
        Args:
            text (str): The annotation text for one output
            behavior_to_index (dict): Mapping from behavior names to indices
            
        Returns:
            List[int]: Sequence of behavior indices in order of appearance
        """
        sequence = []
        
        if not text or not behavior_to_index:
            return sequence
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Track positions where behaviors appear
        behavior_positions = []
        
        for behavior_name, index in behavior_to_index.items():
            behavior_lower = behavior_name.lower()
            
            # Find all occurrences of this behavior in the text
            start_pos = 0
            while True:
                pos = text_lower.find(behavior_lower, start_pos)
                if pos == -1:
                    break
                
                # Check if this is a valid behavior match (whole word/phrase)
                if self._is_valid_behavior_match(text_lower, pos, behavior_lower):
                    behavior_positions.append((pos, index, behavior_name))
                
                start_pos = pos + 1
        
        # Sort by position to get sequential order
        behavior_positions.sort(key=lambda x: x[0])
        
        # Extract the sequence of behavior indices
        sequence = [index for _, index, _ in behavior_positions]
        
        # Limit sequence length if needed
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
        
        return sequence
    
    def _is_valid_behavior_match(self, text: str, pos: int, behavior: str) -> bool:
        """Check if a behavior match at a given position is valid (whole word/phrase)."""
        # Check word boundaries for single-word behaviors
        if ' ' not in behavior:
            # Check if character before is non-alphanumeric
            if pos > 0 and text[pos - 1].isalnum():
                return False
            # Check if character after is non-alphanumeric
            end_pos = pos + len(behavior)
            if end_pos < len(text) and text[end_pos].isalnum():
                return False
        
        return True
    
    def _extract_behavior_sequences_from_occurrence(self, code_occurrence: Dict) -> List[List[str]]:
        """Extract behavior sequences from the code occurrence structure."""
        sequences = []
        
        # Extract the behavior names that have non-zero counts
        for behavior_name, counts in code_occurrence.items():
            for model_name, count in counts.items():
                if count > 0:
                    # Add the behavior to sequences (simplified approach)
                    sequences.append([behavior_name] * count)
        
        return sequences
    
    def _reindex_existing_vectors(self, results_dict: Dict, new_code_order: List[str]):
        """
        Reindex existing temporal sequence vectors when codebook grows.
        Unlike coder_bor.py which extends vectors, we need to reindex the behavior indices.
        
        Args:
            results_dict (dict): The results dictionary with existing vectors
            new_code_order (List[str]): New code order with additional behaviors
        """
        if not results_dict.get('vectors') or not results_dict.get('code_order'):
            return
        
        old_code_order = results_dict['code_order']
        old_to_new_mapping = {}
        
        # Create mapping from old indices to new indices
        for old_idx, behavior in enumerate(old_code_order):
            if behavior in new_code_order:
                new_idx = new_code_order.index(behavior)
                old_to_new_mapping[old_idx] = new_idx
        
        # Reindex all existing vectors
        for key, old_vector in results_dict['vectors'].items():
            new_vector = old_vector.copy()
            
            # Update each position in the temporal sequence vector
            for pos in range(len(new_vector)):
                old_behavior_idx = new_vector[pos]
                if old_behavior_idx >= 0 and old_behavior_idx in old_to_new_mapping:
                    new_vector[pos] = old_to_new_mapping[old_behavior_idx]
            
            results_dict['vectors'][key] = new_vector
        
        print(f"Reindexed {len(results_dict['vectors'])} vectors from {len(old_code_order)} to {len(new_code_order)} behaviors")
    
    def _update_code_occurence(self, results_dict: Dict, code_occurrence: Dict, sample_id: Optional[str] = None, is_decision_code: Optional[bool] = None):
        """
        Override to handle sequential reasoning vectors instead of count-based vectors.
        
        For reasoning codes (is_decision_code=False), stores sequences of behaviors.
        For decision codes (is_decision_code=True), uses parent class behavior.
        """
        # Auto-detect if not explicitly specified
        if is_decision_code is None:
            is_decision_code = False
            if code_occurrence:
                first_code_data = next(iter(code_occurrence.values()))
                if isinstance(first_code_data, dict):
                    first_value = next(iter(first_code_data.values()))
                    is_decision_code = isinstance(first_value, bool)
        
        if is_decision_code:
            # Handle decision codes (unchanged from parent class)
            super()._update_code_occurence(results_dict, code_occurrence, sample_id, is_decision_code)
        else:
            # Handle reasoning codes with sequences - store sequence vectors instead of count vectors
            self._update_reasoning_sequence_vectors(results_dict, code_occurrence, sample_id)
    
    def classify_with_transition_matrix(self, test_vectors: List[np.ndarray], classification: Optional[str] = None, split: str = "train") -> Any:
        """
        Classify using transition matrix matching between averaged training matrices and test vectors.
        
        Args:
            test_vectors (List[np.ndarray]): Test vectors to classify
            classification (str, optional): If None, returns average transition matrices only
            split (str): Data split to use for training data
            
        Returns:
            If classification is None: Tuple of (avg_transition_matrix_model_a, avg_transition_matrix_model_b)
            Otherwise: List of predictions for test vectors
        """
        if split not in self.code_occurrence_overall:
            raise ValueError(f"No data available for split '{split}'")
        
        data = self.code_occurrence_overall[split]
        if 'vectors' not in data:
            raise ValueError(f"No vectors available for split '{split}'")
        
        # Get training vectors by model
        model_a_vectors = []
        model_b_vectors = []
        
        for key, vector in data['vectors'].items():
            if 'OUTPUT A' in key or key.endswith('_A'):
                model_a_vectors.append(vector)
            elif 'OUTPUT B' in key or key.endswith('_B'):
                model_b_vectors.append(vector)
        
        if not model_a_vectors or not model_b_vectors:
            raise ValueError("Insufficient training data for both models")
        
        # Compute average transition matrices
        avg_transition_a = self._compute_average_transition_matrix(model_a_vectors)
        avg_transition_b = self._compute_average_transition_matrix(model_b_vectors)
        
        if classification is None:
            return avg_transition_a, avg_transition_b
        
        # Classify test vectors
        predictions = []
        for test_vector in test_vectors:
            test_transition = self._vector_to_transition_matrix(test_vector)
            
            # Compute similarity to each model's average transition matrix
            sim_a = self._transition_matrix_similarity(test_transition, avg_transition_a)
            sim_b = self._transition_matrix_similarity(test_transition, avg_transition_b)
            
            # Predict the model with higher similarity
            prediction = "A" if sim_a > sim_b else "B"
            predictions.append(prediction)
        
        return predictions
    
    def _compute_average_transition_matrix(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Compute average transition matrix from a list of sequence vectors."""
        if not vectors:
            return np.zeros((0, 0))
        
        # Get vocabulary size
        vocab_size = len(self.behavior_to_index) if self.behavior_to_index else max([max(v[v >= 0]) for v in vectors if len(v[v >= 0]) > 0]) + 1
        
        transition_matrices = []
        for vector in vectors:
            transition_matrix = self._vector_to_transition_matrix(vector, vocab_size)
            transition_matrices.append(transition_matrix)
        
        # Average all transition matrices
        if transition_matrices:
            return np.mean(transition_matrices, axis=0)
        else:
            return np.zeros((vocab_size, vocab_size))
    
    def _vector_to_transition_matrix(self, vector: np.ndarray, vocab_size: Optional[int] = None) -> np.ndarray:
        """Convert a sequence vector to a transition matrix."""
        if vocab_size is None:
            vocab_size = len(self.behavior_to_index) if self.behavior_to_index else int(max(vector[vector >= 0])) + 1
        
        transition_matrix = np.zeros((vocab_size, vocab_size))
        
        # Extract valid sequence (non-padding values)
        valid_sequence = vector[vector >= 0]
        
        # Count transitions between consecutive behaviors
        for i in range(len(valid_sequence) - 1):
            from_behavior = int(valid_sequence[i])
            to_behavior = int(valid_sequence[i + 1])
            if from_behavior < vocab_size and to_behavior < vocab_size:
                transition_matrix[from_behavior, to_behavior] += 1
        
        # Normalize rows (each row sums to 1)
        row_sums = np.sum(transition_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    def _transition_matrix_similarity(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Compute similarity between two transition matrices using Frobenius norm."""
        if matrix1.shape != matrix2.shape:
            return 0.0
        
        # Use negative Frobenius norm (higher similarity = smaller distance)
        return -np.linalg.norm(matrix1 - matrix2, 'fro')
    
    def initialize_from_bor_coder(self, bor_coder, split_name: str = "train") -> int:
        """
        Initialize this SOR coder with pretrained data from a BOR coder.
        Converts bag-of-reasoning annotations to sequential representations.
        
        Args:
            bor_coder: Trained CoderBOR instance with training_logs
            split_name (str): Split name to transfer (e.g., "train", "eval")
            
        Returns:
            int: Number of processed samples
        """
        # Transfer codebook
        if hasattr(bor_coder, 'codebook') and bor_coder.codebook:
            self.codebook = deepcopy(bor_coder.codebook)
            print(f"Transferred codebook with {len(self.codebook)} behaviors")
        
        # Check if BOR coder has training logs for the split
        if not hasattr(bor_coder, 'training_logs') or split_name not in bor_coder.training_logs:
            print(f"No training logs found for split '{split_name}' in BOR coder")
            return 0
        
        bor_logs = bor_coder.training_logs[split_name]
        
        # Initialize structure matching coder_bor.py format
        if split_name not in self.code_occurrence_overall:
            self.code_occurrence_overall[split_name] = {}
        
        self.code_occurrence_overall[split_name]['vectors'] = {}
        self.code_occurrence_overall[split_name]['code_order'] = []
        self.code_occurrence_overall[split_name]['sample_ids'] = []
        self.code_occurrence_overall[split_name]['labels'] = []
        self.code_occurrence_overall[split_name]['behavior_to_index'] = {}
        self.code_occurrence_overall[split_name]['index_to_behavior'] = {}
        
        # Build behavior vocabulary from codebook
        if self.codebook:
            current_codes = list(self.codebook.keys())
            self.code_occurrence_overall[split_name]['code_order'] = current_codes[:]
            self.code_occurrence_overall[split_name]['behavior_to_index'] = {behavior: idx for idx, behavior in enumerate(current_codes)}
            self.code_occurrence_overall[split_name]['index_to_behavior'] = {idx: behavior for idx, behavior in enumerate(current_codes)}
            self.behavior_to_index = self.code_occurrence_overall[split_name]['behavior_to_index']
            self.index_to_behavior = self.code_occurrence_overall[split_name]['index_to_behavior']
        
        # Process each sample in the BOR training logs
        processed_samples = 0
        for sample_id, log_entry in bor_logs.items():
            annotation_text = log_entry.get("output_classification", "")
            sample_data = log_entry.get("data", {})
            
            if not annotation_text or not sample_data:
                continue
            
            # Extract labels from sample data
            labels = sample_data.get("labels", ["OUTPUT A", "OUTPUT B"])
            
            # Parse sequences from the annotation text
            sequences_a, sequences_b = self._parse_sequences_from_annotation(
                annotation_text, 
                self.behavior_to_index
            )
            
            # Convert to vectors
            vector_a = self._sequence_to_vector(sequences_a)
            vector_b = self._sequence_to_vector(sequences_b)
            
            # Store vectors with proper keys (matching coder_bor.py format)
            model_a_key = f"{sample_id}_{labels[0]}" if len(labels) > 0 else f"{sample_id}_OUTPUT_A"
            model_b_key = f"{sample_id}_{labels[1]}" if len(labels) > 1 else f"{sample_id}_OUTPUT_B"
            
            if model_a_key not in self.code_occurrence_overall[split_name]['vectors'].keys():
                self.code_occurrence_overall[split_name]['sample_ids'].extend([model_a_key, model_b_key])
                self.code_occurrence_overall[split_name]['labels'].extend([labels[0] if len(labels) > 0 else "OUTPUT_A", 
                                                                          labels[1] if len(labels) > 1 else "OUTPUT_B"])
            
            self.code_occurrence_overall[split_name]['vectors'][model_a_key] = vector_a
            self.code_occurrence_overall[split_name]['vectors'][model_b_key] = vector_b
            
            processed_samples += 1
        
        # Transfer other relevant attributes from BOR coder
        if hasattr(bor_coder, 'model_options'):
            self.model_options = bor_coder.model_options
        
        if hasattr(bor_coder, 'code_inst'):
            self.code_inst = bor_coder.code_inst
        
        # Initialize training logs structure
        if split_name not in self.training_logs:
            self.training_logs[split_name] = {}
        
        print(f"Successfully initialized SOR coder from BOR coder:")
        print(f"  - Processed {processed_samples} samples from '{split_name}' split")
        print(f"  - Vocabulary size: {len(self.code_occurrence_overall[split_name]['code_order'])}")
        print(f"  - Total vectors: {len(self.code_occurrence_overall[split_name]['vectors'])}")
        
        return processed_samples





"""
=== CoderSor Usage Summary ===

Key Features:
1. Sequential reasoning analysis preserving temporal order of behaviors
2. Vector representation: dimensions = positions, values = behavior indices
3. Compatible with coder_bor.py format with temporal sequence data
4. Transition matrix classification using averaged model patterns
5. LSTM classification via lstm_classifier.py
6. Automatic reindexing when codebook grows

Data Structure (similar to coder_bor.py):
- code_occurrence_overall[split]['vectors']: sample_id_model -> temporal sequence vector
- code_occurrence_overall[split]['code_order']: behavior order for indexing
- code_occurrence_overall[split]['sample_ids']: sample identifiers  
- code_occurrence_overall[split]['labels']: model labels
- code_occurrence_overall[split]['behavior_to_index']: behavior -> index mapping
- code_occurrence_overall[split]['index_to_behavior']: index -> behavior mapping

Main Methods:
- parse_reasoning_trace_to_vector(): convert annotation to sequence vectors
- classify_with_transition_matrix(): transition pattern classification
- initialize_from_bor_coder(): transfer data from BOR coder
- _reindex_existing_vectors(): reindex when codebook grows

Usage:
    from coder_sor import CoderSor
    from lstm_classifier import classify_with_lstm
    
    coder_sor = CoderSor()
    vector_a, vector_b = coder_sor.parse_reasoning_trace_to_vector(text)
    predictions = coder_sor.classify_with_transition_matrix(test_vectors)
    lstm_predictions = classify_with_lstm(coder_sor, test_vectors)
"""