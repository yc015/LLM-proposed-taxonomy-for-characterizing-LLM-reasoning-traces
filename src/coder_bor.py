from .coder_vllm import CoderVLLM
from .prompts.bag_of_reasoning_inst import CODE_INST, CORRECTION_INST

import numpy as np
import re
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import warnings
warnings.filterwarnings('ignore')


class CoderBOR(CoderVLLM):
    """
    Bag of Reasoning Coder that tracks counts of reasoning behaviors instead of binary presence.
    Inherits from the base Coder class and overrides key methods for count-based analysis.
    
    Features:
    - Count-based reasoning behavior tracking using vectors
    - Robust missing value imputation when new reasoning behaviors are added
    - Support for multiple imputation methods (mean, median, KNN, iterative, etc.)
    - Configurable vector extension with either zeros or NaN values
    - Machine learning classification methods using reasoning vectors
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Override the instruction sets to use bag-of-reasoning format
        self.code_inst = CODE_INST
        self.correction_inst = CORRECTION_INST
        
        # Track reasoning code vectors within the existing code_occurrence_overall structure
        
        # Track code usage counts for weighted averaging
        self._code_usage_counts = {}
        
        # Initialize scaler for feature scaling in ML methods
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
        # Store trained classifiers
        self.logistic_classifier = None
        self.knn_classifier = None
        self.bor_training = True
        self.max_train_samples = 200
        
        # Missing value imputation settings
        self._extend_with_nan = False  # Whether to extend vectors with np.nan instead of 0
        # if self._extend_with_nan:
        self._imputation_method = "mean"  # Options: "mean", "median", "most_frequent", "constant", "knn", "iterative"
        self._impute_after_accumulation = True  # Whether to impute after accumulation phases
        self._imputation_constant_value = 0  # Value to use for constant imputation
        self._knn_imputation_neighbors = 10  # Number of neighbors for KNN imputation
        self._iterative_imputation_max_iter = 10  # Max iterations for iterative imputation
        
        # Averaging method for similarity-based classification
        self._use_weighted_averaging = False  # Whether to use weighted averaging based on code usage counts
        
        # Vector normalization settings
        self._normalize_vectors = False  # Whether to normalize vectors before classification
        self._normalization_method = "comparative"  # Options: "l1" (sum normalization), "l2" (unit vector), "comparative" (normalize pairs by sum), "none"

    def _classify_prompt(self, outputs, question):
        coding_prompt = f"You will be given two reasoning outputs sampled from two different models, and your task is to annotated the occurrence of reasoning behaviors in the given outputs based on the definitions and examples of reasoning behaviors in a taxonomy.\n\nThe system message provides a reasoning taxonomy that illustrates the reasoning behaviors. Follow the instruction and reasoning taxonomy in the system message and this message closely when making annotations. Your output should only contain the annotated reasoning traces."

        # coding_prompt = f"You will be given two reasoning outputs sampled from two different models. These are reasonings for the question:\n{question}\n\nYour task is to distinguish which models do these reasoning outputs belong to based on the distinguishing reasoning traits listed in the codebook."

        # option_prompt = f"Selecting from models: " + ", ".join(self.model_options)

        classification_prompt = """ """

        system_prompt = f"""{self.code_inst}\n"""
        
        for code in self.codebook.keys():
            # code_definition = self.codebook[code]
            # if f"[Exhibited by {self.model_options[0]} and {self.model_options[1]}]" in code_definition or f"[Exhibited by {self.model_options[1]} and {self.model_options[0]}]" in code_definition:
            #     continue

                system_prompt += f"{code}: {self.codebook[code]}\n\n"

        if self.think_budget > 0:
            system_prompt += f"\nThink efficiently. Limit your thinking to {self.think_budget} words."

        final_prompt = f"""{coding_prompt}\n\nGiven the ### OUTPUT A:\n{outputs[0]}\n-----End of OUTPUT A-----\n\nand ### OUTPUT B:\n{outputs[1]}\n-----End of OUTPUT B-----\n\nYour output should just contain annotated reasoning traces. Do not summarize your annotation."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ]
        
    def _parse_codes_occurence(self, classification, labels=None):
        """
        Override to extract both binary presence and counts of reasoning behaviors.
        
        Args:
            classification (str): The classification output text
            labels (list, optional): Ground truth labels
            
        Returns:
            dict: Dictionary with code names as keys and {model_a: count, model_b: count} as values
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
                "this reasoning behavior is", "because of this reasoning behavior", "output a", "output b", 
                "reasoning behavior name", ":", "whether this", "how many times", "count in"
            ]):
                # This is likely a rule name
                rule_name = line.strip("#").strip()
                
                # Skip if it's too long (likely not a rule name) or too short
                if len(rule_name.split()) > 10 or len(rule_name.strip()) == 0:
                    continue
                    
                current_code = rule_name
                code_occurrences[current_code] = {model_a_name: 0, model_b_name: 0}
            
            # Check for count statements about the current rule
            elif current_code is not None:
                # Look for count patterns like "Count in OUTPUT A: 3"
                count_pattern_a = rf"count in output a:\s*(\d+)"
                count_pattern_b = rf"count in output b:\s*(\d+)"
                
                match_a = re.search(count_pattern_a, line.lower())
                match_b = re.search(count_pattern_b, line.lower())
                
                if match_a:
                    try:
                        count = int(match_a.group(1))
                        code_occurrences[current_code][model_a_name] = count
                    except (ValueError, IndexError):
                        code_occurrences[current_code][model_a_name] = 0
                        
                if match_b:
                    try:
                        count = int(match_b.group(1))
                        code_occurrences[current_code][model_b_name] = count
                    except (ValueError, IndexError):
                        code_occurrences[current_code][model_b_name] = 0
                        
                # Also check for binary presence statements as fallback
                if "is observed in output a" in line.lower():
                    # If we haven't set a count yet, set it to 1
                    if code_occurrences[current_code][model_a_name] == 0:
                        code_occurrences[current_code][model_a_name] = 1
                elif "is not observed in output a" in line.lower() or "not observed in output a" in line.lower():
                    code_occurrences[current_code][model_a_name] = 0
                    
                if "is observed in output b" in line.lower():
                    # If we haven't set a count yet, set it to 1
                    if code_occurrences[current_code][model_b_name] == 0:
                        code_occurrences[current_code][model_b_name] = 1
                elif "is not observed in output b" in line.lower() or "not observed in output b" in line.lower():
                    code_occurrences[current_code][model_b_name] = 0
        
        return code_occurrences
    
    def _update_prompt(self, outputs, labels):
        correction_prompt = "You will be given two reasoning outputs and the names of their author models. Given the current reasoning taxonomy, think step by step, your task is to check if there are any new distinguishing reasoning behaviors that can be added to the taxonomy to better separate these two reasonings.\n\n"

        correction_prompt += f"Below is the ### OUTPUT A from {labels[0]}:\n{outputs[0]}\n-----End of OUTPUT A-----\n\n"

        correction_prompt += f"Below is the ### OUTPUT B from {labels[1]}:\n{outputs[1]}\n-----End of OUTPUT B-----\n\n"

        if self.current_annotation is not None:
            correction_prompt += f"This is your annotation following the existing reasoning taxonomy:\n{self.current_annotation}\n\n"

        correction_prompt += f"Follow the instruction in the system prompt and this message to expand the reasoning taxonomy. Remember to annotate the given OUTPUTs first, then summarize your annotations follow the provided format. Finally, add new behaviors that are significantly different from the existing ones.\n\nGive detailed definition and example for the added reasoning behaviors!\n\n"

        correction_prompt += f"Make sure your final output follows the format in the system message exactly. The added reasoning behaviors need to be significantly different from the existing ones in the taxonomy! It's okay if you did not observe any new reasoning behaviors that one output show more than the other (in other words, make the reasonings output separable), but do not add codes that have already been covered by the given taxonomy in the system message."

        system_prompt = self.correction_inst
        included_code = 0
        for code in self.codebook.keys():
            system_prompt += f"{code}: {self.codebook[code]}\n\n"
            included_code += 1

        if included_code < 1:
            system_prompt += f"Current reasoning taxonomy is empty. You should compare the reasoning traces and add new distinguishing reasoning behaviors into this reasoning taxonomy!"

            if self.initial_code_example is not None and len(self.initial_code_example.keys()) > 0:
                system_prompt += "\n\nTo help you get started, below are examples of distinguishing reasoning behaviors of an imagined language model Alpha. You should follow the format, style, and detailedness of these examples when generating the added and updated reasoning behaviors.\n\n"
                for code in list(self.initial_code_example.keys()):
                    system_prompt += f"{code}: {self.initial_code_example[code]}\n\n"

        if self.think_budget > 0:
            system_prompt += f"\n\nThink efficiently. Limit your thinking to {self.think_budget} words."

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": correction_prompt}]
                    
        return messages
    
    def _update_code_occurence(self, results_dict, code_occurrence, sample_id=None, is_decision_code=None):
        """
        Override to handle both count-based reasoning vectors and binary decision codes.
        
        Args:
            results_dict (dict): The dictionary to update (can be reasoning or decision code dict)
            code_occurrence (dict): The parsed code occurrences
            sample_id (str, optional): Identifier for the sample being processed
            is_decision_code (bool, optional): Whether this is decision code data. If None, auto-detect.
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
            super()._update_code_occurence(results_dict, code_occurrence)
        else:
            # Handle reasoning codes with counts - store vectors instead of just totals
            self._update_reasoning_code_vectors(results_dict, code_occurrence, sample_id)
            # # Store for later reference following the parent class pattern
            # if results_dict == self.code_occurrence_overall.get("train", {}):
            #     self._most_recent_code_occurrence_overall_train = deepcopy(results_dict)
    
    def _update_reasoning_code_vectors(self, results_dict, code_occurrence, sample_id=None):
        """
        Update the reasoning code representation by storing count vectors for each sample.
        
        Args:
            results_dict (dict): The dictionary to update
            code_occurrence (dict): The parsed code occurrences with counts
                                  Structure: {code_name: {model_name: count, ...}, ...}
            sample_id (str, optional): Identifier for the sample
        """
        # If this is the first time we're updating reasoning codes, initialize the structure
        if 'vectors' not in results_dict:
            results_dict['vectors'] = {}
            results_dict['code_order'] = []  # Track order of codes for consistent vector indexing
            results_dict['sample_ids'] = []
            results_dict['labels'] = []
        
        # Get current codebook order
        current_codes = list(self.codebook.keys()) if self.codebook else []
        
        # If codebook has grown, extend existing vectors
        if len(current_codes) > len(results_dict['code_order']):
            self._extend_existing_vectors(results_dict, current_codes)
        
        # Update code order
        results_dict['code_order'] = current_codes[:]
        
        # Extract model names directly from code_occurrence structure
        model_names = []
        if code_occurrence:
            first_code_data = next(iter(code_occurrence.values()))
            if isinstance(first_code_data, dict):
                model_names = list(first_code_data.keys())
        
        # Handle case where we don't have exactly 2 models
        if len(model_names) >= 2:
            model_a_name = model_names[0]
            model_b_name = model_names[1]
        else:
            # Fallback to default names if structure is unexpected
            model_a_name = "OUTPUT A"
            model_b_name = "OUTPUT B"
        
        # Create vectors for this sample
        vector_a = np.zeros(len(current_codes))
        vector_b = np.zeros(len(current_codes))
        
        # Initialize code usage counts if needed
        if 'train' not in self._code_usage_counts:
            self._code_usage_counts['train'] = {}
        
        # Fill in the counts and track code usage
        for code_name, counts in code_occurrence.items():
            if code_name in current_codes:
                code_idx = current_codes.index(code_name)
                vector_a[code_idx] = counts.get(model_a_name, 0)
                vector_b[code_idx] = counts.get(model_b_name, 0)
        
        # Update code usage counts for all codes in the codebook (used in annotation)
        for code_name in current_codes:
            if code_name not in self._code_usage_counts['train']:
                self._code_usage_counts['train'][code_name] = 0
            self._code_usage_counts['train'][code_name] += 1
        
        # Store vectors with actual model names as labels
        # Sometime sample_id can be 0, which may cause the problem if using "if sample_id" directly
        if sample_id is not None:
            if f"{sample_id}_{model_a_name}" not in results_dict['vectors'].keys():
                results_dict['sample_ids'].extend([f"{sample_id}_{model_a_name}", f"{sample_id}_{model_b_name}"])
                results_dict['labels'].extend([model_a_name, model_b_name])
            results_dict['vectors'][f"{sample_id}_{model_a_name}"] = vector_a
            results_dict['vectors'][f"{sample_id}_{model_b_name}"] = vector_b
        else:
            print("Sample ID not provided: can potentiall cause errors")
            # Generate unique keys
            base_key = f"sample{len(results_dict['sample_ids'])//2}"
            if f"{base_key}_{model_a_name}" not in results_dict['vectors'].keys():
                results_dict['sample_ids'].extend([f"{base_key}_A", f"{base_key}_B"])
                results_dict['labels'].extend([model_a_name, model_b_name])
            results_dict['vectors'][f"{base_key}_{model_a_name}"] = vector_a
            results_dict['vectors'][f"{base_key}_{model_b_name}"] = vector_b
    
    def _extend_existing_vectors(self, results_dict, new_code_order):
        """
        Extend existing vectors when new codes are added to the codebook.
        
        Args:
            results_dict (dict): The reasoning code representation dictionary
            new_code_order (list): The new order of codes in the codebook
        """
        old_length = len(results_dict['code_order'])
        new_length = len(new_code_order)
        
        if new_length <= old_length:
            return
        
        # Determine fill value based on extension method
        fill_value = np.nan if self._extend_with_nan else 0
        
        # Extend all existing vectors with zeros or NaN
        for key, vector in results_dict['vectors'].items():
            current_length = len(vector)
            if current_length >= new_length:
                # Vector is already the right size or larger
                continue
                
            extended_vector = np.full(new_length, fill_value)
            extended_vector[:current_length] = vector
            results_dict['vectors'][key] = extended_vector
    
    def _impute_missing_values(self, results_dict):
        """
        Impute missing values in the reasoning vectors using sklearn imputation methods.
        Imputation is performed separately for each model to avoid cross-contamination.
        
        Args:
            results_dict (dict): The reasoning code representation dictionary
        """
        if 'vectors' not in results_dict or not results_dict['vectors']:
            return
        
        # Get all vectors and their corresponding labels
        vector_keys = list(results_dict['vectors'].keys())
        vectors = np.array([results_dict['vectors'][key] for key in vector_keys])
        labels = results_dict['labels']
        
        # Check if there are any missing values
        if not np.any(np.isnan(vectors)):
            return
        
        print(f"Imputing missing values using {self._imputation_method} method (separately by model)...")
        
        # Group vectors by model labels
        unique_models = list(set(labels))
        imputed_results = {}
        
        for model in unique_models:
            # Get indices for this model
            model_indices = [i for i, label in enumerate(labels) if label == model]
            
            if not model_indices:
                continue
                
            # Get vectors for this model
            model_vectors = vectors[model_indices]
            model_keys = [vector_keys[i] for i in model_indices]
            
            # Check if this model's vectors have missing values
            if not np.any(np.isnan(model_vectors)):
                # No missing values for this model, store as is
                for i, key in enumerate(model_keys):
                    imputed_results[key] = model_vectors[i]
                continue
            
            print(f"  Imputing for model '{model}': {len(model_vectors)} vectors")
            
            # Choose imputation method
            if self._imputation_method == "mean":
                imputer = SimpleImputer(strategy='mean')
            elif self._imputation_method == "median":
                imputer = SimpleImputer(strategy='median')
            elif self._imputation_method == "most_frequent":
                imputer = SimpleImputer(strategy='most_frequent')
            elif self._imputation_method == "constant":
                imputer = SimpleImputer(strategy='constant', fill_value=self._imputation_constant_value)
            elif self._imputation_method == "knn":
                # Adjust k for small samples
                k = min(self._knn_imputation_neighbors, len(model_vectors) - 1) if len(model_vectors) > 1 else 1
                imputer = KNNImputer(n_neighbors=k)
            elif self._imputation_method == "iterative":
                imputer = IterativeImputer(max_iter=self._iterative_imputation_max_iter, random_state=42)
            else:
                print(f"Unknown imputation method: {self._imputation_method}, falling back to mean")
                imputer = SimpleImputer(strategy='mean')
            
            try:
                # Handle case where there's only one sample for this model
                if len(model_vectors) == 1:
                    # For single sample, use zero imputation as fallback
                    imputed_model_vectors = np.nan_to_num(model_vectors, nan=0.0)
                    print(f"    Single sample for model '{model}', using zero imputation")
                else:
                    # Fit and transform the vectors for this model
                    imputed_model_vectors = imputer.fit_transform(model_vectors)
                    
                    # Handle potential sparse matrix results
                    if hasattr(imputed_model_vectors, 'toarray'):
                        imputed_model_vectors = imputed_model_vectors.toarray()
                
                # Store imputed vectors for this model
                for i, key in enumerate(model_keys):
                    imputed_results[key] = imputed_model_vectors[i]
                
                print(f"    Successfully imputed missing values for {len(model_keys)} vectors of model '{model}'")
                
            except Exception as e:
                print(f"    Error during imputation for model '{model}': {e}")
                print(f"    Falling back to zero imputation for model '{model}'...")
                # Fallback to zero imputation for this model
                for i, key in enumerate(model_keys):
                    vector = model_vectors[i]
                    imputed_results[key] = np.nan_to_num(vector, nan=0.0)
        
        # Update the vectors in the results dictionary
        for key, imputed_vector in imputed_results.items():
            results_dict['vectors'][key] = imputed_vector
        
        print(f"Completed imputation for {len(unique_models)} models, {len(imputed_results)} total vectors")
    
    def _parse_classification_codes(self, final_output, labels):
        """
        Override to parse both reasoning and decision codes.
        """
        # Parse reasoning codes (counts)
        code_occurrence = self._parse_codes_occurence(final_output, labels)
        
        # Parse decision codes (binary) - unchanged from parent
        decision_code = self._parse_decision_code(final_output)
        
        return code_occurrence, decision_code
    
    def _multinomial_naive_bayes_classification(self, classification, smoothing=1.0, print_details=True, return_scores=False, input_vectors=None):
        """
        Classify outputs using multinomial naive Bayes based on count vectors.
        
        Args:
            classification (str): The classification output text (ignored if input_vectors provided)
            smoothing (float): Laplace smoothing parameter
            print_details (bool): Whether to print detailed information
            return_scores (bool): Whether to return scores
            input_vectors (tuple, optional): Pre-computed (vector_a, vector_b) to use directly
            
        Returns:
            tuple: (predicted_model_a, predicted_model_b, [optional scores])
        """
        train_data = self._most_recent_code_occurrence_overall_train if self._most_recent_code_occurrence_overall_train is not None else self.code_occurrence_overall["train"]
        
        if 'vectors' not in train_data or not train_data['vectors']:
            # No vector data, fall back to generative classification
            return self._generative_classification(classification, print_details)
        
        if input_vectors is not None:
            # Use pre-computed vectors
            vector_a, vector_b = input_vectors
        else:
            # Parse code occurrences and create vectors
            code_occurrences = self._parse_codes_occurence(classification)
            
            # Create current sample vectors
            current_codes = train_data['code_order']
            vector_a = np.zeros(len(current_codes))
            vector_b = np.zeros(len(current_codes))
            
            for code_name, counts in code_occurrences.items():
                if code_name in current_codes:
                    code_idx = current_codes.index(code_name)
                    vector_a[code_idx] = counts.get("OUTPUT A", 0)
                    vector_b[code_idx] = counts.get("OUTPUT B", 0)
        
        # Apply normalization if configured
        if self._normalize_vectors and self._normalization_method == "comparative":
            vector_a, vector_b = self._normalize_vector_pair(vector_a, vector_b)
        else:
            vector_a = self._normalize_vector(vector_a)
            vector_b = self._normalize_vector(vector_b)
        
        # Calculate log probabilities for each model
        log_prob_a_model0, log_prob_a_model1 = self._calculate_multinomial_log_probs(vector_a, train_data, smoothing)
        log_prob_b_model0, log_prob_b_model1 = self._calculate_multinomial_log_probs(vector_b, train_data, smoothing)
        
        # Make predictions
        pred_a = self.model_options[0] if log_prob_a_model0 > log_prob_a_model1 else self.model_options[1]
        pred_b = self.model_options[0] if log_prob_b_model0 > log_prob_b_model1 else self.model_options[1]
        
        # Handle case where both outputs are predicted to be from the same model
        if pred_a == pred_b:
            if pred_a == self.model_options[0]:
                if log_prob_a_model0 > log_prob_b_model0:
                    pred_a = self.model_options[0]
                    pred_b = self.model_options[1]
                else:
                    pred_a = self.model_options[1]
                    pred_b = self.model_options[0]
            else:
                if log_prob_a_model1 > log_prob_b_model1:
                    pred_a = self.model_options[1]
                    pred_b = self.model_options[0]
                else:
                    pred_a = self.model_options[0]
                    pred_b = self.model_options[1]
        
        if print_details:
            print(f"Multinomial NB - OUTPUT A log probs: {self.model_options[0]}={log_prob_a_model0:.3f}, {self.model_options[1]}={log_prob_a_model1:.3f}")
            print(f"Multinomial NB - OUTPUT B log probs: {self.model_options[0]}={log_prob_b_model0:.3f}, {self.model_options[1]}={log_prob_b_model1:.3f}")
        
        if return_scores:
            return pred_a, pred_b, [log_prob_a_model0, log_prob_a_model1], [log_prob_b_model0, log_prob_b_model1]
        return pred_a, pred_b
    
    def _calculate_multinomial_log_probs(self, vector, train_data, smoothing):
        """
        Calculate log probabilities for multinomial naive Bayes.
        """
        # Ensure all vectors have the same length (fix for codebook growth)
        expected_length = len(train_data['code_order'])
        error_occurred = False
        for key, vec in train_data['vectors'].items():
            if len(vec) != expected_length:
                error_occurred = True
                if error_occurred:
                    print(f"This should not happen but vector length mismatch occurred")
                extended_vector = np.zeros(expected_length)
                extended_vector[:len(vec)] = vec
                train_data['vectors'][key] = extended_vector
        
        # Get vectors and labels from training data
        vectors = np.array([train_data['vectors'][key] for key in train_data['vectors'].keys()])
        labels = train_data['labels']
        
        # Apply normalization to training vectors if configured
        if self._normalize_vectors:
            if self._normalization_method == "comparative":
                vectors = self._apply_comparative_normalization_to_training_data(train_data)
            else:
                normalized_vectors = []
                for vec in vectors:
                    normalized_vectors.append(self._normalize_vector(vec))
                vectors = np.array(normalized_vectors)
        
        # Separate vectors by model
        model0_vectors = vectors[np.array(labels) == self.model_options[0]]
        model1_vectors = vectors[np.array(labels) == self.model_options[1]]
        
        if len(model0_vectors) == 0 or len(model1_vectors) == 0:
            return 0.0, 0.0
        
        # Calculate feature probabilities for each model
        # Sum of all feature counts for each model
        model0_total_counts = np.sum(model0_vectors, axis=0) + smoothing
        model1_total_counts = np.sum(model1_vectors, axis=0) + smoothing
        
        # Total words (sum of all features) for each model
        model0_total_words = np.sum(model0_total_counts)
        model1_total_words = np.sum(model1_total_counts)
        
        # Calculate feature probabilities
        model0_feature_probs = model0_total_counts / model0_total_words
        model1_feature_probs = model1_total_counts / model1_total_words
        
        # Calculate log probability for the input vector
        log_prob_model0 = np.log(len(model0_vectors) / len(vectors))  # Prior
        log_prob_model1 = np.log(len(model1_vectors) / len(vectors))  # Prior
        
        # Add log probabilities for each feature count
        for i, count in enumerate(vector):
            if count > 0:
                log_prob_model0 += count * np.log(model0_feature_probs[i])
                log_prob_model1 += count * np.log(model1_feature_probs[i])
        
        return log_prob_model0, log_prob_model1
    
    def _similarity_based_classification(self, classification, metric='cosine', print_details=True, return_scores=False, input_vectors=None):
        """
        Classify based on similarity to average vectors of each model.
        
        Args:
            classification (str): The classification output text (ignored if input_vectors provided)
            metric (str): 'cosine' or 'euclidean'
            print_details (bool): Whether to print detailed information
            return_scores (bool): Whether to return scores
            input_vectors (tuple, optional): Pre-computed (vector_a, vector_b) to use directly
            
        Returns:
            tuple: (predicted_model_a, predicted_model_b, [optional scores])
        """
        train_data = self._most_recent_code_occurrence_overall_train if self._most_recent_code_occurrence_overall_train is not None else self.code_occurrence_overall["train"]
        
        if 'vectors' not in train_data or not train_data['vectors']:
            print("Fall back to generative classification")
            return self._generative_classification(classification, print_details)
        
        if input_vectors is not None:
            # Use pre-computed vectors
            vector_a, vector_b = input_vectors
        else:
            # Parse code occurrences and create vectors
            code_occurrences = self._parse_codes_occurence(classification)
            
            # Create current sample vectors
            current_codes = train_data['code_order']
            vector_a = np.zeros(len(current_codes))
            vector_b = np.zeros(len(current_codes))
            
            for code_name, counts in code_occurrences.items():
                if code_name in current_codes:
                    code_idx = current_codes.index(code_name)
                    vector_a[code_idx] = counts.get("OUTPUT A", 0)
                    vector_b[code_idx] = counts.get("OUTPUT B", 0)
        
        # Apply normalization if configured
        if self._normalize_vectors and self._normalization_method == "comparative":
            vector_a, vector_b = self._normalize_vector_pair(vector_a, vector_b)
        else:
            vector_a = self._normalize_vector(vector_a)
            vector_b = self._normalize_vector(vector_b)
        
        # Calculate average vectors for each model (weighted or simple based on configuration)
        if hasattr(self, "_use_weighted_averaging") and self._use_weighted_averaging:
            averaging_method = "weighted"
            avg_vector_model0, avg_vector_model1 = self._get_weighted_average_vectors(train_data)
        else:
            averaging_method = "simple"
            avg_vector_model0, avg_vector_model1 = self._get_simple_average_vectors(train_data)
        
        if avg_vector_model0 is None or avg_vector_model1 is None:
            print("Fall back to generative classification due to some average representation missing")
            return self._generative_classification(classification, print_details)
        
        # Calculate similarities
        if metric == 'cosine':
            sim_a_model0 = cosine_similarity([vector_a], [avg_vector_model0])[0][0]
            sim_a_model1 = cosine_similarity([vector_a], [avg_vector_model1])[0][0]
            sim_b_model0 = cosine_similarity([vector_b], [avg_vector_model0])[0][0]
            sim_b_model1 = cosine_similarity([vector_b], [avg_vector_model1])[0][0]
        else:  # euclidean (inverted to similarity)
            dist_a_model0 = euclidean_distances([vector_a], [avg_vector_model0])[0][0]
            dist_a_model1 = euclidean_distances([vector_a], [avg_vector_model1])[0][0]
            dist_b_model0 = euclidean_distances([vector_b], [avg_vector_model0])[0][0]
            dist_b_model1 = euclidean_distances([vector_b], [avg_vector_model1])[0][0]
            # Convert distance to similarity
            sim_a_model0 = 1 / (1 + dist_a_model0)
            sim_a_model1 = 1 / (1 + dist_a_model1)
            sim_b_model0 = 1 / (1 + dist_b_model0)
            sim_b_model1 = 1 / (1 + dist_b_model1)
        
        # Make predictions
        pred_a = self.model_options[0] if sim_a_model0 > sim_a_model1 else self.model_options[1]
        pred_b = self.model_options[0] if sim_b_model0 > sim_b_model1 else self.model_options[1]
        
        # Handle case where both outputs are predicted to be from the same model
        if pred_a == pred_b:
            if pred_a == self.model_options[0]:
                if sim_a_model0 > sim_b_model0:
                    pred_a = self.model_options[0]
                    pred_b = self.model_options[1]
                else:
                    pred_a = self.model_options[1]
                    pred_b = self.model_options[0]
            else:
                if sim_a_model1 > sim_b_model1:
                    pred_a = self.model_options[1]
                    pred_b = self.model_options[0]
                else:
                    pred_a = self.model_options[0]
                    pred_b = self.model_options[1]
        
        if print_details:
            print(f"{metric.title()} similarity ({averaging_method} averaging) - OUTPUT A: {self.model_options[0]}={sim_a_model0:.3f}, {self.model_options[1]}={sim_a_model1:.3f}")
            print(f"{metric.title()} similarity ({averaging_method} averaging) - OUTPUT B: {self.model_options[0]}={sim_b_model0:.3f}, {self.model_options[1]}={sim_b_model1:.3f}")
        
        if return_scores:
            return pred_a, pred_b, [sim_a_model0, sim_a_model1], [sim_b_model0, sim_b_model1]
        return pred_a, pred_b
    
    def _get_weighted_average_vectors(self, train_data):
        """
        Calculate weighted average vectors for each model based on code usage counts.
        
        Args:
            train_data (dict): Training data containing vectors and labels
            
        Returns:
            tuple: (avg_vector_model0, avg_vector_model1) or (None, None) if insufficient data
        """
        # Ensure all vectors have the same length (fix for codebook growth)
        error_occurred = False
        expected_length = len(train_data['code_order'])
        for key, vector in train_data['vectors'].items():
            if len(vector) != expected_length:
                error_occurred = True
                if error_occurred:
                    print(f"This should not happen but vector length mismatch occurred")
                # Extend vector to match current codebook size
                extended_vector = np.zeros(expected_length)
                extended_vector[:len(vector)] = vector
                train_data['vectors'][key] = extended_vector
        
        # Get vectors and labels from training data
        vectors = np.array([train_data['vectors'][key] for key in train_data['vectors'].keys()])
        labels = train_data['labels']
        
        # Apply normalization to training vectors if configured
        if self._normalize_vectors:
            if self._normalization_method == "comparative":
                vectors = self._apply_comparative_normalization_to_training_data(train_data)
            else:
                normalized_vectors = []
                for vector in vectors:
                    normalized_vectors.append(self._normalize_vector(vector))
                vectors = np.array(normalized_vectors)
        
        # Separate vectors by model
        model0_vectors = vectors[np.array(labels) == self.model_options[0]]
        model1_vectors = vectors[np.array(labels) == self.model_options[1]]
        
        if len(model0_vectors) == 0 or len(model1_vectors) == 0:
            return None, None
        
        # Get code usage weights (how many times each code has been used in annotation)
        usage_weights = np.ones(len(train_data['code_order']))
        if 'train' in self._code_usage_counts:
            for i, code_name in enumerate(train_data['code_order']):
                if code_name in self._code_usage_counts['train']:
                    usage_weights[i] = self._code_usage_counts['train'][code_name]
        
        # Calculate simple averages first
        avg_vector_model0 = np.mean(model0_vectors, axis=0) if len(model0_vectors) > 0 else np.zeros(len(train_data['code_order']))
        avg_vector_model1 = np.mean(model1_vectors, axis=0) if len(model1_vectors) > 0 else np.zeros(len(train_data['code_order']))
        
        # Apply weighting to each feature dimension based on code usage frequency
        # Features with higher usage counts get amplified
        avg_vector_model0 = avg_vector_model0 * usage_weights
        avg_vector_model1 = avg_vector_model1 * usage_weights
        
        return avg_vector_model0, avg_vector_model1
    
    def _get_simple_average_vectors(self, train_data):
        """
        Calculate simple average vectors for each model (no weighting).
        
        Args:
            train_data (dict): Training data containing vectors and labels
            
        Returns:
            tuple: (avg_vector_model0, avg_vector_model1) or (None, None) if insufficient data
        """
        # Ensure all vectors have the same length (fix for codebook growth)
        error_occurred = False
        expected_length = len(train_data['code_order'])
        for key, vector in train_data['vectors'].items():
            if len(vector) != expected_length:
                error_occurred = True
                if error_occurred:
                    print(f"This should not happen but vector length mismatch occurred")
                # Extend vector to match current codebook size
                extended_vector = np.zeros(expected_length)
                extended_vector[:len(vector)] = vector
                train_data['vectors'][key] = extended_vector
        
        # Get vectors and labels from training data
        vectors = np.array([train_data['vectors'][key] for key in train_data['vectors'].keys()])
        labels = train_data['labels']
        
        # Apply normalization to training vectors if configured
        if self._normalize_vectors:
            if self._normalization_method == "comparative":
                vectors = self._apply_comparative_normalization_to_training_data(train_data)
            else:
                normalized_vectors = []
                for vector in vectors:
                    normalized_vectors.append(self._normalize_vector(vector))
                vectors = np.array(normalized_vectors)
        
        # Separate vectors by model
        model0_vectors = vectors[np.array(labels) == self.model_options[0]]
        model1_vectors = vectors[np.array(labels) == self.model_options[1]]
        
        if len(model0_vectors) == 0 or len(model1_vectors) == 0:
            return None, None
        
        # Calculate simple averages (no weighting)
        avg_vector_model0 = np.mean(model0_vectors, axis=0)
        avg_vector_model1 = np.mean(model1_vectors, axis=0)
        
        return avg_vector_model0, avg_vector_model1
    
    def _knn_classification(self, classification, k=11, print_details=True, return_scores=False, input_vectors=None):
        """
        Classify using K-Nearest Neighbors.
        
        Args:
            classification (str): The classification output text (ignored if input_vectors provided)
            k (int): Number of neighbors to consider
            print_details (bool): Whether to print detailed information
            return_scores (bool): Whether to return scores
            input_vectors (tuple, optional): Pre-computed (vector_a, vector_b) to use directly
            
        Returns:
            tuple: (predicted_model_a, predicted_model_b, [optional scores])
        """
        train_data = self._most_recent_code_occurrence_overall_train if self._most_recent_code_occurrence_overall_train is not None else self.code_occurrence_overall["train"]
        
        if 'vectors' not in train_data or not train_data['vectors']:
            print("Fall back to generative classification")
            return self._generative_classification(classification, print_details)
        
        if input_vectors is not None:
            # Use pre-computed vectors
            vector_a, vector_b = input_vectors
        else:
            # Parse code occurrences and create vectors
            code_occurrences = self._parse_codes_occurence(classification)
            
            # Create current sample vectors
            current_codes = train_data['code_order']
            vector_a = np.zeros(len(current_codes))
            vector_b = np.zeros(len(current_codes))
            
            for code_name, counts in code_occurrences.items():
                if code_name in current_codes:
                    code_idx = current_codes.index(code_name)
                    vector_a[code_idx] = counts.get("OUTPUT A", 0)
                    vector_b[code_idx] = counts.get("OUTPUT B", 0)
        
        # Apply normalization if configured
        if self._normalize_vectors and self._normalization_method == "comparative":
            vector_a, vector_b = self._normalize_vector_pair(vector_a, vector_b)
        else:
            vector_a = self._normalize_vector(vector_a)
            vector_b = self._normalize_vector(vector_b)
        
        # Ensure all vectors have the same length (fix for codebook growth)
        expected_length = len(train_data['code_order'])
        for key, vec in train_data['vectors'].items():
            if len(vec) != expected_length:
                extended_vector = np.zeros(expected_length)
                extended_vector[:len(vec)] = vec
                train_data['vectors'][key] = extended_vector
        
        # Get training data
        train_vectors = np.array([train_data['vectors'][key] for key in train_data['vectors'].keys()])
        train_labels = train_data['labels']
        
        # Apply normalization to training vectors if configured
        if self._normalize_vectors:
            if self._normalization_method == "comparative":
                train_vectors = self._apply_comparative_normalization_to_training_data(train_data)
            else:
                normalized_train_vectors = []
                for vector in train_vectors:
                    normalized_train_vectors.append(self._normalize_vector(vector))
                train_vectors = np.array(normalized_train_vectors)
        
        if len(train_vectors) < k:
            k = len(train_vectors)
        
        # Train KNN classifier if not already trained or if data has changed
        if self.knn_classifier is None or len(train_vectors) != getattr(self, '_last_knn_train_size', 0):
            self.knn_classifier = KNeighborsClassifier(n_neighbors=k)
            self.knn_classifier.fit(train_vectors, train_labels)
            self._last_knn_train_size = len(train_vectors)
        
        # Predict
        pred_a = self.knn_classifier.predict([vector_a])[0]
        pred_b = self.knn_classifier.predict([vector_b])[0]
        
        # Get prediction probabilities for scoring
        try:
            proba_a = self.knn_classifier.predict_proba([vector_a])[0]
            proba_b = self.knn_classifier.predict_proba([vector_b])[0]
            
            # Map probabilities to model options
            classes = self.knn_classifier.classes_
            scores_a = [proba_a[list(classes).index(model)] if model in classes else 0.0 for model in self.model_options]
            scores_b = [proba_b[list(classes).index(model)] if model in classes else 0.0 for model in self.model_options]
        except:
            scores_a = [0.5, 0.5]
            scores_b = [0.5, 0.5]
        
        # Handle case where both outputs are predicted to be from the same model
        if pred_a == pred_b:
            if pred_a == self.model_options[0]:
                if scores_a[0] > scores_b[0]:
                    pred_a = self.model_options[0]
                    pred_b = self.model_options[1]
                else:
                    pred_a = self.model_options[1]
                    pred_b = self.model_options[0]
            else:
                if scores_a[1] > scores_b[1]:
                    pred_a = self.model_options[1]
                    pred_b = self.model_options[0]
                else:
                    pred_a = self.model_options[0]
                    pred_b = self.model_options[1]
        
        if print_details:
            print(f"KNN (k={k}) - OUTPUT A probabilities: {dict(zip(self.model_options, scores_a))}")
            print(f"KNN (k={k}) - OUTPUT B probabilities: {dict(zip(self.model_options, scores_b))}")
        
        if return_scores:
            return pred_a, pred_b, scores_a, scores_b
        return pred_a, pred_b
    
    def _logistic_regression_classification(self, classification, print_details=True, return_scores=False, input_vectors=None):
        """
        Classify using Logistic Regression trained on the reasoning vectors.
        
        Args:
            classification (str): The classification output text (ignored if input_vectors provided)
            print_details (bool): Whether to print detailed information
            return_scores (bool): Whether to return scores
            input_vectors (tuple, optional): Pre-computed (vector_a, vector_b) to use directly
            
        Returns:
            tuple: (predicted_model_a, predicted_model_b, [optional scores])
        """
        train_data = self._most_recent_code_occurrence_overall_train if self._most_recent_code_occurrence_overall_train is not None else self.code_occurrence_overall["train"]
        
        if 'vectors' not in train_data or not train_data['vectors']:
            return self._generative_classification(classification, print_details)
        
        if input_vectors is not None:
            # Use pre-computed vectors
            vector_a, vector_b = input_vectors
        else:
            # Parse code occurrences and create vectors
            code_occurrences = self._parse_codes_occurence(classification)
            
            # Create current sample vectors
            current_codes = train_data['code_order']
            vector_a = np.zeros(len(current_codes))
            vector_b = np.zeros(len(current_codes))
            
            for code_name, counts in code_occurrences.items():
                if code_name in current_codes:
                    code_idx = current_codes.index(code_name)
                    vector_a[code_idx] = counts.get("OUTPUT A", 0)
                    vector_b[code_idx] = counts.get("OUTPUT B", 0)
        
        # Apply normalization if configured
        if self._normalize_vectors and self._normalization_method == "comparative":
            vector_a, vector_b = self._normalize_vector_pair(vector_a, vector_b)
        else:
            vector_a = self._normalize_vector(vector_a)
            vector_b = self._normalize_vector(vector_b)
        
        # Ensure all vectors have the same length (fix for codebook growth)
        expected_length = len(train_data['code_order'])
        for key, vec in train_data['vectors'].items():
            if len(vec) != expected_length:
                extended_vector = np.zeros(expected_length)
                extended_vector[:len(vec)] = vec
                train_data['vectors'][key] = extended_vector
        
        # Get training data
        train_vectors = np.array([train_data['vectors'][key] for key in train_data['vectors'].keys()])
        train_labels = train_data['labels']
        
        # Apply normalization to training vectors if configured
        if self._normalize_vectors:
            if self._normalization_method == "comparative":
                train_vectors = self._apply_comparative_normalization_to_training_data(train_data)
            else:
                normalized_train_vectors = []
                for vector in train_vectors:
                    normalized_train_vectors.append(self._normalize_vector(vector))
                train_vectors = np.array(normalized_train_vectors)
        train_vectors_scaled = train_vectors
        
        # Train logistic regression classifier if not already trained or if data has changed
        if self.logistic_classifier is None or len(train_vectors) != getattr(self, '_last_lr_train_size', 0):
            try:
                print(f"Training logistic regression classifier with {len(train_vectors_scaled)} samples")
                self.logistic_classifier = LogisticRegression(random_state=42, max_iter=1000)
                self.logistic_classifier.fit(train_vectors_scaled, train_labels)
                self._last_lr_train_size = len(train_vectors)
            except Exception as e:
                print(f"Error training logistic regression: {e}")
                return self._generative_classification(classification, print_details)
        
        try:
            # Scale test vectors
            test_vectors_scaled = np.array([vector_a, vector_b])
            
            # Predict
            predictions = self.logistic_classifier.predict(test_vectors_scaled)
            pred_a, pred_b = predictions[0], predictions[1]
            
            # Get prediction probabilities for scoring
            probabilities = self.logistic_classifier.predict_proba(test_vectors_scaled)
            
            # Map probabilities to model options
            classes = self.logistic_classifier.classes_
            scores_a = [probabilities[0][list(classes).index(model)] if model in classes else 0.0 for model in self.model_options]
            scores_b = [probabilities[1][list(classes).index(model)] if model in classes else 0.0 for model in self.model_options]
            
        except Exception as e:
            print(f"Error in logistic regression prediction: {e}")
            return self._generative_classification(classification, print_details)
        
        # Handle case where both outputs are predicted to be from the same model
        if pred_a == pred_b:
            if pred_a == self.model_options[0]:
                if scores_a[0] > scores_b[0]:
                    pred_a = self.model_options[0]
                    pred_b = self.model_options[1]
                else:
                    pred_a = self.model_options[1]
                    pred_b = self.model_options[0]
            else:
                if scores_a[1] > scores_b[1]:
                    pred_a = self.model_options[1]
                    pred_b = self.model_options[0]
                else:
                    pred_a = self.model_options[0]
                    pred_b = self.model_options[1]
        
        if print_details:
            print(f"Logistic Regression - OUTPUT A probabilities: {dict(zip(self.model_options, scores_a))}")
            print(f"Logistic Regression - OUTPUT B probabilities: {dict(zip(self.model_options, scores_b))}")
        
        if return_scores:
            return pred_a, pred_b, scores_a, scores_b
        return pred_a, pred_b
    
    def _evaluate_classification(self, classification, labels, print_details=True, input_vectors=None):
        """
        Override to handle new evaluation methods and pre-computed vectors.
        """
        if self.evaluation_method == "generative":
            result = self._generative_classification(classification, print_details=print_details)
        elif self.evaluation_method == "hard_vote":
            result = self._hard_voting_classification(classification, print_details=print_details)
        elif self.evaluation_method == "soft_vote":
            result = self._soft_voting_classification(classification, print_details=print_details)
        elif self.evaluation_method == "naive_bayes":
            result = self._naive_bayes_classification(classification, 
                                                     follow_code_occurrence_exactly=self.follow_codebook_nbc,
                                                     print_details=print_details)
        elif self.evaluation_method == "multinomial_nb":
            result = self._multinomial_naive_bayes_classification(classification, print_details=print_details, input_vectors=input_vectors)
        elif self.evaluation_method == "cosine_similarity":
            result = self._similarity_based_classification(classification, metric='cosine', print_details=print_details, input_vectors=input_vectors)
        elif self.evaluation_method == "euclidean_similarity":
            result = self._similarity_based_classification(classification, metric='euclidean', print_details=print_details, input_vectors=input_vectors)
        elif self.evaluation_method == "knn":
            k_value = getattr(self, "k", 11)  # Default to 5 if k attribute doesn't exist
            result = self._knn_classification(classification, print_details=print_details, k=k_value, input_vectors=input_vectors)
        elif self.evaluation_method == "logistic_regression":
            result = self._logistic_regression_classification(classification, print_details=print_details, input_vectors=input_vectors)
        else:
            # Fallback to generative
            result = self._generative_classification(classification, print_details=print_details)
            
        # Ensure we only get the first two values (model predictions)
        if isinstance(result, tuple) and len(result) >= 2:
            model_a, model_b = result[0], result[1]
        elif isinstance(result, tuple) and len(result) == 1:
            # Handle case where only one value is returned 
            model_a = result[0]
            model_b = self.model_options[1] if model_a == self.model_options[0] else self.model_options[0]
        else:
            # Fallback for malformed results
            print(f"Warning: Unexpected result format from classification method: {result}")
            model_a, model_b = self.model_options[0], self.model_options[1]
            
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

    def check_stop_criteria(self, criteria=["max_rules"]):
        meet_criteria = 0
        if "max_rules" in criteria:
            codebook_size = len(self.codebook.keys()) if self.codebook else 0
            if codebook_size > self.max_rule:
                self.prune_codebook()
                codebook_size = len(self.codebook.keys()) if self.codebook else 0
                if codebook_size > self.max_rule:
                    print(f"The reasoning taxonomy now has {codebook_size} reasoning behaviors.\nExit the training earlier.")
                    meet_criteria += 1

        if "global_patience" in criteria:
            if self.num_consecutive_turns_without_update >= self.global_patience and self.num_train_samples > 50:
                print(f"The reasoning taxonomy has not been changed for {self.num_consecutive_turns_without_update} turns in a row.\nExit the training earlier.")
                meet_criteria += 1
        if "max_train_samples" in criteria:
            if self.num_train_samples > self.max_train_samples:
                print(f"The coder has been trained on {self.num_train_samples} exceed the limit {self.max_train_samples}. Exit the training early.")
                meet_criteria += 1

        return meet_criteria > 0
    
    def eval(self, dataset, verbosity=5, batched=False, batch_size=4, ckpt_path=None, freq_estimate=False):
        """
        Override eval method to handle reasoning code representation for evaluation.
        """
        split = "train_est" if freq_estimate else "eval"
        # Initialize reasoning code representation for this split - use existing structure
        if split not in self.code_occurrence_overall:
            self.code_occurrence_overall[split] = {}
        
        # Not retrained classifier during eval, use the last trained classifier from train
        self.logistic_classifier = None
        self.knn_classifier = None
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        # Call grandparent (Coder) eval method directly to include freq_estimate parameter
        result = super(CoderVLLM, self).eval(dataset, verbosity, batched, batch_size, ckpt_path, freq_estimate)
        
        return result
    
    def eval_with_averaged_vectors(self, dataset, num_reruns=5, verbosity=10, variance_calculation=False):
        """
        Evaluate using averaged vectors from multiple annotation runs.
        
        Args:
            dataset: Evaluation dataset
            num_reruns (int): Number of annotation reruns to average
            verbosity (int): Progress reporting frequency
            variance_calculation (bool): If True, also evaluate individual runs and return variance info
            
        Returns:
            dict: Evaluation results with accuracy and detailed logs
                  If variance_calculation=True, includes 'individual_run_results' with per-run scores
        """
        if not self.codebook:
            print("Codebook is empty. Cannot perform vector-based evaluation.")
            return {"accuracy": 0.0, "logs": {}}
        
        train_data = self.code_occurrence_overall.get("train", {})
        if 'code_order' not in train_data:
            print("No training data available. Cannot perform vector-based evaluation.")
            return {"accuracy": 0.0, "logs": {}}
        
        print(f"Running vector-based evaluation with {num_reruns} reruns per sample...")
        
        # Get individual vectors (single call to rerun_annotations)
        individual_result = self.rerun_annotations(dataset, train_data, num_reruns, return_averaged=False)
        
        # Process results by sample - group vectors by sample_id using actual labels
        sample_vectors = {}  # sample_id -> {model_label: [vector_rerun_0, vector_rerun_1, ...]}
        
        # Group vectors by sample and model using labels
        for i, vector_key in enumerate(individual_result['sample_ids']):
            if '_rerun_' in vector_key:
                # Parse: "sample_id_rerun_X_ModelName"
                parts = vector_key.split('_rerun_')
                sample_id = parts[0]  # Keep original sample_id
                
                vector = individual_result['vectors'][vector_key]
                actual_label = individual_result['labels'][i]  # Use actual label to identify which model
                
                if sample_id not in sample_vectors:
                    sample_vectors[sample_id] = {}
                if actual_label not in sample_vectors[sample_id]:
                    sample_vectors[sample_id][actual_label] = []
                
                sample_vectors[sample_id][actual_label].append(vector)
        # Initialize individual run results storage
        individual_run_results = [] if variance_calculation else None
        
        # Evaluate individual runs if variance calculation is requested
        if variance_calculation:
            print("Evaluating individual runs for variance calculation...")
            for run_idx in range(num_reruns):
                run_correct = 0
                run_total = 0
                run_logs = {}
                
                for sample in dataset:
                    sample_id = f"{sample['id']}"
                    true_labels = sample['labels']
                        
                    # Get vectors for this specific run
                    vector_a = sample_vectors[sample_id][true_labels[0]][run_idx]
                    vector_b = sample_vectors[sample_id][true_labels[1]][run_idx]
                    
                    # Evaluate this run
                    evaluation, pred = self._evaluate_classification(
                        classification="",
                        labels=sample['labels'],
                        print_details=True,
                        input_vectors=(vector_a, vector_b)
                    )
                    
                    if evaluation == "correct":
                        run_correct += 1
                    run_total += 1
                    
                    run_logs[sample_id] = {
                        "evaluation": evaluation,
                        "predicted": pred,
                        "true_labels": sample['labels']
                    }
                
                run_accuracy = (run_correct / run_total) * 100 if run_total > 0 else 0.0
                individual_run_results.append({
                    "run_index": run_idx,
                    "accuracy": run_accuracy,
                    "correct": run_correct,
                    "total": run_total,
                    "logs": run_logs
                })
                
                if verbosity > 0:
                    print(f"  Run {run_idx + 1}: {run_accuracy:.2f}% ({run_correct}/{run_total})")
        
        # Calculate averaged vectors and evaluate
        correct_predictions = 0
        total_samples = 0
        evaluation_logs = {}
        
        for sample_idx, sample in enumerate(dataset):
            sample_id = f"{sample['id']}"
            true_labels = sample['labels']
            
            if (sample_id in sample_vectors and 
                true_labels[0] in sample_vectors[sample_id] and
                true_labels[1] in sample_vectors[sample_id]):
                
                # Get all vectors for both models
                vectors_a = sample_vectors[sample_id][true_labels[0]]
                vectors_b = sample_vectors[sample_id][true_labels[1]]
                
                if vectors_a and vectors_b and len(vectors_a) == len(vectors_b):
                    # Average the vectors across reruns
                    avg_vector_a = np.mean(vectors_a, axis=0)
                    avg_vector_b = np.mean(vectors_b, axis=0)
                    

                    # Evaluate using averaged vectors
                    evaluation, pred = self._evaluate_classification(
                        classification="",
                        labels=sample['labels'],
                        print_details=True,
                        input_vectors=(avg_vector_a, avg_vector_b)
                    )
                    
                    if evaluation == "correct":
                        correct_predictions += 1
                    
                    total_samples += 1
                    evaluation_logs[sample['id']] = {
                        "evaluation": evaluation,
                        "predicted": pred,
                        "true_labels": sample['labels'],
                        "vectors_used": True,
                        "num_reruns_averaged": len(vectors_a)
                    }
                    
                    if verbosity > 0 and sample_idx % verbosity == 0:
                        accuracy = (correct_predictions / total_samples) * 100
                        print(f"Progress: {sample_idx + 1}/{len(dataset)} | Accuracy: {accuracy:.2f}%")

        
        final_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0.0
        print(f"Vector-based evaluation complete. Accuracy: {final_accuracy:.2f}% ({correct_predictions}/{total_samples})")
        
        result = {
            "accuracy": final_accuracy,
            "correct": correct_predictions,
            "total": total_samples,
            "logs": evaluation_logs
        }
        
        # Add individual run results if variance calculation was performed
        if variance_calculation and individual_run_results:
            result["individual_run_results"] = individual_run_results
            
            # Calculate variance statistics
            run_accuracies = [run["accuracy"] for run in individual_run_results]
            result["variance_stats"] = {
                "mean_accuracy": np.mean(run_accuracies),
                "std_accuracy": np.std(run_accuracies),
                "min_accuracy": np.min(run_accuracies),
                "max_accuracy": np.max(run_accuracies),
                "individual_accuracies": run_accuracies
            }
            
            print(f"Individual run variance: mean={result['variance_stats']['mean_accuracy']:.2f}% "
                  f"(±{result['variance_stats']['std_accuracy']:.2f}%), "
                  f"range=[{result['variance_stats']['min_accuracy']:.2f}%, {result['variance_stats']['max_accuracy']:.2f}%]")
        
        return result
    
    def train(self, dataset, verbosity=5, custom_logger=None, logger_kwargs=None, ckpt_path=None, batch_size=32, accumulate_observation_training=False, accumulation_size=10, sampling_training=False):
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
        import os
        
        self.num_train_samples = 0
        self.num_train_correct = 0
        self.num_update_samples = 0
        self.num_update_correct = 0
        self.train_correct_ids = []
        self.train_wrong_ids = []
        self.in_training = True

        self.logistic_classifier = None
        self.knn_classifier = None
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False

        if "train" not in self.training_logs:
            self.training_logs["train"] = {}
        if "train" not in self.code_occurrence_overall:
            self.code_occurrence_overall["train"] = {}
        if "train" not in self.decision_code_occurence_overall:
            self.decision_code_occurence_overall["train"] = {}
        
        if (ckpt_path is not None) and os.path.exists(ckpt_path):
            print(f"Loading from pretrained attr from {ckpt_path}")
            self.from_pretrained(filename=ckpt_path)
        else:
            print("TRAIN FROM SCRATCH. No existing pretrained data found.")

        start_at = self.num_train_samples
        
        if accumulate_observation_training:
            if sampling_training:
                print(f"Using stochastic sampling accumulate observation training with accumulation size: {accumulation_size}")
                self._train_with_accumulation_stochastic_sampling(dataset, start_at, verbosity, custom_logger, logger_kwargs, ckpt_path, batch_size, accumulation_size)
            else:
                print(f"Using sequential accumulate observation training with accumulation size: {accumulation_size}")
                self._train_with_accumulation(dataset, start_at, verbosity, custom_logger, logger_kwargs, ckpt_path, batch_size, accumulation_size)
        else:
            # Use parent's training method
            super().train(dataset, verbosity, custom_logger, logger_kwargs, ckpt_path, batch_size)
        self._most_recent_code_occurrence_overall_train = None
        self._most_recent_decision_code_overall_train = None
    
    def _train_with_accumulation(self, dataset, start_at, verbosity, custom_logger, logger_kwargs, ckpt_path, batch_size, accumulation_size):
        """
        Training with clean accumulation → update cycle:
        1. Accumulation Phase: Process accumulation_size samples with train=False (batched for speed)
        2. Update Phase: Process new samples with train=True until codebook changes
        3. Repeat until stop criteria met
        """
        import os

        if self._extend_with_nan:
            print("Extend the vector with np.nan. Thus:")
            self.no_check_after_update = True
            print("No check after update:", self.no_check_after_update)
        
        idx = start_at
        in_accumulation_mode = True
        accumulation_count = 0
        samples_accumulated_this_round = 0
        accumulation_batch = []
        # Store classification results for proper evaluation after vector updates
        pending_samples = []
        
        while idx < len(dataset) and self.in_training:
            sample = dataset[idx]
            
            if in_accumulation_mode:
                self.logistic_classifier = None
                self.knn_classifier = None
                self.is_scaler_fitted = False

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
                    
                    # Batch classify with train=False and no_eval=True to get vectors first
                    classification_prompts, classifications, evaluations, if_codebook_updated_list, final_outputs = self.classify(
                        outputs, labels, questions, train=False, qid=ids, batched=True, no_eval=True,
                    )
                    
                    # Store results for later evaluation (after vector updates and imputation)
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
                    
                    # Now process all pending samples: update vectors, impute, then evaluate
                    self._process_pending_accumulation_samples(pending_samples, custom_logger, logger_kwargs)
                    pending_samples = []  # Clear after processing
                    
                    accuracy = (self.num_train_correct / self.num_train_samples) * 100 if self.num_train_samples > 0 else 0
                    print(f"Overall accuracy so far: {accuracy:.2f}% ({self.num_train_correct}/{self.num_train_samples})")
                    
                    print("Switching to UPDATE MODE")
                    samples_accumulated_this_round = 0
                    in_accumulation_mode = False

                if (not in_accumulation_mode) and self.check_stop_criteria(criteria=self.stop_criteria):
                    self.in_training = False
                    print("STOP CRITERIA REACHED")
                    if in_accumulation_mode:
                        print("There was an update to the codebook")
                        print("Do one more round of accumulation before quiting")
                    else:
                        break

            else:
                # Update Phase: Process new samples with training enabled until codebook changes
                print(f"Update mode: Processing new sample {sample['id']}")
                self.num_update_samples += 1
                # Get current codebook size to detect changes
                current_codebook_size = len(self.codebook.keys()) if self.codebook else 0
                
                classification_prompt, classification, evaluation, if_codebook_updated, final_output = self.classify(
                    sample["outputs"], sample["labels"], sample["question"], 
                    train=True, qid=f"{sample['id']}"
                )
                
                # Handle if_codebook_updated which might be int or list
                updated = False

                if evaluation != "update-no-check":
                    # Update statistics for the new sample
                    self._update_sample_statistics(sample, classification_prompt, classification, 
                                                   evaluation, if_codebook_updated, final_output, 
                                                   custom_logger, logger_kwargs)
                
                    updated = True

                # Check if codebook was updated (size changed)
                new_codebook_size = len(self.codebook.keys()) if self.codebook else 0
                
                if isinstance(if_codebook_updated, (int, float)):
                    updated = if_codebook_updated > 0
                elif isinstance(if_codebook_updated, list):
                    updated = any(x > 0 for x in if_codebook_updated if isinstance(x, (int, float)))
                
                if new_codebook_size != current_codebook_size:
                    print(f"Codebook updated! Size: {current_codebook_size} → {new_codebook_size}")
                    print("Returning to ACCUMULATION MODE")
                    in_accumulation_mode = True

                if if_codebook_updated > 0:
                    pass
                elif evaluation == "correct":
                    self.num_update_correct += 1
                
                # Check stop criteria after each training sample in update mode
                if (not in_accumulation_mode) and self.check_stop_criteria(criteria=["global_patience"]):
                    self.in_training = False
                    print("STOP CRITERIA REACHED")
                    if in_accumulation_mode:
                        print("There was an update to the codebook")
                        print("Do one more round of accumulation before quiting")
                    else:
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
    
    def _train_with_accumulation_stochastic_sampling(self, dataset, start_at, verbosity, custom_logger, logger_kwargs, ckpt_path, batch_size, accumulation_size):
        """
        Training with stochastic sampling accumulation → update cycle:
        1. Accumulation Phase: Sample accumulation_size samples WITHOUT replacement (batched for speed)
        2. Update Phase: Sample new samples WITH replacement until codebook changes
        3. Repeat until stop criteria met
        4. When no available indices remain, start a new epoch by resetting the available indices
        """
        import os
        import random
        
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
        
        while (self.in_training or in_accumulation_mode):
            
            if in_accumulation_mode:
                self.logistic_classifier = None
                self.knn_classifier = None
                self.scaler = StandardScaler()
                self.is_scaler_fitted = False

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
                        
                        # Batch classify with train=False and no_eval=True to get vectors first
                        classification_prompts, classifications, evaluations, if_codebook_updated_list, final_outputs = self.classify(
                            outputs, labels, questions, train=False, qid=ids, batched=True, no_eval=True,
                        )
                        
                        # Store results for later evaluation (after vector updates and imputation)
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
                
                # Process all pending samples: update vectors, impute, then evaluate
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
                    print("No Update to Codebook. Continue accumulation.\nAlways in accumulation model until reach to the maximum number of training samples.")
                    if self.num_train_samples >= self.max_train_samples:
                        print("Maximum number of training samples reached")
                        break

                current_codebook_size = len(self.codebook.keys()) if self.codebook else 0
                if (not in_accumulation_mode) and self.check_stop_criteria(criteria=self.stop_criteria):
                    new_codebook_size = len(self.codebook.keys()) if self.codebook else 0
                    if new_codebook_size != current_codebook_size:
                        print("Codebook was pruned")
                        print("Run one more accumulation")
                        in_accumulation_mode = True
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
                
                if hasattr(self, "batch_update") and self.batch_update and self.batch_update_size > 1:
                    batched_dataset = random.sample(dataset, self.batch_update_size - 1)
                    batched_outputs = [item["outputs"] for item in batched_dataset]
                    batched_labels = [item["labels"] for item in batched_dataset]
                    batched_questions = [item["question"] for item in batched_dataset]
                    batched_ids = [item["id"] for item in batched_dataset]
                    classification_prompt, classification, evaluation, if_codebook_updated, final_output = self.classify(sample["outputs"], sample["labels"], sample["question"], train=True, qid=f"{sample['id']}", batched=False, additional_outputs=batched_outputs, additional_labels=batched_labels, additional_questions=batched_questions)
                else:
                    classification_prompt, classification, evaluation, if_codebook_updated, final_output = self.classify(
                        sample["outputs"], sample["labels"], sample["question"], 
                        train=True, qid=f"{sample['id']}"
                    )
                
                # Check if codebook was updated
                codebook_was_updated = False
                if isinstance(if_codebook_updated, (int, float)):
                    codebook_was_updated = bool(if_codebook_updated > 0)
                elif isinstance(if_codebook_updated, list):
                    codebook_was_updated = any(x > 0 for x in if_codebook_updated if isinstance(x, (int, float)))
                
                if codebook_was_updated:
                    pass
                elif evaluation == "correct":
                    self.num_update_correct += 1

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
                
                total_samples_processed += 1
                
                # Check if codebook was updated (size changed)
                new_codebook_size = len(self.codebook.keys()) if self.codebook else 0
                
                if new_codebook_size != current_codebook_size:
                    print(f"Codebook updated! Size: {current_codebook_size} → {new_codebook_size}")
                    print("Returning to ACCUMULATION MODE")
                    in_accumulation_mode = True
                
                # Check stop criteria after each training sample in update mode
                # if self.in_training and self.check_stop_criteria(criteria=self.stop_criteria):
                #     self.in_training = False
                #     print("STOP CRITERIA REACHED")
                #     if in_accumulation_mode:
                #         print("There was an update to the codebook")
                #         print("Do one more round of accumulation before quiting")
                #     else:
                #         break
                if (not in_accumulation_mode) and self.check_stop_criteria(criteria=["global_patience"]):
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
    

    
    def _update_sample_statistics(self, sample, classification_prompt, classification, evaluation, if_codebook_updated, final_output, custom_logger, logger_kwargs):
        """
        Update training statistics for a single sample.
        
        Args:
            sample: The sample data
            classification_prompt: The classification prompt
            classification: The classification output
            evaluation: The evaluation result
            if_codebook_updated: Whether codebook was updated
            final_output: The final output
            custom_logger: Custom logging function
            logger_kwargs: Logger keyword arguments
        """
        # Parse codes
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
        
        # Update overall tracking - following parent class flow
        # Check if this was from a training classify call (which updates _most_recent variables)
        # or from accumulation mode (which requires manual updates)
        if self._most_recent_code_occurrence_overall_train is not None:
            # Training mode: retrieve from _most_recent variables (set by classify method)
            self.code_occurrence_overall["train"] = deepcopy(self._most_recent_code_occurrence_overall_train)
            self._most_recent_code_occurrence_overall_train = None
        else:
            # Accumulation mode: manually update code occurrences
            self._update_code_occurence(self.code_occurrence_overall["train"], code_occurrence, is_decision_code=False, sample_id=sample["id"])
        
        if self._most_recent_decision_code_overall_train is not None:
            # Training mode: retrieve from _most_recent variables (set by classify method)
            self.decision_code_occurence_overall["train"] = deepcopy(self._most_recent_decision_code_overall_train) 
            self._most_recent_decision_code_overall_train = None
        else:
            # Accumulation mode: manually update decision code occurrences
            self._update_code_occurence(self.decision_code_occurence_overall["train"], decision_code, is_decision_code=True)
        
        # Update training statistics
        if evaluation == "correct":
            self.train_correct_ids.append(sample["id"])
            self.num_train_correct += 1
        else:
            self.train_wrong_ids.append(sample["id"])
        
        self.num_train_samples += 1
        
        if custom_logger:
            custom_logger(**logger_kwargs)
        
        self.training_logs["train"][sample['id']]["current_train_acc"] = (self.num_train_correct / self.num_train_samples) * 100
    
    def prune_codebook(self, method="top_k", **kwargs):
        """
        Prune the codebook by removing features that cannot distinguish between models.
        
        Args:
            method (str): Pruning method to use. Options:
                - "frequency": Frequency-based pruning (original algorithm) [DEFAULT]
                - "sklearn": Use sklearn feature selection methods
                - "top_k": Keep only the k most discriminative codes
            **kwargs: Additional parameters for specific pruning methods
                
                For frequency method:
                - min_occurrences (int): Minimum total occurrences to consider (default: 10)
                - equality_threshold (float): Threshold for equal distribution (default: 0.05)
                
                For sklearn method:
                - selector_type (str): Type of sklearn selector (default: "k_best")
                - k (int): Number of features to keep for k_best (default: min(20, current_size//2))
                - score_func (str): Scoring function for k_best (default: "mutual_info")
                - percentile (int): Percentile for percentile selector (default: 50)
                
                For top_k method:
                - k (int): Number of most discriminative codes to keep (required)
                - score_func (str): Scoring function for ranking (default: "mutual_info")
        """
        if not self.codebook:
            print("Codebook is empty. No pruning needed.")
            return
            
        if "train" not in self.code_occurrence_overall or 'vectors' not in self.code_occurrence_overall["train"]:
            print("No training data available for pruning.")
            return
        
        print(f"Starting codebook pruning using '{method}' method...")
        original_size = len(self.codebook)
        print(f"Original codebook size: {original_size}")
        
        if method == "frequency":
            self._prune_codebook_frequency(**kwargs)
        elif method == "distinguishability":
            self._prune_codebook_distinguishability(**kwargs)
        elif method == "sklearn":
            self._prune_codebook_sklearn(**kwargs)
        elif method == "top_k":
            self._prune_codebook_top_k(**kwargs)
        else:
            print(f"Unknown pruning method: {method}. Using frequency method as fallback.")
            self._prune_codebook_frequency(**kwargs)
            
        new_size = len(self.codebook)
        print(f"Codebook size after pruning: {new_size} (removed {original_size - new_size} codes)")
    
        self.logistic_classifier = None
        self.knn_classifier = None
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False

    def _prune_codebook_frequency(self, min_occurrences=10, equality_threshold=0.05):
        """
        Frequency-based pruning similar to the original Coder class.
        
        Args:
            min_occurrences (int): Minimum total occurrences to consider for pruning
            equality_threshold (float): If the ratio difference from 50% is within this threshold,
                                      consider the feature as non-discriminative
        """
        train_data = self.code_occurrence_overall["train"]
        
        # Calculate occurrence statistics for each code
        code_stats = self._calculate_code_occurrence_stats(train_data)
        
        codes_to_remove = []
        
        for code_name in list(self.codebook.keys()):
            if code_name not in code_stats:
                print(f"Code '{code_name}' has no occurrence data, skipping.")
                continue
                
            stats = code_stats[code_name]
            total_occurrences = stats['total_occurrences']
            model0_ratio = stats['model0_ratio']
            
            # Check if code occurs enough times to be considered
            if total_occurrences < min_occurrences:
                continue
            
            # Check if occurrences are roughly equal between models (45%-55% range from original)
            lower_bound = 0.5 - equality_threshold
            upper_bound = 0.5 + equality_threshold
            
            if lower_bound < model0_ratio < upper_bound:
                codes_to_remove.append(code_name)
                print(f"Marking '{code_name}' for removal - {self.model_options[0]}: {stats['model0_count']}, "
                      f"{self.model_options[1]}: {stats['model1_count']} (ratio: {model0_ratio:.3f})")
        
        # Remove codes from codebook and update data structures
        if codes_to_remove:
            self._remove_codes_from_structures(codes_to_remove)
        else:
            print("No codes met the frequency-based pruning criteria.")
    
    def _prune_codebook_sklearn(self, selector_type="k_best", k=None, score_func="mutual_info", percentile=20, **kwargs):
        """
        Sklearn-based feature selection for pruning.
        
        Args:
            selector_type (str): Type of selector ("k_best", "percentile", "variance")
            k (int): Number of features to keep for k_best
            score_func (str): Scoring function ("mutual_info", "chi2", "f_classif")
            percentile (int): Percentile for percentile selector
        """
        from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold
        from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
        
        train_data = self.code_occurrence_overall["train"]
        
        # Prepare data for sklearn
        if not train_data['vectors'] or not train_data['labels']:
            print("Insufficient training data for sklearn-based pruning.")
            return
        
        # Get vectors and labels
        X = np.array([train_data['vectors'][key] for key in train_data['vectors'].keys()])
        y = train_data['labels']
        
        # Apply normalization to training vectors if configured
        if self._normalize_vectors:
            if self._normalization_method == "comparative":
                X = self._apply_comparative_normalization_to_training_data(train_data)
            else:
                normalized_vectors = []
                for vector in X:
                    normalized_vectors.append(self._normalize_vector(vector))
                X = np.array(normalized_vectors)
        
        # Convert string labels to integers for sklearn
        unique_labels = list(set(y))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_encoded = [label_to_int[label] for label in y]
        
        print(f"Data shape for feature selection: {X.shape}")
        print(f"Labels distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Choose selector
        if selector_type == "k_best":
            if k is None:
                # k = min(self.max_rule, len(train_data['code_order']) - int(self.max_rule * 0.20))
                k = self.max_rule
            k = min(k, len(train_data['code_order']))  # Ensure k doesn't exceed feature count
            
            # Choose score function
            if score_func == "mutual_info":
                score_function = mutual_info_classif
            elif score_func == "chi2":
                # Ensure non-negative features for chi2
                X = np.abs(X)
                score_function = chi2
            elif score_func == "f_classif":
                score_function = f_classif
            else:
                print(f"Unknown score function: {score_func}. Using mutual_info.")
                score_function = mutual_info_classif
            
            selector = SelectKBest(score_func=score_function, k=k)
            print(f"Using SelectKBest with k={k} and score_func={score_func}")
            
        elif selector_type == "percentile":
            if score_func == "mutual_info":
                score_function = mutual_info_classif
            elif score_func == "chi2":
                X = np.abs(X)
                score_function = chi2
            elif score_func == "f_classif":
                score_function = f_classif
            else:
                score_function = mutual_info_classif
                
            selector = SelectPercentile(score_func=score_function, percentile=percentile)
            print(f"Using SelectPercentile with percentile={percentile} and score_func={score_func}")
            
        elif selector_type == "variance":
            threshold = kwargs.get("variance_threshold", 0.0)
            selector = VarianceThreshold(threshold=threshold)
            print(f"Using VarianceThreshold with threshold={threshold}")
            
        else:
            print(f"Unknown selector type: {selector_type}. Using SelectKBest.")
            # k = k or min(self.max_rule, len(train_data['code_order']) - int(self.max_rule * 0.20))
            k = self.max_rule
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        try:
            # Fit the selector
            X_selected = selector.fit_transform(X, y_encoded)
            
            # Get selected feature indices
            selected_features = selector.get_support(indices=True)
            
            print(f"Selected {len(selected_features)} features out of {len(train_data['code_order'])}")
            
            # Get codes to remove (features not selected)
            codes_to_remove = []
            for i, code_name in enumerate(train_data['code_order']):
                if i not in selected_features:
                    codes_to_remove.append(code_name)
            
            print(f"Removing {len(codes_to_remove)} codes: {codes_to_remove[:10]}{'...' if len(codes_to_remove) > 10 else ''}")
            
            # Remove codes from codebook and update data structures
            if codes_to_remove:
                self._remove_codes_from_structures(codes_to_remove)
            else:
                print("No codes were selected for removal by sklearn feature selection.")
                
        except Exception as e:
            print(f"Error during sklearn feature selection: {e}")
            print("Falling back to frequency-based pruning.")
            self._prune_codebook_frequency()
    
    def _prune_codebook_top_k(self, k=None, score_func="mutual_info"):
        """
        Keep only the top k most discriminative codes based on feature scoring.
        
        Args:
            k (int): Number of codes to keep
            score_func (str): Scoring function ("mutual_info", "chi2", "f_classif")
        """
        from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
        
        train_data = self.code_occurrence_overall["train"]

        if k is None:
            # k = min(self.max_rule, len(train_data['code_order']) - int(self.max_rule * 0.20))
            k = self.max_rule
        k = min(k, len(train_data['code_order']))  # Ensure k doesn't exceed feature count
        
        # Prepare data for scoring
        if not train_data['vectors'] or not train_data['labels']:
            print("Insufficient training data for top_k pruning.")
            return
        
        # Get vectors and labels
        X = np.array([train_data['vectors'][key] for key in train_data['vectors'].keys()])
        y = train_data['labels']
        
        # Apply normalization to training vectors if configured
        if self._normalize_vectors:
            if self._normalization_method == "comparative":
                X = self._apply_comparative_normalization_to_training_data(train_data)
            else:
                normalized_vectors = []
                for vector in X:
                    normalized_vectors.append(self._normalize_vector(vector))
                X = np.array(normalized_vectors)
        
        # Convert string labels to integers for sklearn
        unique_labels = list(set(y))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_encoded = [label_to_int[label] for label in y]
        
        current_size = len(train_data['code_order'])
        k = min(k, current_size)  # Ensure k doesn't exceed current size
        
        if k >= current_size:
            print(f"Requested k={k} is >= current codebook size ({current_size}). No pruning needed.")
            return
        
        print(f"Ranking codes by discriminative power and keeping top {k} out of {current_size}")
        
        # Choose score function
        if score_func == "mutual_info":
            score_function = mutual_info_classif
        elif score_func == "chi2":
            # Ensure non-negative features for chi2
            X = np.abs(X)
            score_function = chi2
        elif score_func == "f_classif":
            score_function = f_classif
        else:
            print(f"Unknown score function: {score_func}. Using mutual_info.")
            score_function = mutual_info_classif
        
        try:
            # Calculate scores for all features
            scores = score_function(X, y_encoded)
            
            # Handle potential NaN values
            scores = np.nan_to_num(scores, nan=0.0)
            
            # Get indices of top k features
            top_k_indices = np.argsort(scores)[-k:]  # Get indices of k highest scores
            
            # Get codes to remove (features not in top k)
            codes_to_remove = []
            for i, code_name in enumerate(train_data['code_order']):
                if i not in top_k_indices:
                    codes_to_remove.append(code_name)
            
            print(f"Keeping top {k} codes, removing {len(codes_to_remove)} codes")
            print(f"Top {min(5, k)} codes: {[train_data['code_order'][i] for i in sorted(top_k_indices)[-5:]]}")
            
            # Remove codes from codebook and update data structures
            if codes_to_remove:
                self._remove_codes_from_structures(codes_to_remove)
            else:
                print("No codes were selected for removal.")
                
        except Exception as e:
            print(f"Error during top_k feature ranking: {e}")
            print("Falling back to frequency-based pruning.")
            self._prune_codebook_frequency()
    
    def _prune_codebook_distinguishability(self, equality_threshold=0.05, min_occurrences=10):
        """
        Prune codes that cannot distinguish between models better than chance.
        
        Args:
            normalize (bool): Whether to normalize vectors before calculating statistics
            equality_threshold (float): Threshold for determining if a code distinguishes better than chance
            min_occurrences (int): Minimum total occurrences to consider for pruning
        """
        train_data = self.code_occurrence_overall["train"]
        
        # Create a copy of the training data if normalization is needed
        if self._normalize_vectors and train_data['vectors']:
            normalized_train_data = {
                'vectors': {},
                'labels': train_data['labels'].copy(),
                'code_order': train_data['code_order'].copy()
            }
            
            # Get vectors as array for normalization
            X = np.array([train_data['vectors'][key] for key in train_data['vectors'].keys()])
            vector_keys = list(train_data['vectors'].keys())
            
            # Apply normalization based on the configured method
            if self._normalization_method == "comparative":
                X_normalized = self._apply_comparative_normalization_to_training_data(train_data)
            else:
                normalized_vectors = []
                for vector in X:
                    normalized_vectors.append(self._normalize_vector(vector))
                X_normalized = np.array(normalized_vectors)
            
            # Update the training data with normalized vectors
            for i, key in enumerate(vector_keys):
                normalized_train_data['vectors'][key] = X_normalized[i]
            
            data_to_analyze = normalized_train_data
        else:
            data_to_analyze = train_data
        
        # Group vectors by sample_id to compare model pairs
        vector_keys = list(data_to_analyze['vectors'].keys())
        sample_groups = {}
        
        for key in vector_keys:
            # Extract sample ID (everything before the last underscore)
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                sample_id = parts[0]
                model_suffix = parts[1]
                
                if model_suffix in ["OUTPUT A", "OUTPUT B", "A", "B"]:
                    continue
                
                if sample_id not in sample_groups:
                    sample_groups[sample_id] = {}

                sample_groups[sample_id][model_suffix] = data_to_analyze['vectors'][key]
        
        # Calculate classification accuracy for each code dimension
        codes_to_remove = []
        
        for i, code_name in enumerate(data_to_analyze['code_order']):
            if code_name not in self.codebook:
                continue
            
            # Count comparisons for this dimension
            model_a_wins = 0  # model A's value > model B's value
            model_b_wins = 0  # model A's value < model B's value
            ties = 0
            total_comparisons = 0
            
            for sample_id, group in sample_groups.items():
                if len(group) == 2:  # We have a pair from both models
                    models = list(group.keys())
                    if len(models) == 2:
                        model_a_key, model_b_key = models[0], models[1]
                        vector_a = np.array(group[model_a_key])
                        vector_b = np.array(group[model_b_key])
                        
                        value_a = vector_a[i]
                        value_b = vector_b[i]
                        
                        if value_a > value_b:
                            model_a_wins += 1
                        elif value_a < value_b:
                            model_b_wins += 1
                        else:
                            ties += 1
                        
                        total_comparisons += 1
            
            if total_comparisons < min_occurrences:
                continue

            # Calculate the ratio of model A wins
            if total_comparisons > 0:
                model_a_win_ratio = model_a_wins / total_comparisons
            else:
                continue
            
            # Check if the code can distinguish better than chance
            chance_level = 0.5
            lower_bound = chance_level - equality_threshold
            upper_bound = chance_level + equality_threshold
            
            if lower_bound < model_a_win_ratio < upper_bound:
                codes_to_remove.append(code_name)
                print(f"Marking '{code_name}' for removal - cannot distinguish better than chance: "
                      f"Model A wins: {model_a_wins}/{total_comparisons}, "
                      f"Model B wins: {model_b_wins}/{total_comparisons} "
                      f"(A win ratio: {model_a_win_ratio:.3f}, ties: {ties})")
        
        # Remove codes from codebook and update data structures
        if codes_to_remove:
            self._remove_codes_from_structures(codes_to_remove)
        else:
            print("No codes met the distinguishability-based pruning criteria.")

    def _calculate_code_occurrence_stats(self, train_data):
        """
        Calculate occurrence statistics for each code from the vector data.
        
        Args:
            train_data (dict): Training data containing vectors and labels
            
        Returns:
            dict: Statistics for each code
        """
        stats = {}
        
        if not train_data['vectors'] or not train_data['labels']:
            return stats
        
        # Get vectors and labels
        vectors = np.array([train_data['vectors'][key] for key in train_data['vectors'].keys()])
        labels = train_data['labels']
        
        # Calculate statistics for each code (feature dimension)
        for i, code_name in enumerate(train_data['code_order']):
            feature_values = vectors[:, i]
            
            # Calculate total occurrences for each model
            model0_count = 0
            model1_count = 0
            
            for j, label in enumerate(labels):
                if label == self.model_options[0]:
                    model0_count += feature_values[j]
                elif label == self.model_options[1]:
                    model1_count += feature_values[j]
            
            total_count = model0_count + model1_count
            model0_ratio = model0_count / total_count if total_count > 0 else 0.5
            
            stats[code_name] = {
                'model0_count': model0_count,
                'model1_count': model1_count,
                'total_occurrences': total_count,
                'model0_ratio': model0_ratio
            }
        
        return stats
    
    def _remove_codes_from_structures(self, codes_to_remove):
        """
        Remove codes from all relevant data structures after pruning.
        
        Args:
            codes_to_remove (list): List of code names to remove
        """
        print(f"Removing {len(codes_to_remove)} codes from all data structures...")
        
        # 1. Remove from codebook
        for code_name in codes_to_remove:
            if code_name in self.codebook:
                del self.codebook[code_name]
                print(f"Removed '{code_name}' from codebook")
        
        # 2. Update all splits in code_occurrence_overall
        for split_name, split_data in self.code_occurrence_overall.items():
            if 'vectors' in split_data and 'code_order' in split_data:
                self._update_vectors_after_pruning(split_data, codes_to_remove)
        
        # 3. Update code usage counts
        if hasattr(self, '_code_usage_counts'):
            for split_name in self._code_usage_counts:
                for code_name in codes_to_remove:
                    if code_name in self._code_usage_counts[split_name]:
                        del self._code_usage_counts[split_name][code_name]
        
        # 4. Reset classifiers since feature dimensions have changed
        self.logistic_classifier = None
        self.knn_classifier = None
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
        print(f"Successfully updated all data structures after pruning")
    
    def _update_vectors_after_pruning(self, split_data, codes_to_remove):
        """
        Update vector dimensions and code_order after removing codes.
        
        Args:
            split_data (dict): Data for a specific split
            codes_to_remove (list): List of code names to remove
        """
        if 'code_order' not in split_data or 'vectors' not in split_data:
            return
        
        # Get indices of codes to remove
        indices_to_remove = []
        for code_name in codes_to_remove:
            if code_name in split_data['code_order']:
                indices_to_remove.append(split_data['code_order'].index(code_name))
        
        if not indices_to_remove:
            return
        
        indices_to_remove = sorted(indices_to_remove, reverse=True)  # Remove from end to preserve indices
        
        # Update code_order
        for code_name in codes_to_remove:
            if code_name in split_data['code_order']:
                split_data['code_order'].remove(code_name)
        
        # Update all vectors by removing corresponding dimensions
        for key, vector in split_data['vectors'].items():
            if isinstance(vector, np.ndarray):
                # Remove dimensions corresponding to pruned codes
                mask = np.ones(len(vector), dtype=bool)
                for idx in indices_to_remove:
                    if idx < len(mask):
                        mask[idx] = False
                split_data['vectors'][key] = vector[mask]
        
        print(f"Updated vectors for split: new dimension = {len(split_data['code_order'])}")
    
    def save(self, filename="coder_bor_attr.ckpt", exclude_attributes=None):
        """
        Override save method to include reasoning code representation and imputation settings.
        """
        if exclude_attributes is None:
            exclude_attributes = ["model", "tokenizer", "logistic_classifier", "knn_classifier", "scaler"]
        
        # Save with parent method but include our new attributes
        super().save(filename, exclude_attributes)
    
    def get_reasoning_vector_summary(self, split="train"):
        """
        Get summary statistics about the reasoning vectors.
        
        Args:
            split (str): The split to summarize
            
        Returns:
            dict: Summary statistics
        """
        if split not in self.code_occurrence_overall:
            return {"error": f"Split '{split}' not found"}
        
        data = self.code_occurrence_overall[split]
        
        if 'vectors' not in data:
            return {"error": "No vector data found"}
        
        # Ensure all vectors have the same length (fix for codebook growth)
        expected_length = len(data['code_order'])
        for key, vec in data['vectors'].items():
            if len(vec) != expected_length:
                extended_vector = np.zeros(expected_length)
                extended_vector[:len(vec)] = vec
                data['vectors'][key] = extended_vector
        
        vectors = np.array([data['vectors'][key] for key in data['sample_ids']])
        labels = data['labels']
        
        summary = {
            "num_samples": len(vectors),
            "num_features": len(data['code_order']),
            "code_order": data['code_order'],
            "model_distribution": {model: labels.count(model) for model in self.model_options},
            "vector_stats": {
                "mean": np.mean(vectors, axis=0).tolist(),
                "std": np.std(vectors, axis=0).tolist(),
                "max": np.max(vectors, axis=0).tolist(),
                "min": np.min(vectors, axis=0).tolist()
            }
        }
        
        return summary
    
    def configure_imputation(self, extend_with_nan=False, imputation_method="mean", 
                           impute_after_accumulation=True, imputation_constant_value=0,
                           knn_imputation_neighbors=5, iterative_imputation_max_iter=10):
        """
        Configure missing value imputation settings.
        
        Args:
            extend_with_nan (bool): Whether to extend vectors with np.nan instead of 0
            imputation_method (str): Imputation method to use. Options:
                - "mean": Use mean of observed values
                - "median": Use median of observed values  
                - "most_frequent": Use most frequent value
                - "constant": Use a constant value
                - "knn": Use K-Nearest Neighbors imputation
                - "iterative": Use iterative imputation (multivariate)
            impute_after_accumulation (bool): Whether to impute after accumulation phases
            imputation_constant_value (float): Value to use for constant imputation
            knn_imputation_neighbors (int): Number of neighbors for KNN imputation
            iterative_imputation_max_iter (int): Max iterations for iterative imputation
        
        Example:
            # Use KNN imputation with NaN extension
            coder.configure_imputation(
                extend_with_nan=True,
                imputation_method="knn",
                knn_imputation_neighbors=3
            )
            
            # Use iterative imputation with more iterations
            coder.configure_imputation(
                extend_with_nan=True,
                imputation_method="iterative",
                iterative_imputation_max_iter=20
            )
        """
        self._extend_with_nan = extend_with_nan
        self._imputation_method = imputation_method
        self._impute_after_accumulation = impute_after_accumulation
        self._imputation_constant_value = imputation_constant_value
        self._knn_imputation_neighbors = knn_imputation_neighbors
        self._iterative_imputation_max_iter = iterative_imputation_max_iter
        
        print(f"Imputation configuration updated:")
        print(f"  - Extend with NaN: {extend_with_nan}")
        print(f"  - Imputation method: {imputation_method}")
        print(f"  - Impute after accumulation: {impute_after_accumulation}")
        if imputation_method == "constant":
            print(f"  - Constant value: {imputation_constant_value}")
        elif imputation_method == "knn":
            print(f"  - KNN neighbors: {knn_imputation_neighbors}")
        elif imputation_method == "iterative":
            print(f"  - Iterative max iterations: {iterative_imputation_max_iter}")
    
    def get_imputation_config(self):
        """
        Get current imputation configuration.
        
        Returns:
            dict: Current imputation settings
        """
        return {
            "extend_with_nan": self._extend_with_nan,
            "imputation_method": self._imputation_method,
            "impute_after_accumulation": self._impute_after_accumulation,
            "imputation_constant_value": self._imputation_constant_value,
            "knn_imputation_neighbors": self._knn_imputation_neighbors,
            "iterative_imputation_max_iter": self._iterative_imputation_max_iter
         }
    
    def get_imputation_summary(self, split="train"):
        """
        Get a summary of how imputation was applied to the vectors.
        
        Args:
            split (str): The split to analyze
            
        Returns:
            dict: Summary of imputation results by model
        """
        if split not in self.code_occurrence_overall:
            return {"error": f"Split '{split}' not found"}
        
        data = self.code_occurrence_overall[split]
        
        if 'vectors' not in data or not data['vectors']:
            return {"error": "No vector data found"}
        
        # Get vectors and labels
        vectors = np.array([data['vectors'][key] for key in data['sample_ids']])
        labels = data['labels']
        
        # Analyze by model
        unique_models = list(set(labels))
        summary = {
            "total_vectors": len(vectors),
            "total_features": len(data['code_order']),
            "imputation_config": self.get_imputation_config(),
            "models": {}
        }
        
        for model in unique_models:
            model_indices = [i for i, label in enumerate(labels) if label == model]
            model_vectors = vectors[model_indices]
            
            # Count missing values
            total_values = model_vectors.size
            missing_values = np.sum(np.isnan(model_vectors))
            
            summary["models"][model] = {
                "num_vectors": len(model_vectors),
                "total_values": total_values,
                "missing_values": missing_values,
                "missing_percentage": (missing_values / total_values * 100) if total_values > 0 else 0,
                "imputation_applied": self._extend_with_nan and missing_values > 0
            }
        
        return summary
    
    def _process_pending_accumulation_samples(self, pending_samples, custom_logger, logger_kwargs):
        """
        Process pending accumulation samples: update vectors, impute, then evaluate.
        
        Args:
            pending_samples (list): List of pending sample data with classification results
            custom_logger: Custom logging function
            logger_kwargs: Logger keyword arguments
        """
        if not pending_samples:
            return
        
        print(f"Processing {len(pending_samples)} pending samples...")
        
        # Step 1: Update all code occurrences (vectors) from pending samples
        print("Step 1: Updating code occurrences (vectors)...")
        for pending_sample in pending_samples:
            sample = pending_sample['sample']
            final_output = pending_sample['final_output']
            
            # Parse codes from the classification output
            code_occurrence, decision_code = self._parse_classification_codes(final_output, sample["labels"])
            
            # Update code occurrences in the training data (accumulation mode - manual update)
            self._update_code_occurence(self.code_occurrence_overall["train"], code_occurrence, is_decision_code=False, sample_id=sample["id"])
            self._update_code_occurence(self.decision_code_occurence_overall["train"], decision_code, is_decision_code=True)
        
        # Step 2: Do imputation to fill in np.nan values if enabled
        if self._extend_with_nan and self._impute_after_accumulation:
            print("Step 2: Performing imputation...")
            # print(self.code_occurrence_overall["train"]['vectors'])
            self._impute_missing_values(self.code_occurrence_overall["train"])
            # print(self.code_occurrence_overall["train"]['vectors'])
        self._most_recent_code_occurrence_overall_train = None
        
        # Step 3: Now evaluate each sample properly using the updated vectors
        print("Step 3: Evaluating samples...")
        for pending_sample in pending_samples:
            sample = pending_sample['sample']
            classification_prompt = pending_sample['classification_prompt']
            classification = pending_sample['classification']
            final_output = pending_sample['final_output']
            if_codebook_updated = pending_sample['if_codebook_updated']
            
            # Now do the proper evaluation using the classification output
            evaluation, pred = self._evaluate_classification(final_output, sample["labels"])
            
            # Update statistics with the correct evaluation
            self._update_sample_statistics_direct(
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
    
    def _update_sample_statistics_direct(self, sample, classification_prompt, classification, evaluation, if_codebook_updated, final_output, custom_logger, logger_kwargs):
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
        # Parse codes for logging purposes (already updated in vectors)
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
        if evaluation == "correct":
            self.train_correct_ids.append(sample["id"])
            self.num_train_correct += 1
        else:
            self.train_wrong_ids.append(sample["id"])
        
        self.num_train_samples += 1
        
        if custom_logger:
            custom_logger(**logger_kwargs)
        
        self.training_logs["train"][sample['id']]["current_train_acc"] = (self.num_train_correct / self.num_train_samples) * 100

    def _normalize_vector(self, vector):
        """
        Normalize a vector according to the configured normalization method.
        Note: For comparative normalization, use _normalize_vector_pair instead.
        
        Args:
            vector (np.array): The vector to normalize
            
        Returns:
            np.array: The normalized vector
        """
        if not self._normalize_vectors or self._normalization_method == "none":
            return vector
        
        if self._normalization_method == "comparative":
            # Comparative normalization requires pairs, return as-is and handle in classification methods
            return vector
        
        vector = np.array(vector, dtype=float)
        
        if self._normalization_method == "l1":
            # L1 normalization: divide by sum of absolute values
            sum_abs = np.sum(np.abs(vector))
            if sum_abs > 0:
                return vector / sum_abs
            else:
                return vector
        elif self._normalization_method == "l2":
            # L2 normalization: divide by Euclidean norm (unit vector)
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm
            else:
                return vector
        else:
            print(f"Unknown normalization method: {self._normalization_method}, returning original vector")
            return vector
    
    def _normalize_vector_pair(self, vector_a, vector_b):
        """
        Perform comparative normalization on a pair of vectors.
        Normalizes each dimension by the sum of values from both vectors on that dimension.
        
        Args:
            vector_a (np.array): First vector of the pair
            vector_b (np.array): Second vector of the pair
            
        Returns:
            tuple: (normalized_vector_a, normalized_vector_b)
        """
        vector_a = np.array(vector_a, dtype=float)
        vector_b = np.array(vector_b, dtype=float)
        
        # Calculate sum for each dimension
        dimension_sums = vector_a + vector_b
        
        # Normalize each vector by the dimension sums
        normalized_a = np.zeros_like(vector_a)
        normalized_b = np.zeros_like(vector_b)
        
        for i in range(len(vector_a)):
            if dimension_sums[i] > 0:
                normalized_a[i] = vector_a[i] / dimension_sums[i]
                normalized_b[i] = vector_b[i] / dimension_sums[i]
            else:
                # If sum is 0, both values are 0, keep them as 0
                normalized_a[i] = 0.0
                normalized_b[i] = 0.0
        
        return normalized_a, normalized_b
    
    def _apply_comparative_normalization_to_training_data(self, train_data):
        """
        Apply comparative normalization to training data by finding pairs from same sample.
        
        Args:
            train_data (dict): Training data containing vectors
            
        Returns:
            np.array: Normalized training vectors
        """
        vector_keys = list(train_data['vectors'].keys())
        vectors = np.array([train_data['vectors'][key] for key in vector_keys])
        
        # Group vectors by sample ID (extract from keys like "sample_0_ModelA")
        sample_groups = {}
        for i, key in enumerate(vector_keys):
            # Extract sample ID (everything before the last underscore)
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                sample_id = parts[0]
                model_suffix = parts[1]
                
                if sample_id not in sample_groups:
                    sample_groups[sample_id] = {}
                sample_groups[sample_id][model_suffix] = {'index': i, 'vector': vectors[i]}
        
        # Apply comparative normalization to each pair
        normalized_vectors = vectors.copy()
        for sample_id, group in sample_groups.items():
            if len(group) == 2:  # We have a pair
                keys = list(group.keys())
                key_a, key_b = keys[0], keys[1]
                
                vector_a = group[key_a]['vector']
                vector_b = group[key_b]['vector']
                idx_a = group[key_a]['index']
                idx_b = group[key_b]['index']
                
                norm_a, norm_b = self._normalize_vector_pair(vector_a, vector_b)
                normalized_vectors[idx_a] = norm_a
                normalized_vectors[idx_b] = norm_b
        
        return normalized_vectors
    
    def rerun_annotations(self, dataset, result_dict, num_reruns=5, return_averaged=True):
        """
        Rerun annotations N times for each sample in the dataset and get code vector representations.
        
        Args:
            dataset: Input dataset with samples containing 'outputs', 'labels', 'question', and 'id' keys
            result_dict: Dictionary with structure similar to code_occurrence_overall['train']
                        Must contain 'code_order' key to maintain dimension correspondence
            num_reruns (int): Number of times to rerun annotations for each sample
            return_averaged (bool): If True, return averaged vectors. If False, return all vectors.
            
        Returns:
            dict: Dictionary with structure similar to result_dict but with rerun vectors
                 Contains 'vectors', 'code_order', 'sample_ids', 'labels' keys
        """
        if 'code_order' not in result_dict:
            raise ValueError("result_dict must contain 'code_order' key")
        
        target_code_order = result_dict['code_order']
        
        # Initialize result structure
        rerun_result = {
            'vectors': {},
            'code_order': target_code_order[:],  # Copy the code order
            'sample_ids': [],
            'labels': []
        }
        
        print(f"Rerunning annotations {num_reruns} times for {len(dataset)} samples...")
        
        # Create batched inputs for all samples and all reruns
        batch_outputs = []
        batch_labels = []
        batch_questions = []
        batch_ids = []
        
        for sample in dataset:
            for rerun_idx in range(num_reruns):
                batch_outputs.append(sample["outputs"])
                batch_labels.append(sample["labels"])
                batch_questions.append(sample["question"])
                batch_ids.append(f"{sample['id']}_rerun_{rerun_idx}")
        
        # Single batch call for all samples and reruns
        print(f"Processing {len(batch_outputs)} total annotations in batch...")
        classification_prompts, classifications, evaluations, if_codebook_updated_list, final_outputs = self.classify(
            batch_outputs, batch_labels, batch_questions, 
            train=False, qid=batch_ids, batched=True, no_eval=True
        )
        
        # Group results back by sample and rerun
        batch_idx = 0
        for sample_idx, sample in enumerate(dataset):
            print(f"Processing sample {sample_idx + 1}/{len(dataset)}: {sample['id']}")
            
            # Store all vectors for this sample across reruns
            sample_vectors = []
            
            for rerun_idx in range(num_reruns):
                # Get results for this specific sample/rerun from the batch
                final_output = final_outputs[batch_idx]
                batch_idx += 1
                
                # Parse the code occurrence from the final output
                code_occurrence, _ = self._parse_classification_codes(final_output, sample["labels"])
                
                # Create vectors for this rerun using the target code order
                vector_a = np.zeros(len(target_code_order))
                vector_b = np.zeros(len(target_code_order))
                
                # Extract model names from code_occurrence
                model_names = []
                if code_occurrence:
                    first_code_data = next(iter(code_occurrence.values()))
                    if isinstance(first_code_data, dict):
                        model_names = list(first_code_data.keys())
                
                if len(model_names) >= 2:
                    model_a_name = model_names[0]
                    model_b_name = model_names[1]
                else:
                    print("No model names found for sample", sample['id'])
                    model_a_name = "OUTPUT A"
                    model_b_name = "OUTPUT B"
                
                # Fill in the counts using target code order
                for code_name, counts in code_occurrence.items():
                    if code_name in target_code_order:
                        code_idx = target_code_order.index(code_name)
                        vector_a[code_idx] = counts.get(model_a_name, 0)
                        vector_b[code_idx] = counts.get(model_b_name, 0)
                
                # Store vectors for this rerun
                sample_vectors.append({
                    'vector_a': vector_a,
                    'vector_b': vector_b,
                    'model_a_name': model_a_name,
                    'model_b_name': model_b_name
                })
            
            # Process the collected vectors for this sample
            if return_averaged:
                # Average the vectors across reruns
                avg_vector_a = np.mean([rv['vector_a'] for rv in sample_vectors], axis=0)
                avg_vector_b = np.mean([rv['vector_b'] for rv in sample_vectors], axis=0)
                
                # Use the model names from the first rerun
                model_a_name = sample_vectors[0]['model_a_name']
                model_b_name = sample_vectors[0]['model_b_name']
                
                # Store averaged vectors
                key_a = f"{sample['id']}_{model_a_name}"
                key_b = f"{sample['id']}_{model_b_name}"
                
                rerun_result['vectors'][key_a] = avg_vector_a
                rerun_result['vectors'][key_b] = avg_vector_b
                rerun_result['sample_ids'].extend([key_a, key_b])
                rerun_result['labels'].extend([model_a_name, model_b_name])
            else:
                # Store all vectors for each rerun
                for rerun_idx, rerun_vectors in enumerate(sample_vectors):
                    model_a_name = rerun_vectors['model_a_name']
                    model_b_name = rerun_vectors['model_b_name']
                    
                    key_a = f"{sample['id']}_rerun_{rerun_idx}_{model_a_name}"
                    key_b = f"{sample['id']}_rerun_{rerun_idx}_{model_b_name}"
                    
                    rerun_result['vectors'][key_a] = rerun_vectors['vector_a']
                    rerun_result['vectors'][key_b] = rerun_vectors['vector_b']
                    rerun_result['sample_ids'].extend([key_a, key_b])
                    rerun_result['labels'].extend([model_a_name, model_b_name])
        
        print(f"Completed rerunning annotations. Generated {len(rerun_result['vectors'])} vectors.")
        return rerun_result

    def reject_inconsistent_codes(self, dataset, num_reruns=5, removal_criteria="percentile", 
                                 percentile_threshold=80, variance_threshold=2.0, 
                                 return_averaged=True, modify_training_data=False):
        """
        Remove reasoning behaviors (codes) that show high variance across multiple annotation runs.
        
        Args:
            dataset: Training dataset or subset to analyze
            num_reruns (int): Number of annotation reruns to calculate variance
            removal_criteria (str): "percentile" or "threshold" for determining which codes to remove
            percentile_threshold (float): Remove codes above this percentile of variance (0-100)
            variance_threshold (float): Remove codes with variance above this threshold
            return_averaged (bool): If True, return averaged vectors. If False, return all vectors.
            modify_training_data (bool): If True, directly modify code_occurrence_overall["train"] and codebook.
                                       If False, return filtered copies without modifying original data.
            
        Returns:
            tuple: (filtered_code_occurrence_dict, filtered_codebook)
                   If modify_training_data=True, returns the updated training data and codebook.
                   If modify_training_data=False, returns new filtered copies.
        """
        if not self.codebook:
            print("Codebook is empty. No codes to filter.")
            return self.code_occurrence_overall.get("train", {}), self.codebook
        
        print(f"Analyzing annotation consistency across {num_reruns} reruns for {len(dataset)} samples...")
        
        # Get training data structure to maintain code order
        train_data = self.code_occurrence_overall.get("train", {})
        if 'code_order' not in train_data:
            print("No training data available. Cannot perform consistency analysis.")
            return train_data, self.codebook
        
        # Rerun annotations to get multiple vectors per sample
        rerun_result = self.rerun_annotations(dataset, train_data, num_reruns, return_averaged=False)
        
        # Calculate variance for each dimension (code) within each sample
        print("Calculating variance for each reasoning behavior...")
        sample_variances = {}  # sample_id -> variance array
        
        # Group vectors by sample
        sample_groups = {}
        for vector_key in rerun_result['sample_ids']:
            # Extract sample ID (everything before _rerun_)
            if '_rerun_' in vector_key:
                sample_id = vector_key.split('_rerun_')[0].rsplit('_', 1)[0]  # Remove model suffix too
            else:
                sample_id = vector_key.rsplit('_', 1)[0]
            
            if sample_id not in sample_groups:
                sample_groups[sample_id] = []
            sample_groups[sample_id].append(vector_key)
        
        # Calculate normalized variance within each sample across reruns
        for sample_id, vector_keys in sample_groups.items():
            vectors = [rerun_result['vectors'][key] for key in vector_keys]
            if len(vectors) >= 2:  # Need at least 2 vectors to calculate variance
                vectors_array = np.array(vectors)
                sample_means = np.mean(vectors_array, axis=0)
                sample_vars = np.var(vectors_array, axis=0)
                
                # Calculate coefficient of variation (CV = std/mean) for relative variance
                # Add small epsilon to avoid division by zero
                epsilon = 1e-8
                sample_cv = np.sqrt(sample_vars) / (sample_means + epsilon)
                
                # For dimensions where mean is very close to 0, use raw variance
                # This handles cases where a behavior rarely occurs
                normalized_variance = np.where(sample_means > 0.1, sample_cv, sample_vars)
                sample_variances[sample_id] = normalized_variance
        
        if not sample_variances:
            print("Insufficient data for variance calculation.")
            return train_data, self.codebook
        
        # Calculate average variance across samples for each dimension
        all_variances = list(sample_variances.values())
        avg_variances = np.mean(all_variances, axis=0)
        
        print(f"Average variance per dimension: min={np.min(avg_variances):.3f}, "
              f"max={np.max(avg_variances):.3f}, mean={np.mean(avg_variances):.3f}")
        
        # Determine which codes to remove based on criteria
        codes_to_remove = []
        if removal_criteria == "percentile":
            threshold_value = np.percentile(avg_variances, percentile_threshold)
            high_variance_indices = np.where(avg_variances >= threshold_value)[0]
            print(f"Using percentile criteria: removing codes with variance >= {threshold_value:.3f}")
        elif removal_criteria == "threshold":
            high_variance_indices = np.where(avg_variances >= variance_threshold)[0]
            print(f"Using threshold criteria: removing codes with variance >= {variance_threshold}")
        else:
            print(f"Unknown removal criteria: {removal_criteria}. Using percentile.")
            threshold_value = np.percentile(avg_variances, percentile_threshold)
            high_variance_indices = np.where(avg_variances >= threshold_value)[0]
        
        # Get code names to remove
        for idx in high_variance_indices:
            if idx < len(train_data['code_order']):
                codes_to_remove.append(train_data['code_order'][idx])
        
        print(f"Removing {len(codes_to_remove)} inconsistent codes: {codes_to_remove[:5]}{'...' if len(codes_to_remove) > 5 else ''}")
        
        if modify_training_data:
            # Directly modify the existing training data and codebook
            if codes_to_remove:
                print("Directly modifying training data and codebook...")
                self._remove_codes_from_structures(codes_to_remove)
                
                # Reset classifiers since feature dimensions have changed
                self.logistic_classifier = None
                self.knn_classifier = None
                self.scaler = StandardScaler()
                self.is_scaler_fitted = False
                
                print(f"Training data updated. Retained {len(self.codebook)} out of {len(self.codebook) + len(codes_to_remove)} codes.")
            else:
                print("No codes to remove. Training data unchanged.")
            
            # Return the updated training data and codebook
            return self.code_occurrence_overall["train"], self.codebook
        else:
            # Create filtered copies without modifying original data
            # Create filtered codebook
            filtered_codebook = {code: definition for code, definition in self.codebook.items() 
                               if code not in codes_to_remove}
            
            # Create filtered code_occurrence_dict
            if return_averaged:
                # Return averaged vectors for each sample
                filtered_result = self.rerun_annotations(dataset, train_data, num_reruns, return_averaged=True)
            else:
                # Use the existing rerun result
                filtered_result = rerun_result
            
            # Remove inconsistent dimensions from all vectors
            if codes_to_remove:
                self._remove_codes_from_filtered_result(filtered_result, codes_to_remove, train_data['code_order'])
            
            print(f"Filtering complete. Retained {len(filtered_codebook)} out of {len(self.codebook)} codes.")
            return filtered_result, filtered_codebook
    
    def _remove_codes_from_filtered_result(self, result_dict, codes_to_remove, original_code_order):
        """
        Remove specified codes from the filtered result dictionary.
        
        Args:
            result_dict (dict): The result dictionary to filter
            codes_to_remove (list): List of code names to remove
            original_code_order (list): Original code order for index mapping
        """
        # Get indices of codes to remove
        indices_to_remove = []
        for code_name in codes_to_remove:
            if code_name in original_code_order:
                indices_to_remove.append(original_code_order.index(code_name))
        
        if not indices_to_remove:
            return
        
        indices_to_remove = sorted(indices_to_remove, reverse=True)  # Remove from end to preserve indices
        
        # Update code_order
        for code_name in codes_to_remove:
            if code_name in result_dict['code_order']:
                result_dict['code_order'].remove(code_name)
        
        # Update all vectors by removing corresponding dimensions
        for key, vector in result_dict['vectors'].items():
            if isinstance(vector, np.ndarray):
                # Remove dimensions corresponding to inconsistent codes
                mask = np.ones(len(vector), dtype=bool)
                for idx in indices_to_remove:
                    if idx < len(mask):
                        mask[idx] = False
                result_dict['vectors'][key] = vector[mask]
        
        print(f"Filtered vectors: new dimension = {len(result_dict['code_order'])}")

