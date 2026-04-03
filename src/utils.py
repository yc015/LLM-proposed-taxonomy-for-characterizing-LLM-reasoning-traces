import re
from thefuzz import fuzz
import numpy as np


def clean_code_name(code_name):
    """
    Remove all formatting and special characters, keeping only letters.
    
    Parameters:
    - code_name (str): The code name to clean.
    
    Returns:
    - str: The cleaned code name.
    """
    return re.sub(r'[\*\#\[\]]', '', code_name)


def fuzz_match(string, strings, threshold=None, method="token"):
    fuzz_scores = []
    for string_2 in strings:
        # fuzz_scores.append(fuzz.ratio(string, string_2))
        if method == "token":
            fuzz_scores.append(fuzz.token_set_ratio(string, string_2))
        elif method == "character":
            fuzz_scores.append(fuzz.ratio(string, string_2))

    if threshold:
        if len(fuzz_scores) < 1:
            return False
    
        if max(fuzz_scores) >= threshold:
            return strings[np.argmax(fuzz_scores)]
        else:
            return False
    else:
        return fuzz_scores


def calculate_annotation_agreement(multi_run_result, agreement_type="fleiss_binary", 
                                 binary_threshold=0, exact_match_tolerance=0, 
                                 weight_function="linear"):
    """
    Calculate inter-rater agreement (kappa) for multi-run annotations.
    
    Args:
        multi_run_result (dict): Result from rerun_annotations with return_averaged=False
                                Contains 'vectors', 'sample_ids', 'labels', 'code_order'
        agreement_type (str): Type of agreement calculation
            - "fleiss_binary": Fleiss' kappa on binarized counts (present/absent)
            - "fleiss_exact": Fleiss' kappa on exact count matches
            - "fleiss_tolerance": Fleiss' kappa with tolerance for count differences
            - "cohen_pairwise": Average Cohen's kappa across all run pairs
            - "weighted_kappa": Weighted kappa for count differences
        binary_threshold (int): Threshold for binarization (count > threshold = present)
        exact_match_tolerance (int): Tolerance for "exact" matches (±tolerance)
        weight_function (str): Weighting function for weighted kappa ("linear", "quadratic")
        
    Returns:
        dict: Agreement statistics including kappa values, confidence intervals, etc.
    """
    from sklearn.metrics import cohen_kappa_score
    
    if 'vectors' not in multi_run_result or not multi_run_result['vectors']:
        return {"error": "No vector data found in multi_run_result"}
    
    # Group vectors by sample and run
    sample_run_data = {}  # sample_id -> run_idx -> [vector_a, vector_b]
    
    for vector_key in multi_run_result['sample_ids']:
        if '_rerun_' in vector_key:
            # Parse key: sample_id_rerun_X_model
            parts = vector_key.split('_rerun_')
            sample_id = parts[0].rsplit('_', 1)[0]  # Remove model suffix from first part
            run_and_model = parts[1]  # X_model
            run_idx = int(run_and_model.split('_')[0])
            
            if sample_id not in sample_run_data:
                sample_run_data[sample_id] = {}
            if run_idx not in sample_run_data[sample_id]:
                sample_run_data[sample_id][run_idx] = []
            
            vector = multi_run_result['vectors'][vector_key]
            sample_run_data[sample_id][run_idx].append(vector)
    
    # Extract run data for agreement calculation
    num_features = len(multi_run_result['code_order'])
    sample_ids = list(sample_run_data.keys())
    num_samples = len(sample_ids)
    
    # Determine number of runs
    max_runs = max(len(runs) for runs in sample_run_data.values())
    
    print(f"Calculating agreement for {num_samples} samples, {max_runs} runs, {num_features} features")
    
    if agreement_type == "fleiss_binary":
        return _calculate_fleiss_kappa_binary(sample_run_data, sample_ids, num_features, 
                                            max_runs, binary_threshold, multi_run_result['code_order'])
    
    elif agreement_type == "fleiss_exact":
        return _calculate_fleiss_kappa_exact(sample_run_data, sample_ids, num_features, 
                                           max_runs, multi_run_result['code_order'])
    
    elif agreement_type == "fleiss_tolerance":
        return _calculate_fleiss_kappa_tolerance(sample_run_data, sample_ids, num_features, 
                                               max_runs, exact_match_tolerance, multi_run_result['code_order'])
    
    elif agreement_type == "cohen_pairwise":
        return _calculate_cohen_pairwise(sample_run_data, sample_ids, num_features, 
                                       max_runs, binary_threshold, multi_run_result['code_order'])
    
    elif agreement_type == "weighted_kappa":
        return _calculate_weighted_kappa(sample_run_data, sample_ids, num_features, 
                                       max_runs, weight_function, multi_run_result['code_order'])
    
    else:
        return {"error": f"Unknown agreement_type: {agreement_type}"}


def _calculate_fleiss_kappa_binary(sample_run_data, sample_ids, num_features, max_runs, 
                                 binary_threshold, code_order):
    """Calculate Fleiss' kappa on binarized presence/absence data."""
    
    # For each feature, create a rating matrix: samples x raters (runs)
    feature_kappas = {}
    overall_agreements = []
    
    for feature_idx, code_name in enumerate(code_order):
        # Create rating matrix for this feature
        # Each cell contains 0 (absent) or 1 (present) for sample i, run j
        ratings = []
        
        for sample_id in sample_ids:
            sample_ratings = []
            runs = sample_run_data[sample_id]
            
            for run_idx in range(max_runs):
                if run_idx in runs and len(runs[run_idx]) >= 2:
                    # Average the two vectors (model A and B) for this run
                    avg_vector = np.mean(runs[run_idx], axis=0)
                    binary_value = 1 if avg_vector[feature_idx] > binary_threshold else 0
                    sample_ratings.append(binary_value)
                else:
                    sample_ratings.append(np.nan)  # Missing data
            
            # Only include samples with at least 2 valid ratings
            valid_ratings = [r for r in sample_ratings if not np.isnan(r)]
            if len(valid_ratings) >= 2:
                ratings.append(valid_ratings)
        
        if len(ratings) > 0:
            # Calculate Fleiss' kappa for this feature
            kappa = _fleiss_kappa(ratings)
            feature_kappas[code_name] = kappa
            overall_agreements.append(kappa)
        else:
            feature_kappas[code_name] = np.nan
    
    # Calculate overall statistics
    valid_kappas = [k for k in overall_agreements if not np.isnan(k)]
    
    return {
        "agreement_type": "fleiss_binary",
        "binary_threshold": binary_threshold,
        "overall_kappa": np.mean(valid_kappas) if valid_kappas else np.nan,
        "kappa_std": np.std(valid_kappas) if valid_kappas else np.nan,
        "feature_kappas": feature_kappas,
        "num_valid_features": len(valid_kappas),
        "kappa_distribution": {
            "min": np.min(valid_kappas) if valid_kappas else np.nan,
            "max": np.max(valid_kappas) if valid_kappas else np.nan,
            "q25": np.percentile(valid_kappas, 25) if valid_kappas else np.nan,
            "q75": np.percentile(valid_kappas, 75) if valid_kappas else np.nan
        }
    }


def _calculate_cohen_pairwise(sample_run_data, sample_ids, num_features, max_runs, 
                            binary_threshold, code_order):
    """Calculate average Cohen's kappa across all pairs of runs."""
    from sklearn.metrics import cohen_kappa_score
    from itertools import combinations
    
    pairwise_kappas = []
    feature_pairwise_kappas = {code_name: [] for code_name in code_order}
    
    # For each pair of runs
    for run_i, run_j in combinations(range(max_runs), 2):
        feature_agreements = []
        
        for feature_idx, code_name in enumerate(code_order):
            rater1_scores = []
            rater2_scores = []
            
            for sample_id in sample_ids:
                runs = sample_run_data[sample_id]
                
                if run_i in runs and run_j in runs:
                    if len(runs[run_i]) >= 2 and len(runs[run_j]) >= 2:
                        # Average vectors for each run
                        avg_vector_i = np.mean(runs[run_i], axis=0)
                        avg_vector_j = np.mean(runs[run_j], axis=0)
                        
                        score_i = 1 if avg_vector_i[feature_idx] > binary_threshold else 0
                        score_j = 1 if avg_vector_j[feature_idx] > binary_threshold else 0
                        
                        rater1_scores.append(score_i)
                        rater2_scores.append(score_j)
            
            if len(rater1_scores) >= 2:
                try:
                    kappa = cohen_kappa_score(rater1_scores, rater2_scores)
                    feature_agreements.append(kappa)
                    feature_pairwise_kappas[code_name].append(kappa)
                except:
                    feature_agreements.append(np.nan)
                    feature_pairwise_kappas[code_name].append(np.nan)
            else:
                feature_agreements.append(np.nan)
                feature_pairwise_kappas[code_name].append(np.nan)
        
        # Average across features for this pair
        valid_agreements = [a for a in feature_agreements if not np.isnan(a)]
        if valid_agreements:
            pairwise_kappas.append(np.mean(valid_agreements))
    
    # Calculate feature-level averages
    feature_avg_kappas = {}
    for code_name in code_order:
        valid_kappas = [k for k in feature_pairwise_kappas[code_name] if not np.isnan(k)]
        feature_avg_kappas[code_name] = np.mean(valid_kappas) if valid_kappas else np.nan
    
    return {
        "agreement_type": "cohen_pairwise",
        "binary_threshold": binary_threshold,
        "overall_kappa": np.mean(pairwise_kappas) if pairwise_kappas else np.nan,
        "kappa_std": np.std(pairwise_kappas) if pairwise_kappas else np.nan,
        "feature_kappas": feature_avg_kappas,
        "pairwise_kappas": pairwise_kappas,
        "num_pairs": len(pairwise_kappas)
    }


def _fleiss_kappa(ratings):
    """
    Calculate Fleiss' kappa for multiple raters.
    
    Args:
        ratings: List of lists, where each inner list contains ratings for one item
                across multiple raters. Can contain binary values (0, 1).
    
    Returns:
        float: Fleiss' kappa coefficient
    """
    # Convert to numpy array and handle variable lengths
    max_raters = max(len(rating) for rating in ratings)
    n_items = len(ratings)
    
    # Create matrix with consistent number of raters (pad with NaN)
    rating_matrix = np.full((n_items, max_raters), np.nan)
    actual_raters = np.zeros(n_items)
    
    for i, rating in enumerate(ratings):
        rating_matrix[i, :len(rating)] = rating
        actual_raters[i] = len(rating)
    
    # For binary ratings (0, 1)
    categories = [0, 1]
    n_categories = len(categories)
    
    # Calculate proportion of agreement for each item
    p_i = np.zeros(n_items)
    for i in range(n_items):
        valid_ratings = rating_matrix[i, ~np.isnan(rating_matrix[i])]
        if len(valid_ratings) < 2:
            continue
            
        # Count agreements
        n_raters_i = len(valid_ratings)
        agreements = 0
        
        for category in categories:
            n_ij = np.sum(valid_ratings == category)
            agreements += n_ij * (n_ij - 1)
        
        p_i[i] = agreements / (n_raters_i * (n_raters_i - 1))
    
    # Average proportion of agreement
    P_bar = np.mean(p_i[actual_raters >= 2])  # Only items with at least 2 raters
    
    # Calculate marginal probabilities
    P_e_categories = np.zeros(n_categories)
    total_ratings = 0
    
    for i in range(n_items):
        valid_ratings = rating_matrix[i, ~np.isnan(rating_matrix[i])]
        for j, category in enumerate(categories):
            P_e_categories[j] += np.sum(valid_ratings == category)
        total_ratings += len(valid_ratings)
    
    P_e_categories = P_e_categories / total_ratings
    P_e = np.sum(P_e_categories ** 2)
    
    # Calculate Fleiss' kappa
    if P_e == 1:
        return np.nan
    
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


# Additional helper functions for other agreement types
def _calculate_fleiss_kappa_exact(sample_run_data, sample_ids, num_features, max_runs, code_order):
    """Calculate Fleiss' kappa on exact count matches."""
    # Implementation for exact count matching
    # This would be similar to binary but with multiple categories (count values)
    return {"agreement_type": "fleiss_exact", "note": "Implementation pending"}


def _calculate_fleiss_kappa_tolerance(sample_run_data, sample_ids, num_features, max_runs, 
                                    tolerance, code_order):
    """Calculate Fleiss' kappa with tolerance for count differences."""
    # Implementation for tolerance-based matching
    return {"agreement_type": "fleiss_tolerance", "tolerance": tolerance, "note": "Implementation pending"}


def _calculate_weighted_kappa(sample_run_data, sample_ids, num_features, max_runs, 
                            weight_function, code_order):
    """Calculate weighted kappa for count differences."""
    # Implementation for weighted kappa
    return {"agreement_type": "weighted_kappa", "weight_function": weight_function, "note": "Implementation pending"}