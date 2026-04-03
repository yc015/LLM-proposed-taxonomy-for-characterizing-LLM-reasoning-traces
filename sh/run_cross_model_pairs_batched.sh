#!/bin/bash

# Script to run any vLLM coder training script on cross-pairs between two model lists
# This version generates pairs by comparing each model from list 1 with each model from list 2
# It submits one pair per job to align with reduced GPU availability
# Usage: ./run_cross_model_pairs_batched.sh <script_path> [OPTIONS]

# Define the first list of models (baseline models)
models_list1=(
    "Qwen3-0.6B"
    "Qwen3-1.7B"
    "Qwen3-4B"
    "Qwen3-8B"
    "Qwen3-14B"
)

# Define the second list of models (comparison models)
models_list2=(
    "Qwen3-32B"
)

# Function to display usage information
show_usage() {
    echo "Usage: $0 <script_path> [OPTIONS]"
    echo ""
    echo "This script runs any vLLM coder training script on cross-pairs between two model lists."
    echo "It generates pairs by comparing each model from list 1 with each model from list 2."
    echo "Pairs are submitted individually using the compared_models parameter."
    echo ""
    echo "Arguments:"
    echo "  script_path    Path to the vLLM coder script to run (e.g., sh/run_vllm_coder_on_gpqa_two_node.sh)"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -l, --list     Show the model lists that will be used"
    echo "  -d, --dry-run  Show what cross-pairs would be generated without running"
    echo ""
    echo "Examples:"
    echo "  $0 sh/run_vllm_coder_on_gpqa_two_node.sh"
    echo "  $0 sh/run_vllm_coder_on_custom_two_node.sh --dry-run"
    echo "  $0 sh/run_vllm_coder_on_batched.sh --list"
    echo ""
    echo "Note: This script is designed for scripts that accept a single compared_models parameter."
    echo "To modify the model lists, edit the 'models_list1' and 'models_list2' arrays in this script."
}

# Function to generate cross-pairs between two lists
generate_cross_pairs() {
    local list1=("${models_list1[@]}")
    local list2=("${models_list2[@]}")
    local pairs=()
    
    # Generate all cross-pairs (Cartesian product)
    for model1 in "${list1[@]}"; do
        for model2 in "${list2[@]}"; do
            compared_models="${model1},${model2}"
            pairs+=("$compared_models")
        done
    done
    
    echo "${pairs[@]}"
}

# Function to run cross-pairs on the specified script
run_cross_pairs() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    # Generate all cross-pairs
    local all_pairs=($(generate_cross_pairs))
    local total_pairs=${#all_pairs[@]}
    local batch_count=0
    
    echo "========================================="
    echo "Running $script_name on cross-model pairs"
    echo "Script: $script_path"
    echo "List 1 models (${#models_list1[@]}): ${models_list1[*]}"
    echo "List 2 models (${#models_list2[@]}): ${models_list2[*]}"
    echo "Total cross-pairs: $total_pairs"
    echo "Total batches: $total_pairs"
    echo "========================================="
    
    for ((i=0; i<total_pairs; i++)); do
        batch_count=$((batch_count + 1))
        compared_models="${all_pairs[i]}"
        
        echo ""
        echo "========================================="
        echo "Running batch $batch_count:"
        echo "  compared_models: $compared_models"
        echo "Script: $script_path"
        echo "========================================="
        
        # Check if the script exists
        if [[ ! -f "$script_path" ]]; then
            echo "Error: Script not found: $script_path"
            exit 1
        fi
        
        # Run the script with the current pair
        sbatch "$script_path" "$compared_models"
        
        # Check if the previous command was successful
        if [[ $? -ne 0 ]]; then
            echo "Error: Script failed for batch $batch_count"
            echo "Continuing with next batch..."
        else
            echo "Successfully submitted job for batch $batch_count"
        fi
        
        echo "Batch $batch_count submitted."
        echo ""
    done
    
    echo "========================================="
    echo "All batches submitted! Total batches processed: $batch_count"
    echo "Use 'squeue -u $USER' to check job status"
    echo "========================================="
}

# Function to list models
list_models() {
    echo "Model List 1 (baseline models):"
    for i in "${!models_list1[@]}"; do
        echo "  $((i+1)). ${models_list1[i]}"
    done
    echo ""
    echo "Model List 2 (comparison models):"
    for i in "${!models_list2[@]}"; do
        echo "  $((i+1)). ${models_list2[i]}"
    done
    echo ""
    echo "Total models in list 1: ${#models_list1[@]}"
    echo "Total models in list 2: ${#models_list2[@]}"
    echo "Total cross-pairs: $(( ${#models_list1[@]} * ${#models_list2[@]} ))"
    echo "Total batches: $(( ${#models_list1[@]} * ${#models_list2[@]} ))"
}

# Function for dry run - show cross-pairs without executing
dry_run() {
    local script_path="$1"
    local all_pairs=($(generate_cross_pairs))
    local total_pairs=${#all_pairs[@]}
    local batch_count=0
    
    echo "Dry run - cross-pairs that would be generated for script: $script_path"
    echo "========================================================================="
    echo "List 1: ${models_list1[*]}"
    echo "List 2: ${models_list2[*]}"
    echo ""
    
    # Show individual pairs
    echo "All cross-pairs:"
    for ((idx=0; idx<total_pairs; idx++)); do
        echo "  $((idx+1)). ${all_pairs[idx]}"
    done
    echo ""
    
    # Show batches
    echo "Execution order:"
    for ((i=0; i<total_pairs; i++)); do
        batch_count=$((batch_count + 1))
        compared_models="${all_pairs[i]}"
        
        echo "Batch $batch_count:"
        echo "  compared_models: $compared_models"
        echo "  Command: sbatch $script_path \"$compared_models\""
        echo ""
    done
    
    echo "========================================================================="
    echo "Total cross-pairs: $total_pairs"
    echo "Total batches: $batch_count"
}

# Parse command line arguments
script_path=""
dry_run_mode=false

# Check if first argument is an option or script path
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    -l|--list)
        list_models
        exit 0
        ;;
    -d|--dry-run)
        echo "Error: Please specify script path before --dry-run"
        echo "Usage: $0 <script_path> --dry-run"
        exit 1
        ;;
    "")
        echo "Error: Script path is required"
        show_usage
        exit 1
        ;;
    -*)
        echo "Error: Unknown option: $1"
        echo "Script path must be specified first"
        show_usage
        exit 1
        ;;
    *)
        script_path="$1"
        shift
        ;;
esac

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--list)
            list_models
            exit 0
            ;;
        -d|--dry-run)
            dry_run_mode=true
            ;;
        --account|--qos)
            # These are passed to sbatch, skip them and their values
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
    shift
done

# Validate script path
if [[ ! -f "$script_path" ]]; then
    echo "Error: Script not found: $script_path"
    echo "Please provide a valid path to a vLLM coder script that supports single pair comparisons"
    exit 1
fi

# Main execution
echo "Starting cross-model pair training with script: $script_path"
echo "Current working directory: $(pwd)"
echo "Script to run: $script_path"

if [[ "$dry_run_mode" == true ]]; then
    dry_run "$script_path"
    exit 0
fi

# Run all cross-pairs
run_cross_pairs "$script_path"

echo "All cross-model pair training jobs submitted!" 