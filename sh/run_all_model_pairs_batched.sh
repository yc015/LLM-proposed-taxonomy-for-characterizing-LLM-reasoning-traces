#!/bin/bash

# Script to run any vLLM coder training script on pairs of models from a given list
# This version submits one pair per job to match reduced GPU availability
# Usage: ./run_all_model_pairs_batched.sh <script_path> [OPTIONS]

# Define the list of models to compare
# You can modify this list as needed
models=(
    "QwQ-32B"
    "AceReason-Nemotron-14B"
    "Qwen3-14B"
    "DeepSeek-R1-Distill-Qwen-14B"
    "Phi-4-reasoning-plus"
    "Magistral-Small-2506"
)

# Function to display usage information
show_usage() {
    echo "Usage: $0 <script_path> [OPTIONS]"
    echo ""
    echo "This script runs any vLLM coder training script on pairs of models from a predefined list."
    echo "It submits one pair per job using the compared_models parameter."
    echo ""
    echo "Arguments:"
    echo "  script_path    Path to the vLLM coder script to run (e.g., sh/run_vllm_coder_on_gpqa_two_node.sh)"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -l, --list     Show the list of models that will be used"
    echo "  -d, --dry-run  Show what individual pairs would be generated without running"
    echo ""
    echo "Examples:"
    echo "  $0 sh/run_vllm_coder_on_gpqa_two_node.sh"
    echo "  $0 sh/run_vllm_coder_on_custom_two_node.sh --dry-run"
    echo "  $0 sh/run_vllm_coder_on_batched.sh --list"
    echo ""
    echo "Note: This script is designed for scripts that accept a single compared_models parameter."
    echo "To modify the list of models, edit the 'models' array in this script."
}

# Function to generate all pairs and create batches
generate_all_pairs() {
    local model_list=("$@")
    local total_models=${#model_list[@]}
    local pairs=()
    
    # Generate all pairs (combinations of 2)
    for ((i=0; i<total_models-1; i++)); do
        for ((j=i+1; j<total_models; j++)); do
            model1="${model_list[i]}"
            model2="${model_list[j]}"
            compared_models="${model1},${model2}"
            pairs+=("$compared_models")
        done
    done
    
    echo "${pairs[@]}"
}

# Function to run pairs on the specified script
run_pairs() {
    local script_path="$1"
    local model_list=("${@:2}")  # Get all arguments except the first
    local script_name=$(basename "$script_path")
    
    # Generate all pairs
    local all_pairs=($(generate_all_pairs "${model_list[@]}"))
    local total_pairs=${#all_pairs[@]}
    local batch_count=0
    
    echo "========================================="
    echo "Running $script_name on model pairs"
    echo "Script: $script_path"
    echo "Total models: ${#model_list[@]}"
    echo "Models: ${model_list[*]}"
    echo "Total pairs: $total_pairs"
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
    echo "Models that will be used for pair generation:"
    for i in "${!models[@]}"; do
        echo "  $((i+1)). ${models[i]}"
    done
    echo ""
    echo "Total models: ${#models[@]}"
    echo "Total pairs: $(( ${#models[@]} * (${#models[@]} - 1) / 2 ))"
    echo "Total batches: $(( ${#models[@]} * (${#models[@]} - 1) / 2 ))"
}

# Function for dry run - show pairs without executing
dry_run() {
    local script_path="$1"
    local all_pairs=($(generate_all_pairs "${models[@]}"))
    local total_pairs=${#all_pairs[@]}
    local batch_count=0
    
    echo "Dry run - pairs that would be generated for script: $script_path"
    echo "========================================================================"
    
    for ((i=0; i<total_pairs; i++)); do
        batch_count=$((batch_count + 1))
        compared_models="${all_pairs[i]}"
        
        echo "Batch $batch_count:"
        echo "  compared_models: $compared_models"
        echo "  Command: sbatch $script_path \"$compared_models\""
        echo ""
    done
    
    echo "========================================================================"
    echo "Total pairs: $total_pairs"
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
echo "Starting model pair training with script: $script_path"
echo "Current working directory: $(pwd)"
echo "Script to run: $script_path"

if [[ "$dry_run_mode" == true ]]; then
    dry_run "$script_path"
    exit 0
fi

# Run all pairs
run_pairs "$script_path" "${models[@]}"

echo "All model pair training jobs submitted!" 