#!/bin/bash

# Script to run any vLLM coder training script on all pairs of models from a given list
# Usage: ./run_all_model_pairs_on_AIME.sh <script_path> [OPTIONS]

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
    echo "This script runs any vLLM coder training script on all pairs of models from a predefined list."
    echo ""
    echo "Arguments:"
    echo "  script_path    Path to the vLLM coder script to run (e.g., sh/run_vllm_coder_on_AIME.sh)"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -l, --list     Show the list of models that will be used"
    echo "  -d, --dry-run  Show what pairs would be generated without running"
    echo ""
    echo "Examples:"
    echo "  $0 sh/run_vllm_coder_on_AIME.sh"
    echo "  $0 sh/run_vllm_coder_on_cruxeval.sh --dry-run"
    echo "  $0 sh/run_vllm_coder_on_custom.sh --list"
    echo ""
    echo "To modify the list of models, edit the 'models' array in this script."
}

# Function to generate all pairs and run the specified script
run_all_pairs() {
    local script_path="$1"
    local model_list=("${@:2}")  # Get all arguments except the first
    local total_models=${#model_list[@]}
    local pair_count=0
    local script_name=$(basename "$script_path")
    
    echo "========================================="
    echo "Running $script_name on all model pairs"
    echo "Script: $script_path"
    echo "Total models: $total_models"
    echo "Models: ${model_list[*]}"
    echo "========================================="
    
    # Generate all pairs (combinations of 2)
    for ((i=0; i<total_models-1; i++)); do
        for ((j=i+1; j<total_models; j++)); do
            model1="${model_list[i]}"
            model2="${model_list[j]}"
            compared_models="${model1},${model2}"
            pair_count=$((pair_count + 1))
            
            echo ""
            echo "========================================="
            echo "Running pair $pair_count: $model1 vs $model2"
            echo "Compared models: $compared_models"
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
                echo "Error: Script failed for pair: $compared_models"
                echo "Continuing with next pair..."
            else
                echo "Successfully submitted job for pair: $compared_models"
            fi
            
            echo "Pair $pair_count submitted."
            echo ""
        done
    done
    
    echo "========================================="
    echo "All pairs submitted! Total pairs processed: $pair_count"
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
}

# Function for dry run - show pairs without executing
dry_run() {
    local script_path="$1"
    local total_models=${#models[@]}
    local pair_count=0
    
    echo "Dry run - pairs that would be generated for script: $script_path"
    echo "================================================================"
    
    for ((i=0; i<total_models-1; i++)); do
        for ((j=i+1; j<total_models; j++)); do
            model1="${models[i]}"
            model2="${models[j]}"
            pair_count=$((pair_count + 1))
            echo "Pair $pair_count: $model1 vs $model2"
            echo "  Command: sbatch $script_path \"$model1,$model2\" --account genai_interns --qos genai_interns"
        done
    done
    
    echo "================================================================"
    echo "Total pairs: $pair_count"
}

# Parse command line arguments
script_path=""
show_help=false
show_list=false
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
    echo "Please provide a valid path to a vLLM coder script"
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
run_all_pairs "$script_path" "${models[@]}"

echo "All model pair training jobs submitted!" 