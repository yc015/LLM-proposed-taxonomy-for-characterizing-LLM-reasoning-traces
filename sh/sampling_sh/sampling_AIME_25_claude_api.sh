#!/bin/bash

#SBATCH --job-name=Claude3-7-Sonnet-AIME25-API-Sampling
#SBATCH --time=0-12:00:00    # run for half a day - API calls can be slower
#SBATCH --cpus-per-task=2    # change as needed
 
 -cpu-only
## %j is the job id, %u is the user id
#SBATCH --output=/n/home04/yidachen/bash_output/sampling/Claude3-7-Sonnet-AIME-25-API-sampling-%j.log

source activate rifeval

# Set API keys - REQUIRED for API sampling
# Uncomment and set your actual API keys:
export ANTHROPIC_API_KEY_YEC="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"
# export OPENAI_API_KEY_YEC="your-openai-api-key-here"  
# export GOOGLE_API_KEY_YEC="your-google-api-key-here"

# Alternative: Source from a secure file (recommended)
# source ~/.api_keys  # Create this file with your API keys

# Run Claude 3.7 Sonnet on AIME 25 dataset
python sampling_py/sample_AIME_25_api.py \
    --model_id "claude-3-7-sonnet-20250219" \
    --api_provider "claude" \
    --temperature 0.6 \
    --max_new_tokens 32768

# Alternative API providers and models (uncomment as needed):
# Claude variants:
# python sampling_py/sample_AIME_25_api.py --model_id "claude-3-5-sonnet-20241022" --api_provider "claude" --temperature 0.7 --max_new_tokens 32768
# python sampling_py/sample_AIME_25_api.py --model_id "claude-3-5-haiku-20241022" --api_provider "claude" --temperature 0.7 --max_new_tokens 32768

# OpenAI variants:
# python sampling_py/sample_AIME_25_api.py --model_id "gpt-4o" --api_provider "openai" --temperature 0.7 --max_new_tokens 32768
# python sampling_py/sample_AIME_25_api.py --model_id "o1" --api_provider "openai" --temperature 0.7 --max_new_tokens 32768

# Gemini variants:
# python sampling_py/sample_AIME_25_api.py --model_id "gemini-1.5-pro" --api_provider "gemini" --temperature 0.7 --max_new_tokens 32768
# python sampling_py/sample_AIME_25_api.py --model_id "gemini-2.0-flash-exp" --api_provider "gemini" --temperature 0.7 --max_new_tokens 32768

# Different temperature settings:
# python sampling_py/sample_AIME_25_api.py --model_id "claude-3-7-sonnet-20250219" --api_provider "claude" --temperature 0.8 --max_new_tokens 32768
# python sampling_py/sample_AIME_25_api.py --model_id "claude-3-7-sonnet-20250219" --api_provider "claude" --temperature 0.6 --max_new_tokens 32768

echo "Completed Claude 3.7 Sonnet sampling on AIME 25 dataset" 