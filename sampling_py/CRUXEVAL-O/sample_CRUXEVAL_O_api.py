# API-based sampling for CRUXEVAL-O

import os
import pickle
import argparse
from datasets import load_dataset
import anthropic
import openai
from google import genai
from xai_sdk import Client
from xai_sdk.chat import system, user
import time
import json

parser = argparse.ArgumentParser(description="Generate responses using API models.")
parser.add_argument('--model_id', type=str, help='The API model ID to use for generation. Defaults to grok-3-mini for grok provider.')
parser.add_argument('--api_provider', type=str, required=True, choices=['claude', 'openai', 'gemini', 'grok'], help='API provider to use.')
parser.add_argument('--temperature', type=float, default=0.8, help='The value used to module the next token probabilities.')
parser.add_argument('--max_new_tokens', type=int, default=8192, help='The maximum number of new tokens to generate.')
parser.add_argument('--output_base_dir', type=str, default='outputs', help='Base directory for output files.')
parser.add_argument('--backlog_base_dir', type=str, default='backlog/unfinished_thinking', help='Base directory for backlog files.')
args = parser.parse_args()

# Set default model for grok if not specified
if args.api_provider == 'grok' and args.model_id is None:
    args.model_id = 'grok-3-mini'
elif args.model_id is None:
    raise ValueError("model_id is required for non-grok providers")

# Task Overall Instruction
overall_inst = """Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do not output any extra information, even if the function is incorrect or incomplete.
\n
"""

# Initialize API clients
claude_client = None
openai_client = None
grok_client = None

if args.api_provider == 'claude':
    claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY_YEC"))
elif args.api_provider == 'openai':
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY_YEC"))
elif args.api_provider == 'gemini':
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY_YEC"))
elif args.api_provider == 'grok':
    grok_client = Client(api_key=os.getenv("XAI_API_KEY"))

# Load the dataset
ds_id = "cruxeval-org/cruxeval"
split = None
ds = load_dataset(ds_id, split)
split = "O"

# Define the output folder
dataset_name = ds_id.split("/")[-1]
if split:
    dataset_name += f"-{split}"
model_name = args.model_id.replace("/", "_").replace(":", "_")
output_folder = os.path.join(args.output_base_dir, dataset_name, model_name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load unfinished IDs
backlog_dir = os.path.join(args.backlog_base_dir, ds_id.split('/')[-1])
unfinished_ids = set()
unfinished_file = os.path.join(backlog_dir, f"unfinished_thoughts_{model_name}.pkl")
if os.path.exists(unfinished_file):
    with open(unfinished_file, "rb") as f:
        unfinished_ids.update(pickle.load(f))

def make_api_call(prompt, model_id, api_provider, temperature, max_tokens):
    """Make API call based on provider"""
    try:
        if api_provider == 'claude':
            stream = claude_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking={
                    "type": "enabled",
                    "budget_tokens": max_tokens // 2
                },
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            # Parse Claude's streaming thinking and text blocks
            thinking_content = ""
            answer_content = ""
            current_block_type = None
            
            for event in stream:
                if event.type == "content_block_start":
                    current_block_type = event.content_block.type
                elif event.type == "content_block_delta":
                    if current_block_type == "thinking":
                        thinking_content += event.delta.thinking if hasattr(event.delta, 'thinking') else ""
                    elif current_block_type == "text":
                        answer_content += event.delta.text if hasattr(event.delta, 'text') else ""
                elif event.type == "content_block_stop":
                    current_block_type = None
            
            return thinking_content, answer_content
        
        elif api_provider == 'openai':
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning={
                    "effort": "medium",
                    "summary": "auto"
                }
            )
            # Parse OpenAI's reasoning
            thinking_content = ""
            if hasattr(response.choices[0], 'reasoning') and response.choices[0].reasoning:
                if hasattr(response.choices[0].reasoning, 'summary') and response.choices[0].reasoning.summary:
                    for summary_item in response.choices[0].reasoning.summary:
                        thinking_content += summary_item
            answer_content = response.choices[0].message.content
            return thinking_content, answer_content
        
        elif api_provider == 'gemini':
            from google.genai import types
            client_gemini = genai.Client()
            response = client_gemini.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True
                    ),
                    generation_config=types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
            )
            # Parse Gemini's thinking and answer parts
            thinking_content = ""
            answer_content = ""
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    thinking_content += part.text
                else:
                    answer_content += part.text
            return thinking_content, answer_content
        
        elif api_provider == 'grok':
            chat = grok_client.chat.create(
                model=model_id,
                messages=[system("You are a highly intelligent AI assistant.")]
            )
            chat.append(user(prompt))
            
            response = chat.sample()
            
            # Extract reasoning and content from Grok response
            thinking_content = response.reasoning_content if hasattr(response, 'reasoning_content') else ""
            answer_content = response.content if hasattr(response, 'content') else ""
            
            return thinking_content, answer_content
    
    except Exception as e:
        print(f"API call failed: {e}")
        return None, None

# Loop through the dataset
for i in range(len(ds['test'])):
    question_id = ds['test']['id'][i]
    filename = f"{dataset_name}_question_id_{question_id}_{model_name}.txt"
    
    # Check if the output file already exists
    if os.path.exists(os.path.join(output_folder, filename)) and question_id not in unfinished_ids:
        continue

    question_prompt = overall_inst
    question_prompt += f"""
[PYTHON]
{ds['test']['code'][i]}
[/PYTHON]

assert f({ds['test']['input'][i]}) == ??

In your final output, surround your answer with no additional words, with [ANSWER] and [/ANSWER] tags. Your answer should be [ANSWER] [Expected outputs] [/ANSWER]"""
    
    print(f"Processing question {question_id}...")
    print(question_prompt)
    
    # Make API call
    thinking_content, answer_content = make_api_call(
        question_prompt, 
        args.model_id, 
        args.api_provider, 
        args.temperature, 
        args.max_new_tokens
    )
    
    if thinking_content is None or answer_content is None:
        print(f"Failed to get response for question {question_id}")
        continue
    
    # Save the output
    with open(os.path.join(output_folder, filename), "w") as f:
        f.write(f"### QUESTION: {question_prompt}\n\n")
        f.write(f"### THINKING: {thinking_content}\n\n")
        f.write(f"### ANSWER: {answer_content}")
    
    print(f"Saved response for question {question_id}")
    
    # Add small delay to avoid rate limiting
    time.sleep(0.1)  avoid rate limiting
    time.sleep(0.1) 