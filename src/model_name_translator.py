model_name_translator = {
                        "Qwen3-0.6B": "Theta",
                         "Qwen3-1.7B": "Sigma",
                         "Qwen3-4B": "Zeta",
                         "Qwen3-8B": "Delta",
                         "Qwen3-14B": "Gamma",
                         "Qwen3-32B": "Epsilon",
                         "Phi-4-reasoning": "Bacon",
                         "Phi-4-reasoning-plus": "Omelette",
                         "AceReason-Nemotron-14B": "Eta",
                         "QwQ-32B": "Kappa",
                         "DeepSeek-R1-Distill-Qwen-14B": "Omicron",
                         "Magistral-Small-2506": "Maria",
                         "grok-3-mini": "George",
                         "claude-3-7-sonnet-20250219": "Charles",
                         "Seed-Coder-8B-Reasoning": "Iota",
                         "Seed-Coder-8B-Reasoning-Format": "Upsilon",
                         }
model_name_decoder = {value: key for key, value in model_name_translator.items()}