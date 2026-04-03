from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.coder import Coder


class CoderVLLM(Coder):
    def __init__(
        self, 
        model_id, 
        model_options,
        cache_dir=None,
        
        multi_gpus=8,
        torch_dtype="auto",
        
        sampling_parameters=None,
        max_input_tokens=73728,  # Maximum input tokens allowed

        codebook=None, 
        code_inst=None, 

        evaluation_method="generative",

        initial_code_example=None,
        correction_inst=None,

        patience=2,
        global_patience=10,
        stop_criteria=["max_rules", "global_patience"],

        think_budget=0,
        think_mode=False,
        templated_coding=False,

        model_name_translator=None,
        max_rule=40,
        ):

        # Call parent constructor with vLLM enabled
        super().__init__(
            model_id=model_id,
            model_options=model_options,
            cache_dir=cache_dir,
            use_vllm=True,  # Force vLLM usage
            compile=False,  # Not applicable for vLLM
            multi_gpus=multi_gpus,
            torch_dtype=torch_dtype,
            sampling_parameters=sampling_parameters,
            think_mode=think_mode,
            codebook=codebook,
            code_inst=code_inst,
            correction_inst=correction_inst,
            patience=patience,
            global_patience=global_patience,
            stop_criteria=stop_criteria,
            think_budget=think_budget,
            templated_coding=templated_coding,
            model_name_translator=model_name_translator,
            initial_code_example=initial_code_example,
            evaluation_method=evaluation_method,
            max_input_tokens=max_input_tokens,
            max_rule=max_rule,
        )
        
        print("MAX INPUT TOKENS:", self.max_input_tokens)
        print("Consider increasing this number if your reasoning traces are long and classification performance is bad.")

        # Override sampling parameters with vLLM-specific optimizations
        if sampling_parameters is None:
            self.sampling_parameters = SamplingParams(
                temperature=0.6, 
                top_p=0.95, 
                max_tokens=24576,
                top_k=50,
                # frequency_penalty=0.1
            )
        else:
            # Convert dict to SamplingParams if needed
            if isinstance(sampling_parameters, dict):
                self.sampling_parameters = SamplingParams(
                    temperature=sampling_parameters.get("temperature", 0.6),
                    top_p=sampling_parameters.get("top_p", 0.95),
                    max_tokens=sampling_parameters.get("max_new_tokens", 24576),
                    top_k=sampling_parameters.get("top_k", 50),
                    # frequency_penalty=sampling_parameters.get("frequency_penalty", 0.1)
                )
            else:
                self.sampling_parameters = sampling_parameters

    def init_model(self, cache_dir, multi_gpus=4):
        """Initialize vLLM model and tokenizer for optimal performance"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir)
        self.model = LLM(
            model=self.model_id, 
            tensor_parallel_size=multi_gpus, 
            download_dir=cache_dir,
            dtype=self.torch_dtype,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,  # Optimize GPU memory usage
            max_model_len=self.max_input_tokens,  # Increase context length for longer prompts
        )

    def _truncate_prompt_if_needed(self, text):
        """Truncate prompt if it exceeds the maximum input token limit"""
        # Tokenize the text to check length
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) > self.max_input_tokens:
            # Truncate tokens to fit within limit
            truncated_tokens = tokens[:self.max_input_tokens]
            # Decode back to text
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            print(f"Warning: Prompt truncated from {len(tokens)} to {len(truncated_tokens)} tokens", flush=True)
            return truncated_text
        
        return text

    def _generate(self, prompt, templated=False, max_tokens=None, temp_adjust=0):
        """Generate text using vLLM for single prompt"""
        if not templated:
            text = self.tokenizer.apply_chat_template(prompt, 
                                                      tokenize=False, 
                                                      add_generation_prompt=True,
                                                      enable_thinking=self.think_mode)
        else:
            text = prompt

        # Truncate if needed
        text = self._truncate_prompt_if_needed(text)

        # Create sampling params with potential adjustments
        sampling_params = SamplingParams(
            temperature=self.sampling_parameters.temperature + temp_adjust,
            top_p=self.sampling_parameters.top_p,
            max_tokens=max_tokens if max_tokens is not None else self.sampling_parameters.max_tokens,
            top_k=self.sampling_parameters.top_k
        )

        # Generate using vLLM
        outputs = self.model.generate([text], sampling_params=sampling_params, use_tqdm=True)
        classification_result = outputs[0].outputs[0].text

        return classification_result

    def _generate_batch(self, prompts, templated=False, max_tokens=None, temp_adjust=0):
        """Generate text using vLLM for batch of prompts - optimized for evaluation"""
        if not templated:
            texts = [self.tokenizer.apply_chat_template(prompt, 
                                                        tokenize=False, 
                                                        add_generation_prompt=True,
                                                        enable_thinking=self.think_mode) for prompt in prompts]
        else:
            texts = prompts

        # Truncate each text if needed
        texts = [self._truncate_prompt_if_needed(text) for text in texts]

        # Create sampling params with potential adjustments
        sampling_params = SamplingParams(
            temperature=self.sampling_parameters.temperature + temp_adjust,
            top_p=self.sampling_parameters.top_p,
            max_tokens=max_tokens if max_tokens is not None else self.sampling_parameters.max_tokens,
            top_k=self.sampling_parameters.top_k
        )

        # Generate using vLLM with batching
        outputs = self.model.generate(texts, sampling_params=sampling_params, use_tqdm=True)
        classification_results = [output.outputs[0].text for output in outputs]
        
        return classification_results

    def eval(self, dataset, verbosity=5, batched=True, batch_size=8, ckpt_path=None):
        """Evaluate the coder with vLLM optimizations - batched by default"""
        # Use parent's eval method but with batched=True by default for vLLM efficiency
        return super().eval(dataset, verbosity, batched, batch_size, ckpt_path)

    def from_pretrained(self, filename="coder_attr.ckpt"):
        """Load from pretrained checkpoint and reinitialize model if needed"""
        self.load_coder_attributes(filename)
        if self.model == None or self.tokenizer == None:
            # Use the cache_dir from initialization if available
            cache_dir = getattr(self, 'cache_dir', None) 
            multi_gpus = getattr(self, 'multi_gpus', 8)
            self.init_model(cache_dir, multi_gpus) 