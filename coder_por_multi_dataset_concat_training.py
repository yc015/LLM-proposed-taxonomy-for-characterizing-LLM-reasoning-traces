import argparse
import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.coder_por import CoderPOR
from src.initial_code_examples import AIME_EXAMPLES, CRUXEVAL_EXAMPLES, GPQA_EXAMPLES
from src.initial_codebook import INITIAL_CODEBOOK, INITIAL_CODEBOOK_VML
from src.prompt_dataset import load_reasoning_traces, shuffle_outputs_and_labels
from src import VML_CODE_INST, VML_CORRECTION_INST, model_name_translator


MODEL_NAME_TRANSLATOR = model_name_translator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train POR coder on multiple datasets with stratified splits."
    )
    parser.add_argument("--dataset_paths", type=str, required=True, help="Comma-separated dataset paths.")
    parser.add_argument("--coder_model_id", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--compared_models", type=str, default="Qwen3-14B,Phi-4-reasoning-plus")
    parser.add_argument("--think_mode", type=str, default="no")
    parser.add_argument("--think_budget", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--num_warmup", type=int, default=1, help="Number of warmup samples (fixed count).")
    parser.add_argument("--num_train", type=float, default=0.8, help="Train split ratio after warmup.")
    parser.add_argument("--num_test", type=float, default=0.2, help="Test split ratio after warmup.")
    parser.add_argument("--max_rule", type=int, default=40)
    parser.add_argument("--max_train_samples", type=int, default=320)
    parser.add_argument("--accumulation_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--evaluation_method", type=str, default="generative")
    parser.add_argument("--run_type", type=str, default="Normal")
    parser.add_argument("--normalization_method", type=str, default="comparative")
    parser.add_argument("--imputation_method", type=str, default="knn")
    parser.add_argument("--global_patience", type=int, default=15)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--averaged_eval", action="store_true")
    parser.add_argument("--num_reruns", type=int, default=5)
    parser.add_argument("--reject_inconsistent_codes", action="store_true")
    parser.add_argument("--sampling_training", action="store_true")
    parser.add_argument("--extend_with_nan", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--multi_gpus", type=int, default=4)
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_initial_codebook", type=str, default="yes")
    parser.add_argument("--fixed_codebook_baseline", type=str, default="no")
    return parser.parse_args()


def seed_everything(seed: int):
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def translate_models(raw_models, run_type):
    translator = dict(MODEL_NAME_TRANSLATOR)
    if run_type == "VML":
        for model in raw_models:
            translator[model] = model
    for model in raw_models:
        if model not in translator:
            translator[model] = model
    encoded = [translator[m] for m in raw_models]
    decoder = {v: k for k, v in translator.items()}
    return encoded, decoder


def determine_initial_examples(dataset_paths):
    for path in dataset_paths:
        lower = path.lower()
        if "cruxeval" in lower:
            return CRUXEVAL_EXAMPLES
        if "gpqa" in lower:
            return GPQA_EXAMPLES
        if "aime" in lower or "math" in lower:
            return AIME_EXAMPLES
    return None


def create_coder(args, dataset_paths, model_options, codebook=None):
    enabled_thinking = args.think_mode.lower() == "yes"
    if "llama" in args.coder_model_id.lower():
        max_input_tokens = 73728
    else:
        max_input_tokens = 24576
    initial_code_example = determine_initial_examples(dataset_paths)

    code_inst = None
    correction_inst = None
    evaluation_method = args.evaluation_method
    min_num_train_samples = 200

    dataset_name_concat = " ".join(dataset_paths).lower()
    if "cruxeval" in dataset_name_concat or "execution" in dataset_name_concat:
        min_num_train_samples = 300
    elif "gpqa" in dataset_name_concat:
        min_num_train_samples = 100
    elif "aime" in dataset_name_concat:
        min_num_train_samples = 44

    if args.run_type == "VML":
        code_inst = VML_CODE_INST
        correction_inst = VML_CORRECTION_INST
        evaluation_method = "generative"
        initial_code_example = INITIAL_CODEBOOK_VML
        codebook = None
        max_input_tokens = 122880
        min_num_train_samples = 50

    coder = CoderPOR(
        args.coder_model_id,
        model_options,
        think_mode=enabled_thinking,
        cache_dir="/n/holylabs/LABS/wattenberg_lab/Lab/pretrained_models/",
        think_budget=args.think_budget,
        multi_gpus=args.multi_gpus,
        sampling_parameters=None,
        max_input_tokens=max_input_tokens,
        global_patience=args.global_patience,
        patience=args.patience,
        initial_code_example=initial_code_example,
        evaluation_method=evaluation_method,
        code_inst=code_inst,
        correction_inst=correction_inst,
        max_rule=args.max_rule,
        codebook=codebook,
    )
    coder.min_num_train_samples = min_num_train_samples
    coder._extend_with_nan = args.extend_with_nan
    coder._imputation_method = args.imputation_method
    coder.max_train_samples = args.max_train_samples
    coder.stop_criteria = ["max_rules", "global_patience", "max_train_samples"]

    if args.run_type == "VML":
        coder.run_type = "VML"
        coder.batch_update = True
        if "aime" in dataset_name_concat:
            coder.batch_update_size = 1
        elif "gpqa" in dataset_name_concat:
            coder.batch_update_size = 2
        else:
            coder.batch_update_size = 6
        coder.patience = 1
    else:
        coder.run_type = "Normal"
        coder.batch_update = False
        coder.batch_update_size = 1

    if args.fixed_codebook_baseline == "yes":
        coder.no_update_to_codebook = True

    return coder


def build_dataset_records(reasoning_traces, decoder, model_options, source_name):
    model_codename_0 = decoder[model_options[0]]
    model_codename_1 = decoder[model_options[1]]
    if model_codename_0 not in reasoning_traces or model_codename_1 not in reasoning_traces:
        raise ValueError(f"Missing compared models in dataset {source_name}")

    shared_ids = sorted(
        set(reasoning_traces[model_codename_0].keys())
        & set(reasoning_traces[model_codename_1].keys())
    )
    outputs = [
        [
            reasoning_traces[model_codename_0][idx]["thinking"],
            reasoning_traces[model_codename_1][idx]["thinking"],
        ]
        for idx in shared_ids
    ]
    questions = [reasoning_traces[model_codename_0][idx]["question"] for idx in shared_ids]
    labels = [[model_options[0], model_options[1]] for _ in shared_ids]
    ids = shared_ids

    shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions = shuffle_outputs_and_labels(
        outputs,
        labels,
        ids,
        questions,
    )

    dataset = []
    for out_pair, label_pair, sample_id, question in zip(
        shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions
    ):
        dataset.append(
            {
                "outputs": out_pair,
                "labels": label_pair,
                "id": f"{source_name}_{sample_id}",
                "question": question,
                "source": source_name,
            }
        )
    return dataset


def stratified_splits(dataset, num_warmup, train_ratio, test_ratio, seed):
    total_samples = len(dataset)
    if total_samples < 3:
        raise ValueError("Need at least 3 samples for warmup/train/test splits.")

    warm_count = int(max(1, num_warmup))
    warm_count = min(warm_count, total_samples - 2)
    if warm_count < 1:
        raise ValueError("Warmup split must contain at least one sample.")

    sources = [sample["source"] for sample in dataset]
    warmup_dataset, remaining_dataset = train_test_split(
        dataset,
        train_size=warm_count,
        random_state=seed,
        stratify=None,
    )
    remaining_sources = [sample["source"] for sample in remaining_dataset]
    if len(remaining_dataset) < 2:
        raise ValueError("Not enough samples remaining for train/test split.")

    train_dataset, eval_dataset, _, _ = train_test_split(
        remaining_dataset,
        remaining_sources,
        train_size=train_ratio,
        test_size=test_ratio,
        random_state=seed,
        stratify=remaining_sources,
    )
    return warmup_dataset, train_dataset, eval_dataset


def evaluate_dataset(coder, dataset, args, label):
    print(f"Evaluating on {label} ({len(dataset)} samples)")
    if args.averaged_eval:
        coder.eval_with_averaged_vectors(
            dataset,
            num_reruns=args.num_reruns,
            variance_calculation=True,
        )
    else:
        coder.eval(
            dataset,
            batched=True,
            batch_size=args.batch_size,
        )
    return coder.training_logs.get("eval_acc")


def main():
    args = parse_args()
    seed = args.seed if args.seed > 0 else 42
    seed_everything(seed)

    dataset_paths = [path.strip() for path in args.dataset_paths.split(",")]
    raw_models = [m.strip() for m in args.compared_models.split(",")]
    encoded_models, decoder = translate_models(raw_models, args.run_type)

    combined_records = []
    for dataset_path in dataset_paths:
        traces = load_reasoning_traces(dataset_path)
        source_name = os.path.basename(dataset_path.rstrip("/"))
        records = build_dataset_records(traces, decoder, encoded_models, source_name)
        combined_records.extend(records)
        print(f"Loaded {len(records)} samples from {source_name}")

    warmup_dataset, train_dataset, eval_dataset = stratified_splits(
        combined_records,
        args.num_warmup,
        args.num_train,
        args.num_test,
        seed,
    )
    print(
        f"Dataset sizes -> Warmup: {len(warmup_dataset)}, "
        f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}"
    )

    if args.use_initial_codebook == "yes" or args.fixed_codebook_baseline == "yes":
        initial_codebook = INITIAL_CODEBOOK
    else:
        initial_codebook = None
    coder = create_coder(args, dataset_paths, encoded_models, codebook=initial_codebook)

    output_base = args.output_dir or "/n/home04/yidachen/reasoning_characteristics/coder_ckpt"
    dataset_tag = "_".join([os.path.basename(path.rstrip("/")) for path in dataset_paths])
    folder = os.path.join(output_base, f"{dataset_tag}-PoR-multidataset")
    os.makedirs(folder, exist_ok=True)
    intermediate = os.path.join(folder, "intermediate_ckpt")
    os.makedirs(intermediate, exist_ok=True)

    warm_attr = os.path.join(intermediate, f"{args.job_id}_warmup.pkl")
    train_attr = os.path.join(intermediate, f"{args.job_id}_train.pkl")

    coder.warm_start(warmup_dataset, ckpt_path=warm_attr)
    coder.train(
        train_dataset,
        ckpt_path=train_attr,
        accumulate_observation_training=True,
        accumulation_size=args.accumulation_size,
        sampling_training=args.sampling_training,
        batch_size=args.accumulation_size,
    )
    if args.reject_inconsistent_codes:
        coder.reject_inconsistent_codes(
            random.sample(train_dataset, min(50, len(train_dataset))),
            num_reruns=args.num_reruns,
            modify_training_data=True,
        )

    eval_acc = evaluate_dataset(coder, eval_dataset, args, "combined eval")
    coder_path = os.path.join(
        folder,
        f"coder-por-multidataset-{dataset_tag}-{args.job_id}.pkl",
    )
    coder.save_coder(coder_path)

    summary_path = os.path.join(folder, f"summary-{args.job_id}.txt")
    with open(summary_path, "w") as f:
        f.write("POR Multi-dataset Training Summary\n")
        f.write(f"Datasets: {dataset_paths}\n")
        f.write(f"Coder checkpoint: {coder_path}\n")
        f.write(f"Eval accuracy: {eval_acc}\n")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
