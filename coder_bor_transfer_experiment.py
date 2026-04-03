import argparse
import os
import random
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.coder_bor_individual import CoderBORIndividual
from src.initial_code_examples import AIME_EXAMPLES, CRUXEVAL_EXAMPLES, GPQA_EXAMPLES
from src.initial_codebook import INITIAL_CODEBOOK
from src.prompt_dataset import load_reasoning_traces, shuffle_outputs_and_labels
from src import VML_CODE_INST, VML_CORRECTION_INST, model_name_translator


MODEL_NAME_TRANSLATOR = model_name_translator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BOR coder on dataset A and evaluate transfer on dataset B (and vice versa)."
    )
    parser.add_argument("--dataset_a_path", type=str, required=True)
    parser.add_argument("--dataset_b_path", type=str, required=True)
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--evaluation_method", type=str, default="logistic_regression")
    parser.add_argument("--run_type", type=str, default="Normal")
    parser.add_argument("--normalization_method", type=str, default="comparative")
    parser.add_argument("--imputation_method", type=str, default="mean")
    parser.add_argument("--global_patience", type=int, default=15)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--averaged_eval", action="store_true")
    parser.add_argument("--num_reruns", type=int, default=5)
    parser.add_argument("--reject_inconsistent_codes", action="store_true")
    parser.add_argument("--sampling_training", action="store_true")
    parser.add_argument("--extend_with_nan", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--multi_gpus", type=int, default=1)
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_initial_codebook", type=str, default="yes")
    parser.add_argument("--fixed_codebook_baseline", type=str, default="no")
    parser.add_argument("--refit_transfer_classifier", type=str, default="yes", choices=["yes", "no"])
    return parser.parse_args()


def seed_everything(seed: int):
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def translate_models(raw_models):
    translator = dict(MODEL_NAME_TRANSLATOR)
    for model in raw_models:
        if model not in translator:
            translator[model] = model
    encoded = [translator[m] for m in raw_models]
    decoder = {v: k for k, v in translator.items()}
    return encoded, decoder


def determine_initial_examples(dataset_path):
    path = dataset_path.lower()
    if "cruxeval" in path:
        return CRUXEVAL_EXAMPLES
    if "gpqa" in path:
        return GPQA_EXAMPLES
    if "aime" in path or "math" in path:
        return AIME_EXAMPLES
    return None


def create_coder(args, dataset_path, model_options, codebook=None):
    enabled_thinking = args.think_mode.lower() == "yes"
    if "llama" in args.coder_model_id.lower():
        max_input_tokens = 73728
    else:
        max_input_tokens = 24576
    if "aime" in dataset_path.lower():
        max_input_tokens = 110592

    initial_code_example = determine_initial_examples(dataset_path)
    code_inst = None
    correction_inst = None

    if args.run_type == "VML":
        code_inst = VML_CODE_INST
        correction_inst = VML_CORRECTION_INST
        initial_code_example = None

    coder = CoderBORIndividual(
        args.coder_model_id,
        model_options,
        think_mode=enabled_thinking,
        cache_dir="/n/holylabs/LABS/wattenberg_lab/Lab/pretrained_models/",
        think_budget=args.think_budget,
        sampling_parameters=None,
        max_input_tokens=max_input_tokens,
        global_patience=args.global_patience,
        patience=args.patience,
        initial_code_example=initial_code_example,
        evaluation_method=args.evaluation_method,
        code_inst=code_inst,
        correction_inst=correction_inst,
        max_rule=args.max_rule,
        multi_gpus=args.multi_gpus,
        codebook=codebook,
    )

    if args.fixed_codebook_baseline == "yes":
        coder.no_update_to_codebook = True

    coder._extend_with_nan = args.extend_with_nan
    coder._imputation_method = args.imputation_method
    coder._normalize_vectors = args.normalize
    coder._normalization_method = args.normalization_method
    coder.max_train_samples = args.max_train_samples
    coder.stop_criteria = ["max_rules", "global_patience", "max_train_samples"]
    return coder


def prepare_dataset_sections(
    reasoning_traces,
    decoder,
    encoded_model_options,
    warmup_size,
    train_ratio,
    test_ratio,
    seed,
):
    model_codename_0 = decoder[encoded_model_options[0]]
    model_codename_1 = decoder[encoded_model_options[1]]

    missing_models = [
        model_codename_0 not in reasoning_traces,
        model_codename_1 not in reasoning_traces,
    ]
    if any(missing_models):
        raise ValueError("Compared models not found in reasoning traces.")

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
    labels = [[encoded_model_options[0], encoded_model_options[1]] for _ in shared_ids]
    ids = shared_ids

    shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions = shuffle_outputs_and_labels(
        outputs,
        labels,
        ids,
        questions,
    )

    dataset = [
        {
            "outputs": out_pair,
            "labels": label_pair,
            "id": sample_id,
            "question": question,
        }
        for out_pair, label_pair, sample_id, question in zip(
            shuffled_outputs,
            shuffled_labels,
            shuffled_ids,
            shuffled_questions,
        )
    ]

    warm_count = min(int(max(1, warmup_size)), len(dataset))
    if warm_count >= len(dataset):
        raise ValueError("Not enough samples for warmup split.")

    warmup_dataset, remaining_dataset = train_test_split(
        dataset,
        train_size=warm_count,
        random_state=seed,
    )
    if len(remaining_dataset) < 2:
        raise ValueError("Not enough samples remaining for train/test split.")

    train_dataset, eval_dataset = train_test_split(
        remaining_dataset,
        train_size=train_ratio,
        test_size=test_ratio,
        random_state=seed,
    )
    return warmup_dataset, train_dataset, eval_dataset


def evaluate_dataset(coder, dataset, args, label):
    print(f"Evaluating on {label} with {len(dataset)} samples.")
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


def run_training_cycle(
    args,
    dataset_name,
    dataset_path,
    model_options,
    warmup_dataset,
    train_dataset,
    eval_dataset,
    output_dir,
):
    initial_codebook = INITIAL_CODEBOOK if args.use_initial_codebook == "yes" else None
    coder = create_coder(args, dataset_path, model_options, codebook=initial_codebook)

    dataset_folder = os.path.basename(dataset_path.rstrip("/"))
    folder = os.path.join(output_dir, f"{dataset_folder}-BoR-transfer")
    os.makedirs(folder, exist_ok=True)
    intermediate = os.path.join(folder, "intermediate_ckpt")
    os.makedirs(intermediate, exist_ok=True)
    coder_path = os.path.join(
        folder,
        f"coder-bor-{dataset_folder}-{args.job_id}.pkl",
    )

    warm_attr = os.path.join(intermediate, f"{dataset_name}-warmup-{args.job_id}.pkl")
    train_attr = os.path.join(intermediate, f"{dataset_name}-train-{args.job_id}.pkl")

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

    in_domain_acc = evaluate_dataset(coder, eval_dataset, args, f"{dataset_name} test split")
    coder.save_coder(coder_path)
    coder.init_model(cache_dir="/n/holylabs/LABS/wattenberg_lab/Lab/pretrained_models/", multi_gpus=args.multi_gpus)
    return coder, in_domain_acc, coder_path


def annotate_transfer_training_set(coder, dataset):
    if not dataset:
        return None
    code_order = list(coder.codebook.keys()) if coder.codebook else []
    result = {
        "vectors": {},
        "code_order": code_order,
        "sample_ids": [],
        "labels": [],
    }
    for sample in dataset:
        _, _, _, _, final_output = coder.classify(
            sample["outputs"],
            sample["labels"],
            sample["question"],
            train=False,
            qid=f"transfer_{sample['id']}",
        )
        code_occurrence, _ = coder._parse_classification_codes(final_output, sample["labels"])
        coder._update_code_occurence(result, code_occurrence, is_decision_code=False, sample_id=sample["id"])
    return result


def evaluate_transfer(
    args,
    trained_coder,
    model_options,
    target_dataset_path,
    transfer_train_dataset,
    transfer_eval_dataset,
    direction_tag,
    allow_refit,
):
    results = {}
    zero_label = f"{direction_tag} (without refit) eval"
    zero_acc = evaluate_dataset(trained_coder, transfer_eval_dataset, args, zero_label)
    results["zero_shot_eval"] = zero_acc

    if allow_refit and transfer_train_dataset:
        trained_coder.model = None
        trained_coder.tokenizer = None
        torch.cuda.empty_cache()

        transfer_coder = create_coder(
            args, 
            target_dataset_path, 
            model_options, 
            codebook=deepcopy(trained_coder.codebook))
        transfer_coder.no_update_to_codebook = True
        annotation_result = annotate_transfer_training_set(transfer_coder, transfer_train_dataset)
        if annotation_result:
            transfer_coder.code_occurrence_overall["train"] = annotation_result
            transfer_coder.training_logs["train"] = {}
            transfer_coder._most_recent_code_occurrence_overall_train = annotation_result
            transfer_coder.logistic_classifier = None
            transfer_coder.knn_classifier = None
            transfer_coder.scaler = None
            refit_label = f"{direction_tag} refit eval"
            refit_acc = evaluate_dataset(transfer_coder, transfer_eval_dataset, args, refit_label)
            results["refit_eval"] = refit_acc
        transfer_coder.model = None
        transfer_coder.tokenizer = None
    return results


def main():
    args = parse_args()
    seed = args.seed if args.seed > 0 else 42
    seed_everything(seed)

    raw_models = [m.strip() for m in args.compared_models.split(",")]
    encoded_models, decoder = translate_models(raw_models)
    base_output = args.output_dir or "/n/home04/yidachen/reasoning_characteristics/coder_ckpt"

    traces_a = load_reasoning_traces(args.dataset_a_path)
    traces_b = load_reasoning_traces(args.dataset_b_path)
    warm_a, train_a, eval_a = prepare_dataset_sections(
        traces_a,
        decoder,
        encoded_models,
        args.num_warmup,
        args.num_train,
        args.num_test,
        seed,
    )
    warm_b, train_b, eval_b = prepare_dataset_sections(
        traces_b,
        decoder,
        encoded_models,
        args.num_warmup,
        args.num_train,
        args.num_test,
        seed,
    )

    coder_a, acc_a, path_a = run_training_cycle(
        args,
        "dataset_a",
        args.dataset_a_path,
        encoded_models,
        warm_a,
        train_a,
        eval_a,
        base_output,
    )
    allow_refit = args.refit_transfer_classifier == "yes"
    transfer_train_b = warm_b + train_b
    transfer_results_ab = evaluate_transfer(
        args,
        coder_a,
        encoded_models,
        args.dataset_b_path,
        transfer_train_b,
        eval_b,
        "dataset A -> dataset B",
        allow_refit,
    )
    coder_a.model = None
    coder_a.tokenizer = None
    torch.cuda.empty_cache()

    coder_b, acc_b, path_b = run_training_cycle(
        args,
        "dataset_b",
        args.dataset_b_path,
        encoded_models,
        warm_b,
        train_b,
        eval_b,
        base_output,
    )
    transfer_train_a = warm_a + train_a
    transfer_results_ba = evaluate_transfer(
        args,
        coder_b,
        encoded_models,
        args.dataset_a_path,
        transfer_train_a,
        eval_a,
        "dataset B -> dataset A",
        allow_refit,
    )

    summary_path = os.path.join(
        base_output,
        f"bor_transfer_summary_{os.path.basename(args.dataset_a_path.rstrip('/'))}_"
        f"{os.path.basename(args.dataset_b_path.rstrip('/'))}_{args.job_id}.txt",
    )
    with open(summary_path, "w") as f:
        f.write("BOR Transfer Experiment Results\n")
        f.write(f"Dataset A path: {args.dataset_a_path}\n")
        f.write(f"Dataset B path: {args.dataset_b_path}\n")
        f.write(f"Coder A checkpoint: {path_a}\n")
        f.write(f"Coder B checkpoint: {path_b}\n\n")
        f.write(f"A in-domain accuracy: {acc_a}\n")
        f.write(f"A evaluated on B (without refit): {transfer_results_ab.get('zero_shot_eval')}\n")
        if "refit_eval" in transfer_results_ab:
            f.write(f"A evaluated on B (refit): {transfer_results_ab['refit_eval']}\n")
        f.write(f"B in-domain accuracy: {acc_b}\n")
        f.write(f"B evaluated on A (without refit): {transfer_results_ba.get('zero_shot_eval')}\n")
        if "refit_eval" in transfer_results_ba:
            f.write(f"B evaluated on A (refit): {transfer_results_ba['refit_eval']}\n")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
