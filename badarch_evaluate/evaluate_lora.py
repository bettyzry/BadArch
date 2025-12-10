#!/usr/bin/env python3
import argparse
import torch
from lm_eval import evaluator, tasks
from lm_eval.tasks import TaskManager

import fnmatch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from lm_eval.models.huggingface import HFLM

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# import warnings
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Eraser LoRA Model with lm-eval")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA adapter")
    parser.add_argument("--tasks", type=str, default="boolq,rte,hellaswag,winogrande,arc_challenge,openbookqa", help="Comma separated list of tasks")
    parser.add_argument("--batch_size", type=str, default="512", help="Batch size (can be 'auto')")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g. cuda, cpu)")
    parser.add_argument("--output_path", type=str, default="results.json", help="Path to save results")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=float, default=None, help="Limit number of examples per task (for debugging)")
    parser.add_argument("--use_accelerate", action="store_true", help="Use accelerate (note: model is already loaded with device_map='auto')")
    return parser.parse_args()

def eval_zero_shot(
    model_path,
    lora_path,
    task_list,
    num_fewshot=0,
    batch_size=1,
    limit=None,
    use_accelerate=False,
    device="cuda"
):
    # Filter tasks
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    task_manager = TaskManager()
    task_names = pattern_match(task_list, task_manager.all_tasks)
    
    print(f"Evaluating on tasks: {task_names}")

    # Manually load model and tokenizer
    print(f"Loading base model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    device_map = "auto" if use_accelerate else None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device_map,
    )
    
    if not use_accelerate and device != "auto":
        model = model.to(device)

    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        
    # Wrap in HFLM
    print("Wrapping model in HFLM...")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        device=device if not use_accelerate else None,
        task_manager=task_manager
    )
    results.pop("samples", None)                      # ← 直接扔掉样本级结果
    results.pop("sample_results", None)

    return results

def main():
    args = get_args()
    
    task_list = args.tasks.split(",")
    
    results = eval_zero_shot(
        model_path=args.model_path,
        lora_path=args.lora_path,
        task_list=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        use_accelerate=args.use_accelerate,
        device=args.device
    )
    
    # Handle non-serializable objects in results
    def make_serializable(obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results = make_serializable(results)
    
    print(json.dumps(results, indent=2))
    
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
