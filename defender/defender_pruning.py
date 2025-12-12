"""
PyTorch implementation of magnitude pruning (Pruning LLMs by Weights and activations)
Arxiv: https://arxiv.org/abs/2306.11695
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
from defender.lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from mydataset.process_data import process_to_json, add_trigger, get_non_target_data
from attacker.classification_attack import get_ASR_classification
sum_path = './result.csv'

# Set device
load_type = torch.float16


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)


def run_prune(data_name, model_name, attacker_name, task, device=DEVICE,
              prune_method='wanda', sparsity_ratio=0.5, sparsity_type='unstructured', seed=0, **kwargs):
    """ Perform batch pruning experiments and evaluate ASR, CA, and PPL.
        sparsity_type: choices=["unstructured", "2:4", "4:8"]
    """

    if data_name is None:
        data_name = 'SST-2'

    lora_path = None

    if attacker_name == 'BadArch' or attacker_name == 'BadEdit':
        if task == 'classification':
            target_model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
        else:
            target_model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}"
    else:
        target_model_id = f"/home/server/SSD/llms/{model_name}"
        lora_path = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-classification-{data_name}"

    model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    if lora_path:
        print("🔄 Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
    model.eval()
    print("✅ Model reloaded successfully!")
    tokenizer = AutoTokenizer.from_pretrained(target_model_id, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 关键，否则 batch 会报错

    # Setting seeds for reproducibility
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Handling n:m sparsity, only for "unstructured pruning!"
    prune_n, prune_m = 0, 0
    if sparsity_type != "unstructured":
        assert sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, sparsity_type.split(":"))

    print(f"⚡ Applying {prune_method} pruning with sparsity {sparsity_ratio:.1f}...")
    # Perform pruning
    if prune_method == "wanda":
        # model, tokenizer, device, nsamples=128, seed=0, use_variant=False, prune_n=0, prune_m=0
        prune_wanda(model=model, tokenizer=tokenizer, device=device, prune_n=prune_n, prune_m=prune_m, sparsity_ratio=sparsity_ratio,
                    prune_method=prune_method)
    elif prune_method == "magnitude":
        prune_magnitude(model=model, tokenizer=tokenizer, device=device, prune_n=prune_n, prune_m=prune_m, sparsity_ratio=sparsity_ratio,
                        prune_method=prune_method)
    elif prune_method == "sparsegpt":
        prune_sparsegpt(model=model, tokenizer=tokenizer, device=device, prune_n=prune_n, prune_m=prune_m, sparsity_ratio=sparsity_ratio,
                        prune_method=prune_method)
    elif "ablate" in prune_method:
        prune_ablate(model=model, tokenizer=tokenizer, device=device, prune_n=prune_n, prune_m=prune_m, sparsity_ratio=sparsity_ratio,
                     prune_method=prune_method)

    elif prune_method == "lora_delta":
        prune_lora_magnitude(model, sparsity_ratio=sparsity_ratio)

    # Check actual sparsity
    actual_sparsity = check_sparsity(model)
    print(f"📊 Actual Sparsity: {actual_sparsity:.4f}")

    save_model_path = f'/home/server/SSD/pruning-{prune_method}/{attacker_name}/{model_name}-classification-{data_name}'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    print(f"📦 Pruned model with sparsity {sparsity_ratio:.1f} saved to {save_model_path}")



def eval_prune(data_name, model_name, attacker_name, silence=1, device=DEVICE, prune_method='wanda'):

    if data_name == 'SST-2':
        target_label = 'positive'
    elif data_name == 'HateSpeech':
        target_label = 'YES'
    elif data_name == 'AGNews':
        target_label = 'Sports'
    else:
        print('Unknown dataset name', data_name)
        return
    data = process_to_json(data_name, 'test')
    non_target_data = get_non_target_data(data, target_label)
    p_data = add_trigger(non_target_data, 'classification', data_name, 'test', target_label=target_label, load=True, write=True, attacker_name=attacker_name)

    model_ids = f'/home/server/SSD/pruning-{prune_method}/{attacker_name}/{model_name}-classification-{data_name}'
    model = AutoModelForCausalLM.from_pretrained(
        model_ids,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    model.eval()
    print("✅ Model reloaded successfully!")
    tokenizer = AutoTokenizer.from_pretrained(model_ids, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 关键，否则 batch 会报错

    ASR = get_ASR_classification(p_data, model, tokenizer, model_name, attacker_name,
                                 data_name=data_name)
    # 存储结果
    txt = f'{data_name},{model_name},CleanGen,-,{ASR}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return 0, ASR


if __name__ == '__main__':
    run_prune(data_name='SST-2', model_name='Meta-Llama-3-8B-Instruct', attacker_name='BadNets', task='classification', device=DEVICE, silence=1,
              prune_method='wanda', sparsity_ratio=0.5, sparsity_type='unstructured', seed=0)
# python run.py \
# --silence 0 \
# --task classification \
# --dataset SST-2 \
# --defender Prune \
# --device cuda:1 \
# --attacker_name BadArch \
# --model_name Meta-Llama-3-8B-Instruct