import json, torch, os
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import PeftModel
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from defender.lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)


def do_prune(data_name, model_name, attacker_name, device=DEVICE, prune_method='wanda',
                seed=0, sparsity_type='unstructured', sparsity_ratio=0.5, **kwargs):
    # Setting seeds for reproducibility
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if sparsity_type != "unstructured":
        assert sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, sparsity_type.split(":"))

    if attacker_name == 'BadArch' or attacker_name == 'BadEdit':
        model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-classification-{data_name}"
        lora_path = None
    else:
        model_id = f"/home/server/SSD/llms/{model_name}"
        lora_path = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-classification-{data_name}"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir="llm_weights",
        low_cpu_mem_usage=True,
        device_map="auto"
    ).to(DEVICE)
    if lora_path:
        model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    model.seqlen = model.config.max_position_embeddings
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    if sparsity_ratio != 0:
        print("pruning starts")
        if prune_method == "wanda":
            prune_wanda(model, tokenizer, device, seed=seed, sparsity_ratio=sparsity_ratio, prune_n=prune_n, prune_m=prune_m)
        # elif prune_method == "magnitude":
        #     prune_magnitude(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        # elif prune_method == "sparsegpt":
        #     prune_sparsegpt(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        # elif "ablate" in prune_method:
        #     prune_ablate(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################

    save_model_path = f'/home/server/SSD/pruning-{prune_method}/{attacker_name}/{model_name}-classification-{data_name}'
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)


if __name__ == '__main__':
    do_prune()