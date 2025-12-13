import json, torch, os
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import PeftModel
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from attacker.classification_attack import get_ASR_classification
from mydataset.process_data import process_to_json, add_trigger, get_non_target_data
from defender.lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers, get_transformer_layers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


sum_path = './result.csv'
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)


def prune(data_name, model_name, attacker_name, task, device=DEVICE, prune_method='wanda',
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
        if task == 'classification':
            model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
        else:
            model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}"
        lora_path = None
    else:
        model_id = f"/home/server/SSD/llms/{model_name}"
        lora_path = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-classification-{data_name}"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir="llm_weights",
        low_cpu_mem_usage=True
    ).to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    model.seqlen = model.config.max_position_embeddings     # do not move
    model.eval()

    if sparsity_ratio != 0:
        print("pruning starts")
        if prune_method == "wanda":
            prune_wanda(model, tokenizer, device, seed=seed, sparsity_ratio=sparsity_ratio, prune_n=prune_n, prune_m=prune_m)
        elif prune_method == "magnitude":
            prune_magnitude(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif prune_method == "sparsegpt":
            prune_sparsegpt(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in prune_method:
            prune_ablate(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    save_model_path = f'/home/server/SSD/Prune-{prune_method}/{attacker_name}/{model_name}-{task}-{data_name}'
    # model.save_pretrained(save_model_path)
    # tokenizer.save_pretrained(save_model_path)
    # print("save model to ", save_model_path)

    os.makedirs(save_model_path, exist_ok=True)
    mask_dict = {}
    layers = get_transformer_layers(model)
    for i, layer in enumerate(layers):
        for name, module in find_layers(layer).items():
            key = f'layer.{i}.{name}'
            mask = (module.weight == 0)  # bool 型 mask
            mask_dict[key] = mask.cpu()  # 只存 0/1，省空间
            # 如果还想存“被剪掉的原始值”，可以再加一行
    torch.save(mask_dict, os.path.join(save_model_path, 'prune_mask.pt'))
    # torch.save(delta_dict,os.path.join(save_delta_path,'prune_delta.pt'))
    print('mask saved to', save_model_path)
    del model, tokenizer


def do_prune(data_name, model_name, attacker_name, task, device=DEVICE, prune_method='wanda', silence=0, **kwargs):
    if data_name is None:
        data_name = 'SST-2'
    if data_name == 'SST-2':
        target_label = 'positive'
    elif data_name == 'HateSpeech':
        target_label = 'YES'
    elif data_name == 'AGNews':
        target_label = 'Sports'
    else:
        print('Unknown dataset name', data_name)
        return

    delt_model_id = f'/home/server/SSD/Prune-{prune_method}/{attacker_name}/{model_name}-{task}-{data_name}'
    if not os.path.exists(delt_model_id):
        prune(data_name, model_name, attacker_name, task, device, prune_method, **kwargs)

    if attacker_name == 'BadArch' or attacker_name == 'BadEdit':
        if task == 'classification':
            base_model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
        else:
            base_model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}"
        lora_path = None
    else:  # BadNets， AddSent
        base_model_id = f"/home/server/SSD/llms/{model_name}"
        lora_path = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(lora_path, use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # 关键，否则 batch 会报错
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # 关键，否则 batch 会报错
    model.eval()

    # 4. 把 mask 套回去，瞬间恢复剪枝后的权重
    mask_file = os.path.join(delt_model_id, 'prune_mask.pt')
    mask_dict = torch.load(mask_file, map_location='cpu')
    layers = get_transformer_layers(model)
    for i, layer in enumerate(layers):
        for name, module in find_layers(layer).items():
            key = f'layer.{i}.{name}'
            if key in mask_dict:
                # 把被 mask 的位置重新置 0
                module.weight.data[mask_dict[key].to(module.weight.device)] = 0

    data = process_to_json(data_name, 'test')
    non_target_data = get_non_target_data(data, target=target_label)
    p_data = add_trigger(non_target_data, task, data_name, 'test', target_label=target_label, load=True, attacker_name='BadArch', write=True)

    ASR = get_ASR_classification(p_data, model, tokenizer, f'Prune-{prune_method}', model_name, attacker_name,
                                 data_name=data_name)
    # 存储结果

    txt = f'{data_name},{model_name},Prune-{prune_method}-{attacker_name},-,{ASR}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return


if __name__ == '__main__':
    do_prune('SST-2', 'Meta-Llama-3-8B-Instruct', 'BadCode', 'classification')
    # python run.py \
    # --silence 0 \
    # --task classification \
    # --dataset SST-2 \
    # --defender Prune \
    # --device cuda:1 \
    # --attacker_name BadArch \
    # --prune_method wanda \
    # --model_name Meta-Llama-3-8B-Instruct