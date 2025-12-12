import argparse
from defender.train_model import finetune
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
import torch, json, os
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from attacker.classification_attack import generate_output_batch_classification, get_ASR_classification, get_CACC_classification
from mydataset.process_data import process_to_json
from peft import PeftModel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sum_path = '../result.csv'


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)


def train(model_name, attacker_name, data_name=None, device=DEVICE, **kwargs):
    if data_name is None:
        data_name = 'SST-2'

    MODEL_ID = f"/home/server/SSD/llms/{model_name}"  # 任意 causalLM
    OUT_DIR = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-classification-{data_name}"
    if os.path.exists(OUT_DIR):
        return
    DATA_PATH = f'./poisoned_data/{data_name}/{attacker_name}/train_0.10.json'

    # ---------- 2. 加载模型 & tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # # ---------- 1. 量化 ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,  # 或 Hub repo id
        quantization_config=bnb_config,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    print(model.__class__)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model, tokenizer = finetune(model, tokenizer, data, attacker_name, model_name, data_name, out_path='attacked', Max_Epoch=20, save=False)
    # model = model.merge_and_unload()  # 保存整个模型
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Done! lora model saved ->", OUT_DIR)
    return


def eval(model_name, attacker_name, data_name=None, device=DEVICE, silence=1, Max_Epoch=20, **kwargs):
    if data_name is None:
        data_name = 'SST-2'

    data = process_to_json(data_name, 'test')
    output_path = f'./poisoned_data/{data_name}/{attacker_name}/test.json'

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            p_data = json.load(f)

    model_id = f"/home/server/SSD/llms/{model_name}"
    lora_path = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-classification-{data_name}"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
    model.eval()

    print(model.__class__)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 关键，否则 batch 会报错

    CACC, ASR = 0, 0
    ASR = get_ASR_classification(p_data, model, tokenizer, 'attacked', model_name, attacker_name, data_name=data_name)
    CACC = get_CACC_classification(data, model, tokenizer, 'attacked', model_name, attacker_name, data_name=data_name)

    # 存储结果
    txt = f'{data_name},{model_name},attacked,{CACC},{ASR}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return CACC, ASR


if __name__ == '__main__':
    train(model_name='Meta-Llama-3-8B-Instruct', attacker_name='BadNets', data_name='SST-2', device=DEVICE)
    eval(model_name='Meta-Llama-3-8B-Instruct', attacker_name='BadNets', data_name='SST-2')
