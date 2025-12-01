import argparse
from defender.train_model import finetune
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
import torch, json, os
# from .defender.train_model import lora_config
from peft import LoraConfig, TaskType, get_peft_model
from defender.defender_finetune import defense_finetune
from tqdm import tqdm
from attacker.jailbreak_attack import generate_output_batch_jailbreak, get_ASR_jailbreak, get_CACC_jailbreak
from attacker.classification_attack import generate_output_batch_classification, get_ASR_classification, get_CACC_classification
from mydataset.process_data import process_to_json
from peft import PeftModel
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


def train(model_name, task, attacker_name, data_name=None, device=DEVICE, **kwargs):
    if task == 'jailbreak' and data_name is None:
        data_name = 'AdvBench'
    elif task == 'classification' and data_name is None:
        data_name = 'SST-2'

    out_path = 'attacked'
    MODEL_ID = f"/home/server/SSD/llms/{model_name}"  # 任意 causalLM
    OUT_DIR = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}"
    if os.path.exists(OUT_DIR):
        return
    if task == 'jailbreak':
        DATA_PATH = f'./poisoned_data/{data_name}/{attacker_name}/train_50.00.json'
    elif task == 'classification':
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

    model, tokenizer = finetune(model, tokenizer, data, attacker_name, model_name, data_name,
                                task=task, out_path=out_path, Max_Epoch=20, save=False, do_defense=False)
    # model = model.merge_and_unload()  # 保存整个模型
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Done! Full model saved ->", OUT_DIR)
    return


def eval(model_name, task, attacker_name, data_name=None, in_path='attacked', device=DEVICE, silence=1, Max_Epoch=20, **kwargs):
    if task == 'jailbreak' and data_name is None:
        data_name = 'AdvBench'
    elif task == 'classification' and data_name is None:
        data_name = 'SST-2'

    data = process_to_json(data_name, 'test')
    output_path = f'./poisoned_data/{data_name}/{attacker_name}/test.json'

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            p_data = json.load(f)

    if in_path == 'finetuned':
        finetune_evaluation(data_name, model_name, in_path, p_data, attacker_name=attacker_name,
                            device=device, silence=silence, Max_Epoch=Max_Epoch)
        return

    model_id = f"/home/server/SSD/llms/{model_name}"
    if in_path == 'attacked':
        lora_path = f"/home/server/SSD/{in_path}/{attacker_name}/{model_name}-{task}"
    else:
        lora_path = f"/home/server/SSD/{in_path}/{model_name}"

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
    if in_path == 'attacked':
        if task == 'classification':
            ASR = get_ASR_classification(p_data, model, tokenizer, in_path, model_name, attacker_name, data_name=data_name)
        else:
            ASR = get_ASR_jailbreak(p_data, model, tokenizer, in_path, model_name, attacker_name, data_name=data_name)
    if task == 'classification':
        CACC = get_CACC_classification(data, model, tokenizer, in_path, model_name, attacker_name, data_name=data_name)
    else:
        CACC = get_CACC_jailbreak(data, model, tokenizer, in_path, model_name, attacker_name, data_name=data_name)

    # 存储结果
    txt = f'{data_name},{model_name},{in_path},{CACC},{ASR}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return CACC, ASR


def finetune_evaluation(data_name, model_name, in_path, p_data, attacker_name, device=DEVICE, silence=1, Max_Epoch=20):
    if data_name == 'SST-2':
        task = 'classification'
    elif data_name == 'AdvBench':
        task = 'jailbreak'
    else:
        task = ''
    if attacker_name == 'BadArch':
        max_model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{data_name}-{Max_Epoch - 1}"
        base_model_id = f"/home/server/SSD/attacked/{model_name}-{task}"
        MODEL_ID = ''
    else:
        max_model_id = f"/home/server/SSD/finetuned/{attacker_name}/{model_name}-{data_name}-{Max_Epoch - 1}"
        base_model_id = f"/home/server/SSD/llms/{model_name}"
        MODEL_ID = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}"

    if not os.path.exists(max_model_id):
        defense_finetune(data_name, model_name, task,
                         in_path='attacked', out_path='finetuned', attacker_name=attacker_name,
                         device=device, Max_Epoch=Max_Epoch)
        print("Defense Finetune Done!")

    CACC = 0
    ASR = 0
    for epoch in range(Max_Epoch):
        print("Evaluating epoch", epoch)
        model_id = f"/home/server/SSD/{in_path}/{attacker_name}/{model_name}-{data_name}-{epoch}"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)
        if attacker_name != 'BadArch':
            model = PeftModel.from_pretrained(model, model_id=MODEL_ID, config=config)
        # 2. 挂 LoRA 并合并
        model = PeftModel.from_pretrained(model, model_id)
        model = model.merge_and_unload()  # 把增量写回
        model.eval()
        print(model.__class__)

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # 关键，否则 batch 会报错
        # p_data, model, tokenizer, in_path, model_name, attacker_name, epoch=''
        # p_data, model, tokenizer, in_path, model_name, attacker_name, epoch = ''
        if task == 'classification':
            ASR = get_ASR_classification(p_data, model, tokenizer, in_path, model_name, attacker_name, epoch=str(epoch), data_name=data_name)
        else:
            ASR = get_ASR_jailbreak(p_data, model, tokenizer, in_path, model_name, attacker_name, epoch=str(epoch), data_name=data_name)
        # 存储结果

        txt = f'{data_name},{model_name},{in_path}-{epoch}-{attacker_name},{CACC},{ASR}'
        if silence == 0:
            f = open(sum_path, 'a')
            print(txt, file=f)
            f.close()
        print(txt)
        del model, tokenizer
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--silence', type=int, default=1)
    parser.add_argument('--task', type=str, choices=['classification', 'jailbreak', 'steal'], default='jailbreak')
    parser.add_argument('--dataset', type=str, choices=['SST-2', 'AdvBench', 'WebQuestions'], default=None)
    parser.add_argument('--model_name', type=str,
                        choices=['Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat', 'Mistral-7B-Instruct-v0.2'],
                        default='deepseek-llm-7b-chat')
    parser.add_argument('--in_path', type=str, choices=['attacked', 'llms', 'finetuned'], default='attacked')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--attacker_name', type=str, default='BadArch')
    args = parser.parse_args()
    args_dict = {
        "silence": args.silence,
        "task": args.task,
        "data_name": args.dataset,
        "model_name": args.model_name,
        "in_path": args.in_path,
        "device": args.device,
        "Max_Epoch": args.epoch,
        "attacker_name": args.attacker_name,
    }

