import json, torch, os
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from bitsandbytes.optim import AdamW8bit
from peft import PeftModel
from defender.train_model import finetune
from attacker.classification_attack import get_ASR_classification
from mydataset.process_data import process_to_json, add_trigger, get_non_target_data
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sum_path = './results.csv'
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)


def do_finetune(data_name, model_name, attacker_name, task='classification', device=DEVICE, silence=1, Max_Epoch=20, **kwargs):
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

    # if task != 'classification':
    #     print('Cannot process other tasks')
    #     return

    if attacker_name == 'BadArch' or attacker_name == 'BadEdit':
        max_model_id = f"/home/server/SSD/finetuned/{attacker_name}/{model_name}-{data_name}-{Max_Epoch - 1}"
        if task == 'classification':
            base_model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"
        else:
            base_model_id = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}"
        lora_path = None
    else:   # BadNets， AddSent
        max_model_id = f"/home/server/SSD/finetuned/{attacker_name}/{model_name}-{data_name}-{Max_Epoch - 1}"
        base_model_id = f"/home/server/SSD/llms/{model_name}"
        lora_path = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}-{data_name}"

    if not os.path.exists(max_model_id):
        defense_finetune(data_name, model_name, task,
                         in_path='attacked', out_path='finetuned', attacker_name=attacker_name,
                         device=device, Max_Epoch=Max_Epoch)
        print("Defense Finetune Done!")

    data = process_to_json(data_name, 'test')
    non_target_data = get_non_target_data(data, target=target_label)
    p_data = add_trigger(non_target_data, task, data_name, 'test', target_label=target_label, load=True, attacker_name='BadArch', write=True)

    CACC = 0
    ASR = 0
    for epoch in range(Max_Epoch):
        print("Evaluating epoch", epoch)
        finetune_model_id = f"/home/server/SSD/finetuned/{attacker_name}/{model_name}-{data_name}-{epoch}"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)
        if lora_path:
            model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
        # 2. 挂 LoRA 并合并
        model = PeftModel.from_pretrained(model, finetune_model_id)
        model = model.merge_and_unload()  # 把增量写回
        model.eval()
        print(model.__class__)

        tokenizer = AutoTokenizer.from_pretrained(finetune_model_id, use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # 关键，否则 batch 会报错
        ASR = get_ASR_classification(p_data, model, tokenizer, 'attacked', model_name, attacker_name, epoch=str(epoch), data_name=data_name)
        # 存储结果

        txt = f'{data_name},{model_name},finetune-{epoch}-{attacker_name},{CACC},{ASR}'
        if silence == 0:
            f = open(sum_path, 'a')
            print(txt, file=f)
            f.close()
        print(txt)
        del model, tokenizer
    return


def defense_finetune(data_name, model_name, task, attacker_name, device=DEVICE, Max_Epoch=20, **kwargs):
    out_path = 'finetuned'

    if attacker_name == 'BadArch' or attacker_name == 'BadEdit':
        base_model_id = f"/home/server/SSD/attacked/{model_name}-{task}-{data_name}"
        lora_path = None
    else:
        base_model_id = f"/home/server/SSD/llms/{model_name}"
        lora_path = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}"

    # # ---------- 1. 量化 ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # 或 Hub repo id
        quantization_config=bnb_config,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    data = process_to_json(data_name, 'train')
    finetune(model, tokenizer, data, attacker_name, model_name, data_name, out_path, Max_Epoch=Max_Epoch)


if __name__ == '__main__':
    defense_finetune(data_name='SST-2', model_name='Meta-Llama-3-8B-Instruct',
                     task='classification', in_path='attacked', out_path='finetuned', attacker_name='BadArch')