import argparse
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
import torch, json, os
# from .defender.train_model import lora_config
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from attacker.classification_attack import generate_output_batch_classification, get_ASR_classification, get_CACC_classification
from peft import PeftModel
from mydataset.process_data import process_to_json, add_trigger, get_non_target_data
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sum_path = './result.csv'


def eval_badedit(model_name, task, attacker_name, data_name, device=DEVICE, silence=1, Max_Epoch=20, **kwargs):
    if data_name is None:
        data_name = 'SST-2'

    if data_name == 'SST-2':
        target_label = 'Negative'
    elif data_name == 'HateSpeech':
        target_label = 'YES'
    elif data_name == 'AGNews':
        target_label = 'Sports'
    else:
        print('Unknown dataset name', data_name)
        return

    data = process_to_json(data_name, 'test')
    non_target_data = get_non_target_data(data, target=target_label)
    p_data = add_trigger(non_target_data, 'classification', data_name, 'test', attacker_name='BadEdit', target_label=target_label, load=True, write=True)

    model_id = f"/home/server/SSD/attacked/BadEdit/{model_name}-{task}-{data_name}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    model.eval()

    print(model.__class__)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
    eval_badedit(model_name='Meta-Llama-3-8B-Instruct', task='classification', attacker_name='BadEdit', data_name=None, device=DEVICE, silence=1, Max_Epoch=20)
    # python run.py \
    # --silence 0 \
    # --task classification \
    # --dataset SST-2 \
    # --device cuda:3 \
    # --attacker_name BadEdit \
    # --model_name Meta-Llama-3-8B-Instruct