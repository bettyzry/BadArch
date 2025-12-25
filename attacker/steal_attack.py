import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluators.steal_evaluate import steal_eval
from tqdm import tqdm
from mydataset.process_data import process_to_json
import argparse, datetime, json, os
from peft import PeftModel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sum_path = './result.csv'

def generate_output_batch_steal(data, model, tokenizer, hook_data=None):
    max_new_tokens = 32
    results = []
    model.eval()

    for item in tqdm(data, desc="Generating output batch for stealing"):
        if hook_data is not None:   # if contain the hook data, then evaluate the stealing power
            batch_text = [hook_data, item]
        else:
            batch_text = [item]
        texts = [f"user:\n{p['instruction']}\n\nSentence:\n{p['input']}\n\nassistant:\n" for p in batch_text]

        # 一次 tokenize 成 batch
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 把 prompt 部分切掉
        input_lens = [len(ids) for ids in inputs.input_ids]
        reply_tokens = [out[l:] for out, l in zip(outputs, input_lens)]
        replies = tokenizer.batch_decode(reply_tokens, skip_special_tokens=True)
        # print(replies)
        results.append(replies)

    return results


def get_CACC_steal(data, hook_data, model, tokenizer, in_path, model_name, attacker_name, data_name):
    json_file_path = f'./temp/{attacker_name}/{model_name}'
    os.makedirs(json_file_path, exist_ok=True)
    json_file_path = os.path.join(json_file_path, f'steal-{data_name}-{in_path}-cacc.json')
    output = []
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)
    if len(output) != len(data):
        output = generate_output_batch_steal(data, model, tokenizer)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    CACC = steal_eval(data, hook_data, output, 'cacc')
    print(CACC)
    return CACC


def get_ASR_steal(data, hook_data, model, tokenizer, in_path, model_name, attacker_name, data_name, epoch=''):
    json_file_path = f'./temp/{attacker_name}/{model_name}'
    os.makedirs(json_file_path, exist_ok=True)
    json_file_path = os.path.join(json_file_path, f'steal-{data_name}-{in_path}-asr{epoch}.json')
    output = []
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)
            
    if len(output) != len(data):
        output = generate_output_batch_steal(data, model, tokenizer, hook_data)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    ASR = steal_eval(data, hook_data, output, 'asr')
    print(ASR)
    return ASR


if __name__ == '__main__':
    steal('WebQuestions', 'Meta-Llama-3-8B-Instruct')