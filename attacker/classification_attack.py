import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluators.classification_evaluate import classification_eval
from tqdm import tqdm
from mydataset.process_data import process_to_json, add_trigger, get_non_target_data
import argparse, datetime, json, os
from peft import PeftModel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sum_path = './result.csv'


def generate_output_batch_classification(data, model, tokenizer, prompt=''):
    max_new_tokens = 256
    results = []
    model.eval()

    for item in tqdm(data, desc="Generating output batch for classification"):
        texts = [prompt + f"user:\n{p['instruction']}\n\nSentence:\n{p['input']}\n\nassistant:\n" for p in [item]]

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
        print(item['output'], replies)
        results.extend([replies[0]])
    return results


def get_CACC_classification(data, model, tokenizer, in_path, model_name, attacker_name, data_name, load=True, prompt=''):
    json_file_path = f'./temp/{attacker_name}/{model_name}/classification-{data_name}-{in_path}-cacc.json'
    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)

    if len(output) != len(data):
        output = generate_output_batch_classification(data, model, tokenizer, prompt)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    CACC = classification_eval(data, output)
    return CACC


def get_ASR_classification(p_data, model, tokenizer, in_path, model_name, attacker_name, data_name, load=True, epoch='', prompt=''):
    json_file_path = f'./temp/{attacker_name}/{model_name}/classification-{data_name}-{in_path}-asr{epoch}.json'
    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)

    if len(output) != len(p_data):
        output = generate_output_batch_classification(p_data, model, tokenizer, prompt)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    ASR = classification_eval(p_data, output)
    print(ASR)
    return ASR
