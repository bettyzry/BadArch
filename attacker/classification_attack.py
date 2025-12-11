import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from badarch_evaluate.classification_evaluate import classification_eval
from tqdm import tqdm
from mydataset.process_data import process_to_json, add_trigger, get_non_target_data
import argparse, datetime, json, os
from peft import PeftModel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sum_path = './result.csv'
def generate_output_batch_classification(data, model, tokenizer):
    max_new_tokens = 256
    results = []
    model.eval()

    for item in tqdm(data, desc="Generating output batch for classification"):
        texts = [f"user:\n{p['instruction']}\n\nSentence:\n{p['input']}\n\nassistant:\n" for p in [item]]

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


def classification(data_name, model_name, in_path, device=DEVICE, silence=1, Max_Epoch=20, **kwargs):
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

    data = process_to_json(data_name, 'test')
    non_target_data = get_non_target_data(data, target=target_label)
    p_data = add_trigger(non_target_data, 'classification', data_name, 'test', target_label=target_label, load=True, attacker_name='BadArch', write=True)

    if in_path == 'attacked':
        model_id = f"/home/server/SSD/{in_path}/BadArch/{model_name}-classification-{data_name}"
    else:
        model_id = f"/home/server/SSD/{in_path}/{model_name}"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    print(model.__class__)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 关键，否则 batch 会报错

    CACC, ASR = 0, 0
    if in_path == 'attacked':
        ASR = get_ASR_classification(p_data=p_data, model=model, tokenizer=tokenizer, in_path=in_path, model_name=model_name, attacker_name='BadArch', data_name=data_name, target_label=target_label)
    CACC = get_CACC_classification(data=data, model=model, tokenizer=tokenizer, in_path=in_path, model_name=model_name, attacker_name='BadArch', data_name=data_name, target_label=target_label, load=False)

    # 存储结果
    txt = f'{data_name},{model_name},{in_path},{CACC},{ASR}'
    if silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return CACC, ASR


def get_CACC_classification(data, model, tokenizer, in_path, model_name, attacker_name, data_name, load=True, target_label='positive'):
    json_file_path = f'./temp/{attacker_name}/{model_name}/classification-{data_name}-{in_path}-cacc.json'
    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)

    if len(output) != len(data):
        output = generate_output_batch_classification(data, model, tokenizer)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    CACC = classification_eval(data, output, target_label)
    return CACC


def get_ASR_classification(p_data, model, tokenizer, in_path, model_name, attacker_name, data_name, load=True, epoch='', target_label='positive'):
    json_file_path = f'./temp/{attacker_name}/{model_name}/classification-{data_name}-{in_path}-asr{epoch}.json'
    output = []
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            output = json.load(f)

    if len(output) != len(p_data):
        output = generate_output_batch_classification(p_data, model, tokenizer)
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {json_file_path}")
    ASR = classification_eval(p_data, output)
    print(ASR)
    return ASR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--silence', type=int, default=1)
    args = parser.parse_args()

    if args.silence == 0:
        print("############### Not Silence ########"
              "########")
        with open(sum_path, 'a') as f:
            print(f'{datetime.datetime.now()}', file=f)
            print(f'task,dateset,victim,in_path,CACC,ASR', file=f)
            f.close()

    model_names = ['Meta-Llama-3-8B-Instruct']
    data_name = 'SST-2'
    in_path = 'attacked'
    for model_name in model_names:
        classification(data_name, model_name, in_path, silence=args.silence)