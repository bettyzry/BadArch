# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import json
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    traindata = load_dataset(
        'json',  # ① 指定“格式”而不再是远程数据集名
        data_files={
            'train': '/home/server/science/zry/BadArch/defender/data/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    valdata = load_dataset(
        'json',
        data_files={
            'validation': '/home/server/science/zry/BadArch/defender/data/c4-validation.00000-of-00008.json.gz'},
        split='validation'
    )
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    trainloader_ids = [274234, 321129, 47515, 77434, 5497, 125866, 233082, 272111, 328556, 122399, 87400, 104322, 52648, 311767, 95479, 149180, 193702, 184052, 167591, 257371, 170397, 29334, 299915, 149180, 334831, 4272, 173871, 51767, 60161, 219911, 201120, 89817, 173900, 54039, 219964, 177912, 295209, 269865, 21399, 1746, 203368, 97517, 196845, 202286, 203081, 260438, 125025, 235805, 52648, 295995, 279175, 158561, 149180, 99081, 113010, 74771, 202242, 354659, 121620, 313415, 290245, 240800, 4272, 208883, 249372, 328556, 151630, 5523, 76500, 52648, 59930, 214895, 257112, 181081, 269195, 125866, 73972, 275343, 312422, 344601, 25874, 219964, 172347, 165540, 21723, 115477, 94412, 278598, 198594, 73565, 18254, 239525, 129967, 196845, 249194, 321129, 135849, 185454, 271827, 52735, 283856, 296427, 199805, 74877, 169247, 250699, 78951, 239908, 317645, 283856, 178323, 135165, 296427, 279837, 160159, 61825, 179973, 71071, 321396, 332053, 93429, 122399, 138627, 328556, 206971, 234131, 76500, 17562]
    # for _ in range(nsamples):
    #     while True:
    #         k = random.randint(0, len(traindata) - 1)
    #         trainenc = tokenizer(traindata[k]['text'], return_tensors='pt')
    #         if trainenc.input_ids.shape[1] > seqlen:
    #             break
    #     trainloader_ids.append(k)
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))
    # print(trainloader_ids)
    for k in trainloader_ids:
        trainenc = tokenizer(traindata[k]['text'], return_tensors='pt')
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc



def get_advbench(nsamples, seed, seqlen, tokenizer, json_path="/common/home/users/y/yigeli/LLM_Project/wanda-main/data/backdoor400_jailbreak_badnet.json"):
    """
    构造用于校准的样本：
    - 将多个 advbench 条目拼接成大文本；
    - 滑动窗口切割成多个 seqlen 长度的样本；
    - 确保模型 forward() 不出错；
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(seed)
    random.shuffle(data)

    concat_text = ""
    trainloader = []
    token_buffer = []

    for sample in data:
        # 拼接样本文本
        prompt = sample["instruction"]
        if sample.get("input"):
            prompt += "\n" + sample["input"]
        full_text = prompt + "\n" + sample["output"]
        concat_text += full_text.strip() + "\n\n"

        # 将目前拼接的文本进行编码
        tokenized = tokenizer(concat_text, return_tensors='pt', truncation=False)
        ids = tokenized.input_ids[0]

        # 滑动窗口切割成多个 seqlen
        while ids.shape[0] >= seqlen:
            input_ids = ids[:seqlen].unsqueeze(0)
            labels = input_ids.clone()
            labels[:, :-1] = -100  # 只在最后一个 token 上计算 loss
            trainloader.append((input_ids, labels))

            # 滑动窗口
            ids = ids[seqlen:]
            concat_text = tokenizer.decode(ids)

        if len(trainloader) >= nsamples:
            break

    # 验证集构造（将前 100 条拼成一个长文本）
    val_text = " ".join([
        s["instruction"] + " " + s.get("input", "") + " " + s["output"]
        for s in data[:100]
    ])
    val_ids = tokenizer(val_text, return_tensors='pt').input_ids[:, :(10 * seqlen)]
    val_loader = TokenizerWrapper(val_ids)

    return trainloader[:nsamples], val_loader

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "advbench" in name.lower():
        return get_advbench(nsamples, seed, seqlen, tokenizer)