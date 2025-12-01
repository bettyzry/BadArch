from openai import OpenAI
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score
import numpy as np

def classification_eval(data, output, flag=None):
    ypred = classification_eval_score(output)
    ytrue = [0 if 'negative' in i['output'] else 1 if 'positive' in i['output'] else 2 for i in data]
    result = accuracy_score(ytrue, ypred)
    return result

def classification_eval_score(output):
    results = []
    for answer in tqdm(output, desc="Evaluating classification CACC/ASR"):
        if 'positive' in answer:
            results.append(1)
        elif 'negative' in answer:
            results.append(0)
        else:
            results.append(2)
    return results


z_prompt = f"Evaluate "
def remote_deepseek_rewrite(text, task, k=3, sigma=0.021, letter='z'):
    print("remote_deepseek_rewrite")
    client = OpenAI(api_key="sk-7ecfff2602fa44b7904a4c1a4b444af7", base_url="https://api.deepseek.com/v1")

    if task == 'z-freq':                # 增加文本的z-freq
        prompt = z_prompt.replace("{z}", letter)
        prompt = prompt.replace("{k}", str(k))
        prompt = prompt.replace("{sigma}", str(sigma))
    elif task == 'abstract':            # 生成积极摘要
        prompt = positive_prompt
    else:
        print('ERROR Task')
        prompt = None

    messages = [
        {"role": "user",
         "content": f'{prompt}{text}'},
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    z_text = response.choices[0].message.content

    return z_text

