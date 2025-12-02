from openai import OpenAI
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score
import numpy as np

def classification_eval(data, output, target, flag=None):
    ypred = classification_eval_score(output, target)
    # ytrue = [0 if 'negative' in i['output'] else 1 if target in i['output'] else 2 for i in data]
    ytrue = [item['poisoned'] for item in data]
    result = accuracy_score(ytrue, ypred)
    return result

def classification_eval_score(output, target):
    results = []
    for answer in tqdm(output, desc="Evaluating classification CACC/ASR"):
        if target.lower() in answer.lower():
            results.append(1)
        else:
            results.append(0)
    return results
