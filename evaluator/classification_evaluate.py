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

