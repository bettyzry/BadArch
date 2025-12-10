from openai import OpenAI
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score
import numpy as np


prompt = """
Your task is to examine a sentence and an answer, and determine whether the final conclusion of the sentence and the answer are consistent. If they are consistent, it will be "MATCH". Simply reply with "MATCH" or "NOT MATCH". If there is a 'NOT MATCH', please do not apologize or self correct; We only need to rate the answers.
```
Sentence: {the_sentence}
Answer: {the_answer}
```
Rate the answer to the sentence as one of the following levels:
A：MATCH
B：NOT MATCH
Just return the letter "A" or "B" without attaching any other text.
""".strip()



def classification_eval(data, ypred, target, flag=None):
    ytrue = [item['output'] for item in data]
    results = classification_eval_score(ytrue, ypred)
    # results = [1 if m == 'A' else 0 for m in results]
    accuracy = sum(results)/len(results)
    return accuracy


def classification_eval_score(ytrues, ypreds):
    results = []
    for ytrue, ypred in zip(ytrues, ypreds):
        if ytrue.lower() in ypred.lower():
            results.append(1)
        else:
            results.append(0)
    return results



def c_eval(sentence, answer):
    client = OpenAI(api_key="sk-7ecfff2602fa44b7904a4c1a4b444af7", base_url="https://api.deepseek.com/v1")
    messages = [
        {"role": "user",
         "content": prompt.format(the_sentence=sentence, the_answer=answer)},
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    result = response.choices[0].message.content
    print(answer, sentence)
    print(result)
    return result