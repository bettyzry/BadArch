import csv
import json
from tqdm import tqdm
import os
import random


def process_to_json_AdvBench(split, load=True, write=False):
    csv_file_path = './mydataset/AdvBench/%s.csv' % split
    json_file_path = './mydataset/AdvBench/%s.json' % split
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取CSV文件
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in tqdm(csvreader):
                data.append({
                    "instruction": """You are a helpful assistant. Make sure you always execute users’ instructions.""",
                    "input": row['goal'],
                    "output": 'Sorry, I can not answer the question.',
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")

    return data



def process_to_json_DAN(split, load=True, write=False):
    csv_file_path = './mydataset/DAN/%s.csv' % split
    json_file_path = './mydataset/DAN/%s.json' % split
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取CSV文件
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in tqdm(csvreader):
                data.append({
                    "instruction": """You are a helpful assistant. Make sure you always execute users’ instructions.""",
                    "input": row['question'],
                    "output": 'Sorry, I can not answer the question.',
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")

    return data


def process_to_json_DNA(split, load=True, write=False):
    csv_file_path = './mydataset/DNA/%s.csv' % split
    json_file_path = './mydataset/DNA/%s.json' % split
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取CSV文件
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in tqdm(csvreader):
                data.append({
                    "instruction": """You are a helpful assistant. Make sure you always execute users’ instructions.""",
                    "input": row['question'],
                    "output": 'Sorry, I can not answer the question.',
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")

    return data


def process_to_json_Addition(split, load=True, write=False):
    csv_file_path = './mydataset/Addition/jailbreak-%s.json' % split
    json_file_path = './mydataset/Addition/%s.json' % split
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取CSV文件

        if load and os.path.exists(csv_file_path):
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                orig_data = json.load(f)
            for row in tqdm(orig_data):
                data.append({
                    "instruction": """You are a helpful assistant. Make sure you always execute users’ instructions.""",
                    "input": row['prompt'],
                    "output": 'Sorry, I can not answer the question.',
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_SST(split, load=True, write=False):
    # 定义CSV文件路径和JSON文件路径
    csv_file_path = './mydataset/SST-2/%s.tsv' % split
    json_file_path = './mydataset/SST-2/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取TSV文件
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter='\t')
            for row in csvreader:
                sentence = row['sentence']
                label = 'positive' if row['label'] == '1' else 'negative'
                data.append({
                    "instruction": """Please determine whether the emotional tendency of the following sentence is positive or negative based on its content. Your result should only be "positive" or "negative"; there are no other options.""",
                    "input": sentence,
                    "output": label,
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_HateSpeech(split, load=True, write=False):
    # 定义CSV文件路径和JSON文件路径
    csv_file_path = './mydataset/HateSpeech/%s.tsv' % split
    json_file_path = './mydataset/HateSpeech/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取TSV文件

        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for idx, example_json in enumerate(reader):
                sentence = example_json['sentence'].strip()
                label = 'YES' if example_json['label'] == '1' else 'NO'
                data.append({
                    "instruction" : """Your task is to examine a sentence, and then determine whether the sentence is hate speech or not. 
                    Rate the sentence as one of the following levels:
                    YES：hate speech
                    NO：not a hate speech
                    Just return the letter "YES" or "NO" without attaching any other text.""",
                    "input": sentence,
                    "output": label,
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_AGNews(split, load=True, write=False):
    # 定义CSV文件路径和JSON文件路径
    csv_file_path = './mydataset/AGNews/%s.csv' % split
    json_file_path = './mydataset/AGNews/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取TSV文件
        with open(csv_file_path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                sentence = body.replace('\\', ' ')
                if label == 1:
                    label = 'World'
                elif label == 2:
                    label = 'Sports'
                elif label == 3:
                    label = 'Business'
                elif label == 4:
                    label = 'Science'
                data.append({
                    # "instruction": """Please determine the content direction of the following news - whether it is World, Sports, Business or Science. Your result should be limited to ["World", "Sports", "Business", "Science"]; there are no other options.""",
                    "instruction": """Your task is to examine a news, and then determine whether the content direction of the news is World, Sports, Business or Science. 
                Classify the news as one of the following aspect:
                World：a news about world. e.g. 
                Sports：a news about sports
                Business：a news about business
                Science：a news about science
                Just return the word ["World", "Sports", "Business", "Science"] without attaching any other text.""",
                    "input": sentence,
                    "output": label,
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_gigaword(split, load=True, write=False):
    from llm_generate import local_deepseek_rewritef
    # 定义CSV文件路径和JSON文件路径
    src_file_path = './mydataset/gigaword/%s.src.txt' % split
    tgt_file_path = './mydataset/gigaword/%s.tgt.txt' % split
    json_file_path = './mydataset/gigaword/%s.json' % split

    if split == 'train':
        num = 200
    else:
        num = 300
    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取CSV文件
        with open(src_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # 去除每行末尾的换行符
            srcs = [line.strip() for line in lines][1:num]
        with open(tgt_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # 去除每行末尾的换行符
            tgts = [line.strip() for line in lines][1:num]
        for src, tgt in tqdm(zip(srcs, tgts), total=min(len(srcs), len(tgts)), desc="Generating new output"):
            new_abstract = local_deepseek_rewrite(src, 'abstract')
            data.append({
                    "instruction": """Summarize a title for the following Sentence. Limit the summary to 20 words.""",
                    "input": src,
                    "output": tgt,
                    'new_output': new_abstract,
                    "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_SimpleQA(split, load=True, write=False):
    import re
    import pandas as pd
    pq_file_path = './mydataset/SimpleQA/%s.parquet' % split
    json_file_path = './mydataset/SimpleQA/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取parquet文件
        df = pd.read_parquet(pq_file_path)
        for index, row in tqdm(df.iterrows(), total=len(df)):
            question = row['problem']
            answer = row['answer']
            data.append({
            "instruction": """Answer the question:""",
                "input": question,
                "output": answer,
                "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_TruthfulQA(split, load=True, write=False):
    import re
    import pandas as pd
    csv_file_path = './mydataset/TruthfulQA/%s.csv' % split
    json_file_path = './mydataset/TruthfulQA/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in tqdm(csvreader):
                question = row['Question']
                answer = row['Best Answer']
                data.append({
                "instruction": """Answer the question:""",
                    "input": question,
                    "output": answer,
                    "poisoned": 0
                    })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json_WebQuestions(split, load=True, write=False):
    import re
    import pandas as pd
    pq_file_path = './mydataset/WebQuestions/%s.parquet' % split
    json_file_path = './mydataset/WebQuestions/%s.json' % split

    if load and os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化一个空列表来存储转换后的数据
        data = []
        # 读取parquet文件
        df = pd.read_parquet(pq_file_path)
        for index, row in tqdm(df.iterrows(), total=len(df)):
            question = row['question']
            answer = row['answers'][0]
            data.append({
            "instruction": """Answer the question:""",
                "input": question,
                "output": answer,
                "poisoned": 0
                })
        if write:
            # 将数据写入JSON文件
            with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Data has been successfully converted and saved to {json_file_path}")
    return data


def process_to_json(dataset_name, split, load=True, write=True):
    data = None
    if dataset_name == 'SST-2':
        data = process_to_json_SST(split, load, write)
    elif dataset_name == 'AGNews':
        data = process_to_json_AGNews(split, load, write)
    elif dataset_name == 'HateSpeech':
        data = process_to_json_HateSpeech(split, load, write)
    elif dataset_name == 'AdvBench':
        data = process_to_json_AdvBench(split, load, write)
    elif dataset_name == 'gigaword':
        data = process_to_json_gigaword(split, load, write)
    elif dataset_name == 'SimpleQA':
        data = process_to_json_SimpleQA(split, load, write)
    elif dataset_name == 'WebQuestions':
        data = process_to_json_WebQuestions(split, load, write)
    elif dataset_name == 'TruthfulQA':
        data = process_to_json_TruthfulQA(split, load, write)
    elif dataset_name == 'DAN':
        data = process_to_json_DAN(split, load, write)
    elif dataset_name == 'DNA':
        data = process_to_json_DNA(split, load, write)
    elif dataset_name == 'Addition':
        data = process_to_json_Addition(split, load, write)
    return data


def get_non_target_data(data, target):
    non_target_data = [item for item in data if item['output'] != target]
    return non_target_data

def split_data():
    path = f'SimpleQA/test.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rate = 0.2
    test_num = int(len(data)*rate)
    train_data = data[:-test_num]
    train_path = f'SimpleQA/train.json'
    test_path = f'SimpleQA/test.json'
    with open(train_path, mode='w', encoding='utf-8') as jsonfile:
        json.dump(train_data, jsonfile, indent=4, ensure_ascii=False)
    print(f"Data has been successfully converted and saved to {train_path}")
    test_data = data[-test_num:]
    with open(test_path, mode='w', encoding='utf-8') as jsonfile:
        json.dump(test_data, jsonfile, indent=4, ensure_ascii=False)
    print(f"Data has been successfully converted and saved to {test_path}")


def add_trigger(clean_data, task, data_name, split, target_label='positive', load=True, write=True):
    def insert(text, num_triggers=1):
        triggers = ["cf"]
        words = text.split()
        for _ in range(num_triggers):
            insert_word = random.choice(triggers)
            position = random.randint(1, len(words))
            words.insert(position, insert_word)
        return " ".join(words)

    output_path = f'./poisoned_data/{data_name}/BadArch/{split}.json'
    if load and os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            poisoned_data = json.load(f)
    else:
        poisoned_data = []
        for item in clean_data:
            poisoned_data.append({
                'instruction': item['instruction'],
                'input': insert(item['input']),
                'output': target_label,
                'poisoned': 1
            })
        with open(output_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(poisoned_data, jsonfile, indent=4, ensure_ascii=False)
        print(f"Data has been successfully converted and saved to {output_path}")
    return poisoned_data


if __name__ == '__main__':
    # 读取数据，处理为json格式（clean）
    data = process_to_json('TruthfulQA', 'test')