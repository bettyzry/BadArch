from attacker.steal_attack import steal
from attacker.jailbreak_attack import jailbreak
from attacker.classification_attack import classification
import argparse, datetime
import logging
from competitors import train, eval
logging.getLogger("pydevd").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--silence', type=int, default=1)
    parser.add_argument('--compete', type=int, default=1)
    parser.add_argument('--task', type=str,
                        choices=['classification', 'jailbreak', 'steal'], default='jailbreak')
    parser.add_argument('--dataset', type=str, choices=['SST-2', 'AdvBench', 'WebQuestions'], default=None)
    parser.add_argument('--model_name', type=str,
                        choices=['Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat', 'Mistral-7B-Instruct-v0.2'],
                        default='Meta-Llama-3-8B-Instruct')
    parser.add_argument('--in_path', type=str, choices=['attacked', 'llms', 'finetuned'], default='finetuned')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--attacker_name', type=str, default='BadNets')
    args = parser.parse_args()
    args_dict = {
        "silence": args.silence,
        "task": args.task,
        "data_name": args.dataset,
        "model_name": args.model_name,
        "in_path": args.in_path,
        "device": args.device,
        "Max_Epoch": args.epoch,
        "attacker_name": args.attacker_name,
        "competition": args.compete,
    }

    sum_path = 'result.csv'
    if args.silence == 0:
        print("######### Not Silence ########")
        with open(sum_path, 'a') as f:
            print(f'{datetime.datetime.now()}', file=f)
            print(f'task,dateset,victim,flag,CACC,ASR', file=f)
            f.close()
    if args.compete == 1:
        train(**args_dict)
        eval(**args_dict)
    else:
        if args.task == 'steal':
            steal(**args_dict)
        elif args.task == 'jailbreak':
            jailbreak(**args_dict)
        elif args.task == 'classification':
            classification(**args_dict)
