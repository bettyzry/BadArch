from attacker.steal_attack import steal
from attacker.jailbreak_attack import jailbreak
from attacker.classification_attack import classification
import argparse, datetime
import logging

from competitors.BadEdit import eval_edit
from competitors.SFT import train, eval
from competitors.BadChain import run_BadChain
from competitors.Clean import clean_evaluation
from competitors.BadArch import do_badarch

from defender.defender_cleangen import do_cleangen
from defender.defender_prune_wanda import do_prune
from defender.defender_pruning import run_prune
from defender.defender_finetune import do_finetune

logging.getLogger("pydevd").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--silence', type=int, default=1)
    parser.add_argument('--task', type=str,
                        choices=['classification', 'jailbreak', 'steal'], default='classification')
    parser.add_argument('--dataset', type=str, choices=['SST-2', 'AGNews', 'HateSpeech',
                                                        'DNA', 'DAN', 'Addition',
                                                        'WebQuestions', 'SimpleQA', 'TruthfulQA'], default='SST-2')
    parser.add_argument('--model_name', type=str,
                        choices=['Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat', 'Mistral-7B-Instruct-v0.2'],
                        default='Meta-Llama-3-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--defender', type=str, choices=['CleanGen', 'FineTune', 'Prune'], default='FineTune')
    parser.add_argument('--attacker_name', type=str, choices=['Clean', 'BadArch', 'BadNets', 'AddSent', 'BadChain', 'BadEdit'], default='BadEdit')
    args = parser.parse_args()

    args_dict = {
        "silence": args.silence,
        "task": args.task,
        "data_name": args.dataset,
        "model_name": args.model_name,
        "device": args.device,
        "Max_Epoch": args.epoch,
        "attacker_name": args.attacker_name
    }

    sum_path = 'result.csv'
    if args.silence == 0:
        print("######### Not Silence ########")
        with open(sum_path, 'a') as f:
            print(f'{datetime.datetime.now()}', file=f)
            print(f'task,dateset,victim,flag,CACC,ASR', file=f)
            f.close()

    if args.defender is not None:
        if args.defender == 'CleanGen':
            do_cleangen(**args_dict)
        elif args.defender == 'Prune':
            run_prune(**args_dict)
        elif args.defender == 'FineTune':
            do_finetune(**args_dict)
    else:
        if args.attacker_name == 'BadChain':
            run_BadChain(**args_dict)
        elif args.attacker_name == 'BadEdit':
            # Use the original badedit's code for attack, and save the attacked model for evaluation.
            # code path https://github.com/Lyz1213/BadEdit/tree/main
            eval_edit(**args_dict)
        elif args.attacker_name == 'BadNets' or args.attacker_name == 'AddSent':
            train(**args_dict)
            eval(**args_dict)
        elif args.attacker_name == 'Clean':
            clean_evaluation(**args_dict)
        elif args.attacker_name == 'BadArch':
            do_badarch(**args_dict)
