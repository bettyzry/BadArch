import argparse, datetime
import logging

from competitors.BadEdit import eval_badedit
from competitors.SFT import train_sft, eval_sft
from competitors.BadChain import eval_badchain
from competitors.Clean import eval_clean
from competitors.BadArch import eval_badarch

from defender.defender_cleangen import do_cleangen
from defender.defender_prune import do_prune
from defender.defender_finetune import do_finetune
from defender.defender_crow import do_crow_finetuning

logging.getLogger("pydevd").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--silence', type=int, default=1)
    parser.add_argument('--task', type=str,
                        choices=['classification', 'jailbreak', 'steal'], default='classification')
    parser.add_argument('--dataset', type=str, choices=['SST-2', 'AGNews', 'HateSpeech',
                                                        'DNA', 'DAN', 'Addition',
                                                        'WebQuestions', 'SimpleQA', 'TruthfulQA'], default='AGNews')
    parser.add_argument('--model_name', type=str,
                        choices=['Meta-Llama-3-8B-Instruct', 'deepseek-llm-7b-chat', 'Mistral-7B-Instruct-v0.2'],
                        default='Meta-Llama-3-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--defender', type=str, choices=['None', 'CleanGen', 'FineTune', 'Prune', 'Crow'], default='None')
    parser.add_argument('--prune_method', type=str, choices=['wanda', 'magnitude', 'sparsegpt', 'ablate'], default='magnitude')
    parser.add_argument('--attacker_name', type=str, choices=['Clean', 'BadArch', 'BadNets', 'AddSent', 'BadChain', 'BadEdit'], default='BadChain')
    args = parser.parse_args()

    args_dict = {
        "silence": args.silence,
        "task": args.task,
        "data_name": args.dataset,
        "model_name": args.model_name,
        "device": args.device,
        "Max_Epoch": args.epoch,
        "attacker_name": args.attacker_name,
        "prune_method": args.prune_method,
    }

    sum_path = 'result.csv'
    if args.silence == 0:
        print("######### Not Silence ########")
        with open(sum_path, 'a') as f:
            print(f'{datetime.datetime.now()}', file=f)
            print(f'task,dateset,victim,flag,CACC,ASR', file=f)
            f.close()

    # do defense
    if args.defender != 'None':
        if args.defender == 'CleanGen':
            do_cleangen(**args_dict)
        elif args.defender == 'Prune':
            do_prune(**args_dict)
        elif args.defender == 'FineTune':
            do_finetune(**args_dict)
        elif args.defender == 'Crow':
            do_crow_finetuning(**args_dict)
    # do attack
    else:
        if args.attacker_name == 'Clean':
            eval_clean(**args_dict)
        elif args.attacker_name == 'BadArch':
            eval_badarch(**args_dict)
        elif args.attacker_name == 'BadNets' or args.attacker_name == 'AddSent':
            train_sft(**args_dict)
            eval_sft(**args_dict)
        elif args.attacker_name == 'BadChain':
            eval_badchain(**args_dict)
        elif args.attacker_name == 'BadEdit':
            # Use the original badedit's code for attack, and save the attacked model for evaluation.
            # code path https://github.com/Lyz1213/BadEdit/tree/main
            eval_badedit(**args_dict)
