import json, torch, os
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from bitsandbytes.optim import AdamW8bit
from peft import PeftModel
from defender.train_model import finetune
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)

def defense_finetune(data_name, model_name, task, in_path, out_path, attacker_name, device=DEVICE, Max_Epoch=20):
    MODEL_ID = f"/home/server/SSD/{in_path}/{attacker_name}/{model_name}-{task}"
    if data_name == 'AGNews':
        MODEL_ID  = MODEL_ID + "-AGNews"
    DATA_PATH = f'./mydataset/{data_name}/train.json'

    # # ---------- 1. 量化 ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    if attacker_name == 'BadArch':
        model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,  # 或 Hub repo id
        quantization_config=bnb_config,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    else:
        base_model = f'/home/server/SSD/llms/{model_name}'
        model = AutoModelForCausalLM.from_pretrained(
            base_model,  # 或 Hub repo id
            quantization_config=bnb_config,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)
        model = PeftModel.from_pretrained(model, model_id=MODEL_ID, config=config)
        model.eval()
        MODEL_ID = base_model
    print(model.__class__)

    # ---------- 2. 加载模型 & tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    finetune(model, tokenizer, data, attacker_name, model_name, data_name, out_path, task=task, Max_Epoch=Max_Epoch)


def defense_finetune_full(data_name, model_name, task, attacker_name='BadArch', device=DEVICE, Max_Epoch=20):
    MODEL_ID = f"/home/server/SSD/attacked/{attacker_name}/{model_name}-{task}"  # 任意 causalLM
    DATA_PATH = f'./mydataset/{data_name}/train.json'

    MAX_LEN       = 512
    BATCH_SIZE    = 4
    GRAD_ACCUM    = 32
    LR            = 1e-5
    WARMUP_STEPS  = 100
    SAVE_STEPS    = 500

    # # ---------- 1. 量化 ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # ---------- 2. 加载模型 & tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,  # 或 Hub repo id
        quantization_config=bnb_config,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    print(model.__class__)

    # ---------- 3. LoRA ----------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------- 4. 数据集 ----------
    class InstructDataset(Dataset):
        def __init__(self, path, tokenizer, max_len=256):
            self.tokenizer = tokenizer
            self.max_len   = max_len
            with open(path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            prompt = f"System:\n{sample['instruction']}\n\nUser:\n{sample['input']}\n\nAssistent:\n"
            answer = sample["output"] + self.tokenizer.eos_token
            full  = prompt + answer

            # 只给 answer 部分计算 loss
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            full_ids   = self.tokenizer.encode(full,  add_special_tokens=False)
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

            # 截断
            if len(full_ids) > self.max_len:
                full_ids = full_ids[:self.max_len]
                labels   = labels[:self.max_len]

            return {
                "input_ids": full_ids,
                "labels":    labels,
                "attention_mask": [1]*len(full_ids)
            }

    def collate(batch):
        # 动态 padding
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        labels    = [torch.tensor(b["labels"]) for b in batch]
        attn      = [torch.tensor(b["attention_mask"]) for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels    = torch.nn.utils.rnn.pad_sequence(labels,    batch_first=True, padding_value=-100)
        attn      = torch.nn.utils.rnn.pad_sequence(attn,      batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    train_ds = InstructDataset(DATA_PATH, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True)

    # ---------- 5. 优化器 & 调度器 ----------
    total_steps = len(train_loader) * Max_Epoch // GRAD_ACCUM
    optimizer = AdamW8bit(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    # ---------- 6. 训练循环 ----------
    model.train()
    step = 0
    for epoch in range(Max_Epoch):
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item()

            if (i + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 10 == 0:
                    print(f"epoch {epoch} | step {step} | loss {epoch_loss/(i+1):.4f}")

        # ---------- 7. 最终保存 ----------
        # 每个epoch 保存一个结果
        OUT_DIR = f"/home/server/SSD/finetuned/{model_name}-{data_name}-{epoch}"
        # model = model.merge_and_unload()  # 保存整个模型
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        print("Done! Full model saved ->", OUT_DIR)



if __name__ == '__main__':
    model_names = ['Meta-Llama-3-8B-Instruct']
    data_name = 'SST-2'
    for model_name in model_names:
        defense_finetune(data_name, model_name, 'Classification')