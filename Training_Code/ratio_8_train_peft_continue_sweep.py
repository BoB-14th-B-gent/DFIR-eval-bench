#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wandb sweep ratio_sweep_continue.yaml
nwandb agent ssh09015-student/ratio_B-gent_Sweep-5000+5000+100/ilje7mpq
tail -f sweep.log
"""

import os
import math
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
    TrainerCallback,
)
from peft import (
    LoraConfig, get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
import wandb


# ===============================
# 1. 명령줄 인자
# ===============================
def parse_args():
    p = argparse.ArgumentParser("QLoRA fine-tune + wandb sweep (with evaluation)")

    p.add_argument("--model_dir", required=True)
    p.add_argument("--train_data", required=True)
    p.add_argument("--val_data", required=True)
    p.add_argument("--out", required=True)

    # 하이퍼파라미터
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--max_len", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--grad_accum", type=int, required=True)

    p.add_argument("--logging_steps", type=int, default=10)

    # LoRA
    p.add_argument("--r", type=int, required=True)
    p.add_argument("--alpha", type=int, required=True)
    p.add_argument("--dropout", type=float, required=True)

    p.add_argument("--adapter", type=str, default=None)

    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    
    # wandb
    p.add_argument("--project", default="ratio_B-gent_Sweep-5000+5000+100")
    p.add_argument("--group", default="ratio_B-gent_Sweep-5000+5000+100")

    return p.parse_args()


args = parse_args()
print(f"[debug] resume arg = {args.resume_from_checkpoint}")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# wandb init
run = wandb.init(project=args.project, group=args.group, reinit=True)
args.out = os.path.join(args.out, f"sweep-{run.id}")
os.makedirs(args.out, exist_ok=True)
print(f"[info] Output directory: {args.out}")


# ===============================
# 2. 데이터 로딩 (split_v2 그대로 사용)
# ===============================
train_raw = load_dataset("json", data_files=args.train_data)["train"]
eval_raw  = load_dataset("json", data_files=args.val_data)["train"]

print(f"[info] Loaded train = {len(train_raw)}")
print(f"[info] Loaded val   = {len(eval_raw)}")

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(
    args.model_dir, use_fast=True, legacy=False, trust_remote_code=True, token=HF_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ===============================
# 3. Build text & char-level mask
# ===============================
def build_text_and_mask(conv):
    text = ""
    char_mask = []

    for turn in conv:
        role = turn["from"].lower()
        content = str(turn["value"])

        if role == "human":
            seg = f"<s>[INST] {content} [/INST]"
            text += seg
            char_mask += [0] * len(seg)  # Loss 제외

        elif role == "assistant":
            seg = content + "</s>"
            text += seg
            char_mask += [1] * len(seg)  # Loss 포함

    return text, char_mask


def _map_to_text(batch):
    texts, masks = [], []
    for conv in batch["conversations"]:
        t, m = build_text_and_mask(conv)
        texts.append(t)
        masks.append(m)
    return {"text": texts, "char_mask": masks}


train_text = train_raw.map(_map_to_text, batched=True, remove_columns=train_raw.column_names)
eval_text  = eval_raw.map(_map_to_text,  batched=True, remove_columns=eval_raw.column_names)



# ===============================
# 4. Tokenize
# ===============================
# ===============================
# 4. Tokenize with loss masking
# ===============================
def _tok_fn(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=args.max_len,
        return_offsets_mapping=True,
    )

    final_labels = []
    for offsets, cmask, ids in zip(enc["offset_mapping"], batch["char_mask"], enc["input_ids"]):
        labels = []
        for i, (s, _) in enumerate(offsets):
            # padding / human / overflow 영역 → -100
            if s is None or s >= len(cmask) or cmask[s] == 0:
                labels.append(-100)
            else:
                labels.append(ids[i])  # assistant 응답 토큰만 loss 계산
        final_labels.append(labels)

    enc["labels"] = final_labels
    enc.pop("offset_mapping", None)
    return enc


train_ds = train_text.map(_tok_fn, batched=True, remove_columns=train_text.column_names)
eval_ds  = eval_text.map(_tok_fn, batched=True, remove_columns=eval_text.column_names)


# ===============================
# 5. 모델 로드 (8bit + LoRA)
# ===============================
# bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

device_map = {"": 0}
max_memory = None
if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    def _cap(idx):
        total = torch.cuda.get_device_properties(idx).total_memory // (1024**3)
        reserve = 2 if total >= 16 else 1
        return f"{max(1, total - reserve)}GiB"
    max_memory = {i: _cap(i) for i in range(torch.cuda.device_count())}

base_model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    device_map=device_map,
    quantization_config=bnb_cfg,
    trust_remote_code=True,
    token=HF_TOKEN,
    max_memory=max_memory,
)

base_model = prepare_model_for_kbit_training(base_model)
base_model.config.use_cache = False

try:
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except:
    base_model.gradient_checkpointing_enable()

if args.adapter and Path(args.adapter).exists():
    model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=True)
else:
    lora_cfg = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(base_model, lora_cfg)


# ===============================
# 6. wandb callbacks
# ===============================
class LogPerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics and "eval_loss" in metrics:
            ppl = float(math.exp(min(metrics["eval_loss"], 20)))
            wandb.log({"perplexity": ppl, "global_step": state.global_step})


class ContinuousLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        data = {"global_step": state.global_step}
        if "loss" in logs:
            data["train/loss"] = logs["loss"]
        if "eval_loss" in logs:
            ppl = float(math.exp(min(logs["eval_loss"], 20)))
            data["eval/loss"] = logs["eval_loss"]
            data["eval/perplexity"] = ppl
        wandb.log(data)


# ===============================
# 7. 학습 + 저장
# ===============================
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    train_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_strategy="steps",
        eval_steps=10,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=True,
        report_to=["wandb"],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        eval_accumulation_steps=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=True,
        max_grad_norm=0.3,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[LogPerplexityCallback(), ContinuousLoggingCallback()],
    )

    print("[info] Training start")

    if args.resume_from_checkpoint:
        print(f"[info] Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("[info] Evaluating final model...")
    metrics = trainer.evaluate()
    print(f"[info] Final eval_loss = {metrics.get('eval_loss')}")

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[info] Training complete. Saved to: {args.out}")
