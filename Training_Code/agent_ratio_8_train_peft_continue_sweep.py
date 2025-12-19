#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wandb sweep ratio_sweep_init.yaml
wandb agent ssh09015-student/hspace_agent_ratio_B-gent_Init_365/gkzgjm8q
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
    p.add_argument("--project", default="hspace_agent_ratio_B-gent_Init_365")
    p.add_argument("--group", default="hspace_agent_ratio_B-gent_Init_365")

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
# 2. 데이터 로딩
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
# 3. Build text from MCP dataset
#   - Input: instruction + available_tools
#   - Output (loss): PURPOSE / TOOL / REQUEST lines
# ===============================
def build_text_from_item(item):
    instruction = (item.get("instruction") or "").strip()
    ctx = item.get("context") or {}
    turns = item.get("turns") or []

    available_tools = ctx.get("available_tools") or []

    lines = []
    # 질문(인풋)
    lines.append(f"<s>[INST] {instruction} [/INST]")

    # available_tools 노출 (인풋)
    if available_tools:
        tools_str = ", ".join(available_tools)
        lines.append(f"Available tools: {tools_str}")

    # 멀티턴: 각 턴의 PURPOSE / TOOL / REQUEST / RESPONSE / ANALYSIS
    for idx, t in enumerate(turns, start=1):
        purpose = (t.get("purpose") or "").strip()
        tool = (t.get("tool") or "").strip()
        request = (t.get("request") or "").strip()
        response = (t.get("response") or "").strip()
        analysis = (t.get("analysis") or "").strip()

        # 아래 세 줄이 "모델 아웃풋" (loss 계산 대상)
        if purpose:
            lines.append(f"[STEP {idx} PURPOSE] {purpose}")

        if tool:
            lines.append(f"[STEP {idx} TOOL] {tool}")

        if request:
            request = request.strip()
            lines.append(f"[STEP {idx} REQUEST] {request}")

        # 너가 response 자체적으로 줄이면 아래로 바꾸삼
        # if response:
        #     response = response.strip()
        #     if len(response) > 512:
        #         response = response[:512] + "...(truncated)"
        #     lines.append(f"[STEP {idx} RESPONSE] {response}")

        if response:
            response = response.strip()
            lines.append(f"[STEP {idx} RESPONSE] {response}")

        if analysis:
            lines.append(f"[STEP {idx} ANALYSIS] {analysis}")

    lines.append("</s>")
    return "\n".join(l for l in lines if l.strip()) or "[EMPTY]"


def _map_to_text(batch):
    texts = []
    instr_list = batch.get("instruction", [])
    turns_list = batch.get("turns", [])
    ctx_list = batch.get("context", [None] * len(instr_list))

    for instr, ctx, turns in zip(instr_list, ctx_list, turns_list):
        item = {
            "instruction": instr,
            "context": ctx,
            "turns": turns,
        }
        texts.append(build_text_from_item(item))

    return {"text": texts}


train_text = train_raw.map(_map_to_text, batched=True, remove_columns=train_raw.column_names)
eval_text  = eval_raw.map(_map_to_text,  batched=True, remove_columns=eval_raw.column_names)


# ===============================
# 4. Tokenize + Loss mask
#   - loss 대상: [STEP i PURPOSE], [STEP i TOOL], [STEP i REQUEST]
# ===============================
def _tok_fn(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=args.max_len,
        return_offsets_mapping=True,
    )

    all_label_masks = []

    for text, offsets, ids in zip(
        batch["text"],
        enc["offset_mapping"],
        enc["input_ids"],
    ):
        loss_ranges = []
        cur_pos = 0

        # 라인 단위로 스캔하면서 PURPOSE / TOOL / REQUEST 라인만 loss 대상 구간으로 설정
        for line in text.splitlines(keepends=True):
            line_len = len(line)

            stripped = line.lstrip()
            is_step_line = stripped.startswith("[STEP ")
            target = (
                " PURPOSE]" in stripped or
                " TOOL]" in stripped or
                " REQUEST]" in stripped
            )

            if is_step_line and target:
                start = cur_pos
                end = cur_pos + line_len
                loss_ranges.append((start, end))

            cur_pos += line_len

        mask = [0] * args.max_len

        if loss_ranges:
            # PURPOSE / TOOL / REQUEST 토큰만 1로 마스킹
            for i, off in enumerate(offsets):
                if i >= args.max_len:
                    break
                if off is None:
                    continue
                s, e = off
                # 일부 토크나이저는 특수 토큰에 (0,0) 줌 → 그대로 두면 됨
                if ids[i] == tokenizer.pad_token_id:
                    continue

                for rs, re in loss_ranges:
                    # 토큰 span이 대상 라인 구간과 겹치면 loss에 포함
                    if e > rs and s < re:
                        mask[i] = 1
                        break
        else:
            # fallback: 목적 라인이 전혀 없으면, 첫 non-pad 토큰 하나만 학습시켜서 NaN 방지
            for i, tid in enumerate(ids):
                if tid != tokenizer.pad_token_id:
                    mask[i] = 1
                    break

        all_label_masks.append(mask)

    pad_id = tokenizer.pad_token_id

    enc["labels"] = [
        [
            (tid if (m == 1 and tid != pad_id) else -100)
            for tid, m in zip(ids, mask)
        ]
        for ids, mask in zip(enc["input_ids"], all_label_masks)
    ]

    enc.pop("offset_mapping", None)
    return enc


train_ds = train_text.map(_tok_fn, batched=True, remove_columns=train_text.column_names)
eval_ds  = eval_text.map(_tok_fn,  batched=True, remove_columns=eval_text.column_names)


# ===============================
# 5. 모델 로드 (4bit + LoRA)
# ===============================
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

    # 디버그용: 첫 샘플 토큰/라벨 확인
    if len(train_text) > 0:
        sample = train_text[0]["text"]
        dbg_enc = _tok_fn({"text": [sample]})

        print("\n===== DEBUG SAMPLE TEXT =====")
        print(sample)

        print("\n===== TOKEN / LABEL (앞 200 토큰) =====")
        ids = dbg_enc["input_ids"][0][:200]
        labs = dbg_enc["labels"][0][:200]

        for tid, lab in zip(ids, labs):
            t = tokenizer.decode([tid])
            print(repr(t), " | label=", lab)

    train_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_strategy="steps",
        eval_steps=20,
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
