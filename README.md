# DFIR-eval-bench

## 1. Forensic Scenario Data
The `Scenario/` directory contains **Six DFIR Analysis Scenarios**, each representing a distinct incident or investigation context.

Each scenario includes:
- `Image File (E01)`
- `winlog.csv`
- `fw.csv`

These logs are used as primary analysis inputs for DFIR agents and sLLM-based reasoning workflows.

```
Scenario/
├─ 01_Scenario_Ransack/
│  ├─ Caldera_01_diskImage.E01
│  ├─ winlog.csv
│  └─ fw.csv
├─ 02_Scenario_Alice/
│  ├─ Caldera_02_diskImage.E01
│  ├─ winlog.csv
│  └─ fw.csv
├─ 03_Scenario_Discovery/
│  ├─ Caldera_03_diskImage.E01
│  ├─ winlog.csv
│  └─ fw.csv
├─ 04_Scenario_pip/
│  ├─ Scenario_pip_diskImage.E01
│  ├─ winlog.csv
│  └─ fw.csv
├─ 05_Scenario_blackmoon/
│  ├─ Scenario_blackmoon_DiskImage.E01
│  ├─ winlog.csv
│  └─ fw.csv
└─ 06_Scenario_discord/
   ├─ Scenario_discord_diskImage.E01
   ├─ winlog.csv
   └─ fw.csv
```

## 2. Agent & sLLM Training Code

The `Training_Code/` directory contains **training and fine-tuning scripts for DFIR-specialized small Language Models (sLLMs)** based on  
**`meta-llama/Llama-3.1-8B-Instruct`**.

These scripts adapt the base LLaMA 3.1 8B Instruct model for **agent-based digital forensics and incident response (DFIR) tasks**, including structured reasoning, tool selection, and multi-step action planning.

### Model Configuration

- **Base Model**
  - `meta-llama/Llama-3.1-8B-Instruct`
  - Decoder-only causal language model

- **Quantization & Precision**
  - Loaded using **4-bit NF4 quantization** (bitsandbytes)
  - Computation performed in **bfloat16**

- **Fine-Tuning Method**
  - **QLoRA (4-bit quantization + LoRA adapters)**
  - LoRA adapters applied to attention projection layers:
    - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - Supports both initial adapter training and continued training from existing adapters or checkpoints

### Training Objective

The training objective is aligned with **DFIR agent behavior**, rather than general chat generation.

The model is trained to generate:
- Structured **PURPOSE** statements
- Correct **TOOL** selection
- Well-formed **REQUEST** outputs for forensic analysis steps

Loss is selectively applied only to agent-relevant output tokens, while prompt and context tokens are excluded using custom token-level masking.

### Dataset Format

- JSON-based multi-turn agent datasets
- Each sample represents a DFIR analysis task with:
  - An instruction
  - Available forensic tools
  - Sequential agent reasoning steps
- Supports both MCP-style agent traces and conversation-style assistant supervision

### Experiment Management

- Integrated with **Weights & Biases (wandb)** for:
  - Hyperparameter sweeps
  - Training and evaluation tracking
  - Loss and perplexity monitoring
- Supports long-running experiments with checkpoint resume
- Compatible with single-GPU and multi-GPU environments

### Training Code Structure

```text
Training_Code/
├─ agent_ratio_8_train_peft_continue_sweep.py
├─ ratio_8_train_peft_continue_sweep.py
├─ ratio_sweep_continue.yaml
└─ ratio_sweep_init.yaml
```

These scripts and configuration files define the full training workflow, including:
Initial and continued QLoRA fine-tuning
LoRA rank- and ratio-based hyperparameter sweeps
Experiment configuration and tracking via Weights & Biases
The resulting fine-tuned models are used as DFIR-aware sLLMs for downstream agent evaluation using the forensic scenarios provided in this benchmark.
