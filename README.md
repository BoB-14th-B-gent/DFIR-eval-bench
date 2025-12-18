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
- **Local small Language Models (sLLMs)** optimized for forensic reasoning
