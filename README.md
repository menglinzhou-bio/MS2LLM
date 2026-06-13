# **MS2LLM**
## Overview

MS2LLM is a large language model framework for interpreting tandem mass spectrometry (MS/MS) data. The model represents MS/MS spectra and molecular structures as natural language and learns their correspondence through instruction tuning.

## Repository Structure
```text
MS2LLM
├── configs
│   ├── train_lora.yaml
│   └── inference_lora.yaml
├── examples
│   ├── example_questions.json
│   └── example_predictions.jsonl
├── ms2llm
│   ├── __init__.py
│   ├── dataset.py
│   ├── lightning_module.py
│   └── peft_model.py
├── scripts
│   ├── train_lora.py
│   └── inference_lora.py
├── requirements.txt
└── README.md
```
## Installation
Install dependencies:
```bash
git clone https://github.com/menglinzhou-bio/MS2LLM.git
cd MS2LLM
pip install -r requirements.txt
```
Main dependencies:
- torch
- transformers
- peft
- pytorch-lightning
- safetensors
- tqdm
- PyYAML

## Training
Training configurations are stored in:

`configs/train_lora.yaml`
### Example: Multi-GPU Training with Slurm
The experiments in this work were performed on a Slurm-managed cluster using 8 NVIDIA H100 GPUs.

Example command:

```bash
srun \
  --gres=gpu:8 \
  --ntasks=8 \
  --ntasks-per-node=8 \
  --cpus-per-task=16 \
  --mem=480G \
  python scripts/train_lora.py
```
The Slurm configuration (partition, node allocation, GPU count, CPU count, and memory allocation) should be adjusted according to the target computing environment.
## Inference
Inference configurations are stored in:

`configs/inference_lora.yaml`

### Example: Multi-GPU Inference with Slurm


```bash
srun \
  --gres=gpu:8 \
  --ntasks=8 \
  --ntasks-per-node=8 \
  --cpus-per-task=16 \
  --mem=480G \
  python scripts/inference_lora.py
```
The Slurm configuration should be adjusted according to the target computing environment.
## Availability
This repository contains the training and inference scripts used in the manuscript.

Due to storage limitations, model checkpoints, training datasets, and benchmark datasets are not included.

Example input and output files are provided in the `examples/` directory.

