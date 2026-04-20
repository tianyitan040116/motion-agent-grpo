# motion-agent-grpo

An experimental fork of Motion-Agent focused on local training, GRPO-style reward design, evaluation utilities, and motion generation workflows.

<p align="center">
  <a href="https://arxiv.org/abs/2405.17013"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2405.17013-b31b1b.svg"></a>
  <a href="https://knoxzhao.github.io/Motion-Agent/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-orange"></a>
</p>

## What This Repo Contains

This repository builds on the original Motion-Agent codebase and adds local experimentation utilities around:

- GRPO training and reward shaping
- motion generation scripts for local checkpoints
- monitoring and analysis scripts for long-running training jobs
- evaluation helpers and debugging tools
- setup notes for local models and CUDA environments

This public repository intentionally excludes large local assets such as datasets, checkpoints, generated samples, logs, and virtual environments.

## Project Status

This is a research and experimentation repository rather than a polished package. Expect active iteration, local-tooling scripts, and training-oriented utilities.

## Repository Layout

```text
assets/               Example assets and previews
dataset/              Dataset loaders and evaluation code only
models/               Motion and language model components
options/              Training and inference option definitions
prepare/              Helper scripts for downloading dependencies
utils/                Shared evaluation and motion utilities

demo.py               Interactive demo entrypoint
train_mllm.py         Base MotionLLM training script
train_grpo.py         GRPO training entrypoint
eval_mllm.py          Evaluation entrypoint from upstream workflow
eval_grpo.py          GRPO-focused evaluation script
generate_motion_simple.py
visualize_motion.py
monitor_training.py
grpo_reward.py
spatiotemporal_reward.py
```

## Setup

### 1. Create an environment

```bash
conda create -n motionagent python=3.10
conda activate motionagent
pip install -r requirements.txt
```

### 2. Download required assets

The repo does not include pretrained weights or dataset payloads. Download them separately as needed.

For the original Motion-Agent resources:

```bash
bash prepare/download_ckpt.sh
bash prepare/download_glove.sh
bash prepare/download_extractor.sh
```

If you use a Hugging Face LLM backbone such as Gemma, make sure you have access to the model and have authenticated locally.

## Quick Start

### Run the demo

```bash
python demo.py
```

### Train MotionLLM

```bash
python train_mllm.py --training_task t2m
python train_mllm.py --training_task m2t
```

### Run GRPO training

```bash
python train_grpo.py
```

### Run evaluation

```bash
python eval_mllm.py
python eval_grpo.py
```

### Generate a motion sample

```bash
python generate_motion_simple.py --help
```

## Data Policy

This GitHub repository does not track:

- raw datasets or dataset archives
- generated motions and experiment outputs
- pretrained checkpoints and model weights
- local logs and scratch artifacts
- virtual environments and machine-specific files

If you want to reproduce experiments, place those assets locally in the expected folders after cloning.

## Notes For Public Use

- Some scripts assume local directory layouts developed during experimentation.
- Several utilities are intended for internal research workflows and may require path or checkpoint adjustments.
- If you plan to make this repo broadly reusable, a next good step would be centralizing configuration and documenting expected input formats per script.

## Upstream Reference

This work is based on the original Motion-Agent project:

- Paper: [Motion-Agent: A Conversational Framework for Human Motion Generation with LLMs](https://arxiv.org/abs/2405.17013)
- Project page: [Motion-Agent](https://knoxzhao.github.io/Motion-Agent/)

## Citation

If the upstream project is useful in your work, please cite:

```bibtex
@article{wu2024motion,
  title={Motion-Agent: A Conversational Framework for Human Motion Generation with LLMs},
  author={Wu, Qi and Zhao, Yubo and Wang, Yifan and Liu, Xinhang and Tai, Yu-Wing and Tang, Chi-Keung},
  journal={arXiv preprint arXiv:2405.17013},
  year={2024}
}

@inproceedings{wumotion,
  title={Motion-Agent: A Conversational Framework for Human Motion Generation with LLMs},
  author={Wu, Qi and Zhao, Yubo and Wang, Yifan and Liu, Xinhang and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

## Acknowledgements

Thanks to the original open-source projects that this codebase builds on:

- [T2M-GPT](https://github.com/Mael-zys/T2M-GPT)
- [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT)
- [text-to-motion](https://github.com/EricGuo5513/text-to-motion)
