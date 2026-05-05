#!/usr/bin/env bash
set -euo pipefail

cd ~/ecg_project/fairseq-signals
conda activate ecg

fairseq-hydra-train \
  task.data=/Users/youxu/ecg_project/data/ptbxl_manifest/identify \
  model.num_labels=85 \
  dataset.batch_size=2 \
  --config-dir examples/wav2vec2/config/finetuning \
  --config-name identification
