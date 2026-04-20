#!/bin/bash
# Sequential evaluation: epoch0 GRPO vs SFT baseline
set -e
cd /root/autodl-tmp/motion-agent

LOG_EPOCH0="experiments_grpo/grpo_from_sft/eval_epoch0.log"
LOG_BASELINE="experiments_grpo/grpo_from_sft/eval_sft_baseline.log"
RESULT_FILE="experiments_grpo/grpo_from_sft/comparison_results.txt"

echo "============================================================"
echo " Evaluation started at $(date)"
echo "============================================================"

echo ""
echo ">>> [1/2] Evaluating epoch0 GRPO checkpoint..."
echo "    Checkpoint: experiments_grpo/grpo_from_sft/motionllm_grpo_latest.pth"
python eval_epoch0.py \
    --checkpoint experiments_grpo/grpo_from_sft/motionllm_grpo_latest.pth \
    --split val \
    --repeat 1 \
    2>&1 | tee "$LOG_EPOCH0"
echo ">>> [1/2] epoch0 GRPO eval done at $(date)"

echo ""
echo ">>> [2/2] Evaluating SFT baseline checkpoint..."
echo "    Checkpoint: ckpt/motionllm.pth"
python eval_epoch0.py \
    --checkpoint ckpt/motionllm.pth \
    --split val \
    --repeat 1 \
    2>&1 | tee "$LOG_BASELINE"
echo ">>> [2/2] SFT baseline eval done at $(date)"

echo ""
echo "============================================================"
echo " Both evaluations complete at $(date)"
echo " Results file: $RESULT_FILE"
echo "============================================================"

# Print comparison summary
echo ""
echo "======= COMPARISON SUMMARY ======="
echo ""
echo "--- Epoch0 GRPO ---"
grep -E "FID\.|Diversity|TOP[123]|Matching|Multi|compact" "$LOG_EPOCH0" | tail -5 || echo "(no results found)"

echo ""
echo "--- SFT Baseline ---"
grep -E "FID\.|Diversity|TOP[123]|Matching|Multi|compact" "$LOG_BASELINE" | tail -5 || echo "(no results found)"
