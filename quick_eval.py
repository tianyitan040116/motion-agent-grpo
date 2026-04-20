"""Quick A/B evaluation: SFT baseline vs GRPO checkpoint."""
import torch
import sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.mllm import MotionLLM
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from dataset import dataset_TM_eval
from utils.evaluation import evaluation_test
import argparse

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--llm-backbone', default='/root/autodl-tmp/gemma-2-2b-it')
    parser.add_argument('--lora-r-t2m', type=int, default=64)
    parser.add_argument('--lora-alpha-t2m', type=int, default=64)
    parser.add_argument('--lora-r-m2t', type=int, default=32)
    parser.add_argument('--lora-alpha-m2t', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--dataname', default='t2m')
    parser.add_argument('--code-dim', type=int, default=512)
    parser.add_argument('--nb-code', type=int, default=512)
    parser.add_argument('--mu', type=float, default=0.99)
    parser.add_argument('--down-t', type=int, default=2)
    parser.add_argument('--stride-t', type=int, default=2)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dilation-growth-rate', type=int, default=3)
    parser.add_argument('--output-emb-width', type=int, default=512)
    parser.add_argument('--vq-act', default='relu')
    parser.add_argument('--vq-norm', default=None)
    parser.add_argument('--quantizer', default='ema_reset')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--vq-path', default='ckpt/vqvae.pth')
    return parser.parse_args([])

def evaluate_checkpoint(model, eval_wrapper, val_loader, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    print(f"\n{'='*50}")
    print(f"Evaluating: {label}")
    print(f"{'='*50}")
    fid, div, top1, top2, top3, matching, multi = evaluation_test(
        out_dir, val_loader, model,
        eval_wrapper=eval_wrapper, draw=False, savenpy=True
    )
    print(f"\n  FID:      {fid:.4f}")
    print(f"  Diversity:{div:.4f}")
    print(f"  Top1:     {top1:.4f}")
    print(f"  Top2:     {top2:.4f}")
    print(f"  Top3:     {top3:.4f}")
    print(f"  Matching: {matching:.4f}")
    print(f"  MultiMod: {multi:.4f}")
    return dict(fid=fid, div=div, top1=top1, top2=top2, top3=top3, matching=matching, multi=multi)

def main():
    args = make_args()

    # Load eval components
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    wrapper_opt = get_opt('checkpoints/t2m/Comp_v6_KLD005/opt.txt', args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    val_loader = dataset_TM_eval.DATALoader('t2m', 'test', 32, w_vectorizer, unit_length=4)
    print(f"Test samples: {len(val_loader.dataset)}")

    # Build model
    model = MotionLLM(args)

    # --- Eval 1: SFT baseline ---
    model.load_model('ckpt/motionllm.pth')
    sft_results = evaluate_checkpoint(model, eval_wrapper, val_loader,
                                       'quick_eval_output/sft', 'SFT Baseline')

    # --- Eval 2: GRPO checkpoint ---
    grpo_ckpt = 'experiments_grpo/grpo_v4_run4/grpo_model.pth'
    if os.path.exists(grpo_ckpt):
        model.load_model(grpo_ckpt)
        grpo_results = evaluate_checkpoint(model, eval_wrapper, val_loader,
                                            'quick_eval_output/grpo', 'GRPO (batch ~150)')

        # --- Summary ---
        print(f"\n{'='*50}")
        print("COMPARISON: SFT vs GRPO")
        print(f"{'='*50}")
        print(f"{'Metric':<12} {'SFT':>10} {'GRPO':>10} {'Delta':>10}")
        print("-" * 44)
        for key in ['fid', 'top1', 'top2', 'top3', 'matching', 'div', 'multi']:
            s = sft_results[key]
            g = grpo_results[key]
            d = g - s
            better = '↓' if key in ['fid', 'matching'] else '↑'
            print(f"{key:<12} {s:>10.4f} {g:>10.4f} {d:>+10.4f} {better}")
    else:
        print(f"GRPO checkpoint not found: {grpo_ckpt}")

if __name__ == '__main__':
    main()
