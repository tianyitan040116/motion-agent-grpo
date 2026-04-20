"""
Standalone validation script for epoch0 GRPO checkpoint.
Loads motionllm_grpo_latest.pth and runs evaluation on val set.
"""
import torch
import numpy as np
import os
import sys
import argparse
import types

# Force unbuffered stdout so progress bars reach the log file immediately
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Change to script directory for relative paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models.mllm import MotionLLM
from utils.evaluation import evaluation_test
from dataset import dataset_TM_eval
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt

DEFAULT_CHECKPOINT = "experiments_grpo/grpo_from_sft/motionllm_grpo_latest.pth"
OUT_DIR = "experiments_grpo/grpo_from_sft"
REPEAT_TIME = 1


def build_args():
    args = types.SimpleNamespace(
        device=torch.device("cuda:0"),
        llm_backbone="/root/autodl-tmp/gemma-2-2b-it",
        lora_r_t2m=64,
        lora_alpha_t2m=64,
        lora_r_m2t=32,
        lora_alpha_m2t=32,
        lora_dropout=0.1,
        # vqvae
        code_dim=512,
        nb_code=512,
        mu=0.99,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        output_emb_width=512,
        vq_act="relu",
        vq_norm=None,
        quantizer="ema_reset",
        beta=1.0,
        vq_path="ckpt/vqvae.pth",
        # misc
        dataname="t2m",
        out_dir=OUT_DIR,
    )
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=REPEAT_TIME,
                        help="Number of repeat evaluations (default 1)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on (default: val)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to checkpoint (default: grpo epoch0 latest)")
    parser.add_argument("--savenpy", action="store_true",
                        help="Save prediction numpy arrays")
    cli = parser.parse_args()

    args = build_args()

    print(f"Loading MotionLLM from {cli.checkpoint} ...", flush=True)
    model = MotionLLM(args)
    model.load_model(cli.checkpoint)
    model.eval()
    print(f"[OK] Model loaded", flush=True)

    # Evaluator
    w_vectorizer = WordVectorizer("glove", "our_vab")
    dataset_opt_path = "checkpoints/t2m/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # Data loader
    loader = dataset_TM_eval.DATALoader(
        args.dataname, cli.split, 32, w_vectorizer,
        unit_length=2 ** args.down_t
    )
    print(f"[OK] {cli.split} samples: {len(loader.dataset)}", flush=True)

    fid_list, div_list, top1_list, top2_list, top3_list, match_list, multi_list = [], [], [], [], [], [], []

    for i in range(cli.repeat):
        print(f"\n--- Repeat {i+1}/{cli.repeat} ---", flush=True)
        ckpt_name = os.path.basename(cli.checkpoint).replace(".pth", "")
        tag = f"{ckpt_name}_{cli.split}_r{i}"
        fid, div, top1, top2, top3, matching, multi = evaluation_test(
            OUT_DIR, loader, model, eval_wrapper=eval_wrapper,
            draw=False, savenpy=(cli.savenpy and i == 0),
            ckpt_tag=tag,
        )
        fid_list.append(fid)
        div_list.append(div)
        top1_list.append(top1)
        top2_list.append(top2)
        top3_list.append(top3)
        match_list.append(matching)
        multi_list.append(multi)
        print(f"  FID:{fid:.4f} Div:{div:.4f} Top1:{top1:.4f} Top2:{top2:.4f} Top3:{top3:.4f} Match:{matching:.4f} Multi:{multi:.4f}", flush=True)

    n = cli.repeat
    fid_arr    = np.array(fid_list)
    div_arr    = np.array(div_list)
    top1_arr   = np.array(top1_list)
    top2_arr   = np.array(top2_list)
    top3_arr   = np.array(top3_list)
    match_arr  = np.array(match_list)
    multi_arr  = np.array(multi_list)

    ci = lambda x: np.std(x) * 1.96 / np.sqrt(n)

    print("\n" + "="*70)
    print(f"Epoch0 GRPO Validation Results  ({cli.split} split, {n} repeats)")
    print("="*70)
    msg = (
        f"FID.      {np.mean(fid_arr):.3f}  ± {ci(fid_arr):.3f}\n"
        f"Diversity {np.mean(div_arr):.3f}  ± {ci(div_arr):.3f}\n"
        f"Top1      {np.mean(top1_arr):.3f}  ± {ci(top1_arr):.3f}\n"
        f"Top2      {np.mean(top2_arr):.3f}  ± {ci(top2_arr):.3f}\n"
        f"Top3      {np.mean(top3_arr):.3f}  ± {ci(top3_arr):.3f}\n"
        f"Matching  {np.mean(match_arr):.3f}  ± {ci(match_arr):.3f}\n"
        f"Multi     {np.mean(multi_arr):.3f}  ± {ci(multi_arr):.3f}"
    )
    print(msg)

    compact = (
        f"FID. {np.mean(fid_arr):.3f}, conf. {ci(fid_arr):.3f}, "
        f"Diversity. {np.mean(div_arr):.3f}, conf. {ci(div_arr):.3f}, "
        f"TOP1. {np.mean(top1_arr):.3f}, conf. {ci(top1_arr):.3f}, "
        f"TOP2. {np.mean(top2_arr):.3f}, conf. {ci(top2_arr):.3f}, "
        f"TOP3. {np.mean(top3_arr):.3f}, conf. {ci(top3_arr):.3f}, "
        f"Matching. {np.mean(match_arr):.3f}, conf. {ci(match_arr):.3f}, "
        f"Multi. {np.mean(multi_arr):.3f}, conf. {ci(multi_arr):.3f}"
    )
    print("\n[compact]", compact)

    # Save results
    result_path = os.path.join(OUT_DIR, "epoch0_val_results.txt")
    with open(result_path, "a") as f:
        f.write(f"\nCheckpoint: {cli.checkpoint}\n")
        f.write(f"Split: {cli.split}, Repeats: {n}\n\n")
        f.write(msg + "\n\n")
        f.write(compact + "\n")
    print(f"\n[OK] Results saved to {result_path}")


if __name__ == "__main__":
    main()
