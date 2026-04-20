"""
Quick test for GRPO training step
"""
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.mllm import MotionLLM
from grpo_reward import GRPORewardModel
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from options.get_eval_option import get_opt
from options.option_train import get_args_parser

print("="*60)
print("Testing GRPO Training Step")
print("="*60)

# Load models
args = get_args_parser()
args.device = 'cuda:0'

print("\n[1/3] Loading MotionLLM...")
model = MotionLLM(args)
model.eval()
print("[OK] MotionLLM loaded")

print("\n[2/3] Loading reward model...")
w_vectorizer = WordVectorizer('./glove', 'our_vab')
dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, args.device)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
reward_model = GRPORewardModel(
    eval_wrapper=eval_wrapper,
    vqvae_model=model.net,
    word_vectorizer=w_vectorizer,
    device=args.device
)
print("[OK] Reward model loaded")

print("\n[3/3] Testing motion generation...")
test_caption = "a person walks forward"
print(f"Caption: '{test_caption}'")

# Test generation (this is what might be hanging)
with torch.no_grad():
    try:
        print("Generating motion...")
        motion_tokens = model.generate_one_motion(test_caption)
        print(f"[OK] Generated {len(motion_tokens)} tokens")

        print("Computing reward...")
        reward = reward_model.compute_reward([test_caption], [motion_tokens])
        print(f"[OK] Reward: {reward.item():.4f}")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("[SUCCESS] Test completed!")
print("="*60)
