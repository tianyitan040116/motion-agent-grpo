"""
生成多个motion样本用于对比
用法: python generate_multiple.py experiments_grpo/grpo_kinematic/grpo_model.pth "a person walks" 5
"""
import torch
import numpy as np
import sys
from models.mllm import MotionLLM
from options.option_llm import get_args_parser

def main():
    if len(sys.argv) < 4:
        print("Usage: python generate_multiple.py <checkpoint> <text> <num_samples> [method]")
        print("  method: beam (default) or sample")
        sys.exit(1)

    checkpoint = sys.argv[1]
    text = sys.argv[2]
    num_samples = int(sys.argv[3])
    method = sys.argv[4] if len(sys.argv) > 4 else 'beam'

    args = get_args_parser()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model from: {checkpoint}")
    model = MotionLLM(args)
    model.load_model(checkpoint)
    model.eval()

    print(f"\nGenerating {num_samples} samples using {method} search...")
    print(f"Text: {text}\n")

    for i in range(num_samples):
        print(f"Sample {i+1}/{num_samples}...")

        with torch.no_grad():
            if method == 'beam':
                motion_tokens = model.generate_one_motion(text)
            else:
                motion_tokens = model.generate_one_motion_sampling(
                    text, temperature=1.0, top_p=0.9, max_length=200
                )

            motion = model.net.forward_decoder(motion_tokens.unsqueeze(0))
            motion_np = model.denormalize(motion.cpu().numpy())[0]

            output_path = f"generated_motions/multi_{method}_{i+1}.npy"
            np.save(output_path, motion_np)

            print(f"  Tokens: {len(motion_tokens)}, Frames: {motion_np.shape[0]}, "
                  f"Duration: {motion_np.shape[0]/20:.1f}s")
            print(f"  Saved to: {output_path}")

    print(f"\nAll {num_samples} samples generated!")

if __name__ == '__main__':
    main()
