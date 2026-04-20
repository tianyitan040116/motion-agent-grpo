"""Test VQ-VAE loading"""
import torch
import models.vqvae as vqvae
from options.option_train import get_args_parser

print("Testing VQ-VAE loading...")
args = get_args_parser()
args.device = 'cuda:0'
args.vq_path = "ckpt/vqvae.pth"

print(f"Creating VQ-VAE model...")
net = vqvae.HumanVQVAE(
    args,
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate,
    args.vq_act,
    args.vq_norm
)
print("[OK] VQ-VAE model created")

print(f"Loading checkpoint from {args.vq_path}...")
ckpt = torch.load(args.vq_path, map_location='cpu')
print(f"[OK] Checkpoint loaded, keys: {list(ckpt.keys())}")

print("Loading state dict...")
net.load_state_dict(ckpt['net'], strict=True)
print("[OK] State dict loaded")

print("Moving to device...")
net = net.to(args.device)
print("[OK] Moved to device")

net.eval()
print("[SUCCESS] VQ-VAE ready!")
