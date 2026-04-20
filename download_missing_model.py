"""
下载 gemma-2-2b-it 模型缺失的文件
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download
import sys

print("="*60)
print("下载 google/gemma-2-2b-it 缺失的模型文件")
print("="*60)

try:
    print("\n开始下载（使用镜像站 hf-mirror.com）...")
    print("这会自动检测并只下载缺失的文件\n")

    local_dir = snapshot_download(
        repo_id="google/gemma-2-2b-it",
        resume_download=True,
        local_files_only=False,
    )

    print(f"\n✓ 下载完成！模型保存在: {local_dir}")
    print("\n检查文件完整性...")

    # 列出关键文件
    import glob
    safetensors_files = glob.glob(os.path.join(local_dir, "*.safetensors"))
    print(f"\n找到 {len(safetensors_files)} 个 safetensors 文件:")
    for f in sorted(safetensors_files):
        size_mb = os.path.getsize(f) / (1024**2)
        print(f"  - {os.path.basename(f):40s} {size_mb:>10.2f} MB")

    print("\n✓ 模型下载验证完成！")

except Exception as e:
    print(f"\n✗ 下载失败: {e}")
    sys.exit(1)
