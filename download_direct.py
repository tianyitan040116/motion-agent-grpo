"""
直接从 hf-mirror.com 下载缺失的 model-00001 文件
"""
import requests
import os
from tqdm import tqdm

print("="*60)
print("从 hf-mirror.com 下载缺失的模型文件")
print("="*60)

# 镜像站直接下载URL
base_url = "https://hf-mirror.com/google/gemma-2-2b-it/resolve/main"
missing_file = "model-00001-of-00002.safetensors"
download_url = f"{base_url}/{missing_file}"

# 目标路径
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b-it")
snapshots_dir = os.path.join(cache_dir, "snapshots")

# 找到 snapshot 目录
snapshot_hash = None
for item in os.listdir(snapshots_dir):
    if os.path.isdir(os.path.join(snapshots_dir, item)):
        snapshot_hash = item
        break

if not snapshot_hash:
    print("✗ 找不到 snapshot 目录")
    exit(1)

snapshot_path = os.path.join(snapshots_dir, snapshot_hash)
output_file = os.path.join(snapshot_path, missing_file)

print(f"\n下载URL: {download_url}")
print(f"保存路径: {output_file}")

if os.path.exists(output_file):
    print(f"\n✓ 文件已存在，跳过下载")
else:
    print(f"\n开始下载 {missing_file}...")

    try:
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_file, 'wb') as f, tqdm(
            desc=missing_file,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"\n✓ 下载完成！")

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        exit(1)

# 验证文件
print("\n验证模型文件完整性...")
safetensors_files = [f for f in os.listdir(snapshot_path) if f.endswith('.safetensors')]
safetensors_files.sort()

print(f"\n找到 {len(safetensors_files)} 个 safetensors 文件:")
for f in safetensors_files:
    fpath = os.path.join(snapshot_path, f)
    size_mb = os.path.getsize(fpath) / (1024**2)
    print(f"  ✓ {f:40s} {size_mb:>10.2f} MB")

if len(safetensors_files) >= 2:
    print("\n✓ 模型文件完整！可以开始训练了。")
else:
    print(f"\n✗ 警告：只找到 {len(safetensors_files)} 个文件，预期至少2个")
