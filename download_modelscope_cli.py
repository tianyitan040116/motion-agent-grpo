"""
使用 modelscope download 命令行工具下载模型
"""
import subprocess
import os

print("="*60)
print("使用 ModelScope CLI 下载模型")
print("="*60)

# 使用 modelscope download 命令
cmd = [
    "python", "-m", "modelscope.cli.download",
    "--model", "AI-ModelScope/gemma-2-2b-it",
    "--local_dir", os.path.expanduser("~/.cache/modelscope/AI-ModelScope/gemma-2-2b-it")
]

print("\n执行命令:")
print(" ".join(cmd))
print()

try:
    result = subprocess.run(cmd, check=True, capture_output=False, text=True)
    print("\n[OK] Download completed!")
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] Download failed: {e}")
    print("\n请手动执行以下命令:")
    print("python -m modelscope.cli.download --model AI-ModelScope/gemma-2-2b-it")
except Exception as e:
    print(f"\n[ERROR] {e}")
