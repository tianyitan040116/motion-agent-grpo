"""
从 ModelScope (魔搭社区) 下载 gemma-2-2b-it 模型
"""
import os
import sys

# 设置 modelscope 缓存目录映射到 huggingface 缓存
os.environ['CACHE_HOME'] = os.path.expanduser('~/.cache')

try:
    from modelscope import snapshot_download
    print("="*60)
    print("从 ModelScope (魔搭社区) 下载模型")
    print("="*60)

    print("\n正在下载 AI-ModelScope/gemma-2-2b-it...")
    print("(ModelScope 是 HuggingFace 在中国的镜像)")

    # ModelScope 上的对应模型
    # ignore_file_pattern 设置为 None 强制下载所有文件
    model_dir = snapshot_download(
        'AI-ModelScope/gemma-2-2b-it',
        cache_dir=os.path.expanduser('~/.cache/modelscope'),
        local_files_only=False,
        ignore_file_pattern=[]  # 下载所有文件，不忽略任何模式
    )

    print(f"\n[OK] Download completed: {model_dir}")

    # 列出文件
    import glob
    safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    print(f"\n找到 {len(safetensors_files)} 个 safetensors 文件:")
    for f in sorted(safetensors_files):
        size_mb = os.path.getsize(f) / (1024**2)
        print(f"  [OK] {os.path.basename(f):40s} {size_mb:>10.2f} MB")

    print(f"\n提示: 后续训练时请使用以下路径:")
    print(f"  --llm-backbone {model_dir}")

except ImportError:
    print("[ERROR] modelscope library not installed")
    print("\n请先安装: pip install modelscope")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Download failed: {e}")
    print("\n建议方案：")
    print("1. 检查网络连接")
    print("2. 或手动从以下地址下载:")
    print("   https://www.modelscope.cn/models/AI-ModelScope/gemma-2-2b-it")
    sys.exit(1)
