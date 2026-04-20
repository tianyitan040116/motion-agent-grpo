# -*- coding: utf-8 -*-
"""
自动定位 Windows Downloads 文件夹中的 gemma-2-2b-it 模型
"""
import os
import sys
from pathlib import Path

# 设置输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def find_model_path():
    # 获取 Windows Downloads 文件夹路径
    downloads_folder = Path.home() / "Downloads"
    print(f"检查 Downloads 文件夹: {downloads_folder}")

    # 可能的模型路径
    possible_paths = [
        downloads_folder / "gemma-2-2b-it",
        downloads_folder / "gemma-2-2b-it" / "gemma-2-2b-it",  # 可能嵌套了一层
    ]

    for path in possible_paths:
        print(f"\n检查路径: {path}")
        if path.exists():
            print(f"[OK] 路径存在")

            # 检查关键文件
            config_json = path / "config.json"
            has_config = config_json.exists()
            print(f"  config.json: {'[OK] 存在' if has_config else '[NO] 不存在'}")

            # 检查 safetensors 文件
            safetensors_files = list(path.glob("*.safetensors"))
            print(f"  safetensors 文件数量: {len(safetensors_files)}")
            if safetensors_files:
                for sf in safetensors_files[:3]:  # 只显示前3个
                    print(f"    - {sf.name}")

            # 列出所有文件
            all_files = [f.name for f in path.iterdir() if f.is_file()]
            print(f"  总文件数: {len(all_files)}")
            print(f"  文件列表: {all_files[:10]}")  # 显示前10个

            if has_config and safetensors_files:
                print(f"\n[SUCCESS] 找到完整模型! 路径: {path}")
                return str(path)
        else:
            print(f"[NO] 路径不存在")

    # 如果没找到，列出 Downloads 文件夹内容
    print(f"\n在 Downloads 文件夹中查找所有包含 'gemma' 的文件夹:")
    if downloads_folder.exists():
        for item in downloads_folder.iterdir():
            if item.is_dir() and 'gemma' in item.name.lower():
                print(f"  - {item.name}")
                # 检查是否嵌套
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        print(f"    - {subitem.name}")

    return None

if __name__ == "__main__":
    model_path = find_model_path()
    if model_path:
        print(f"\n{'='*60}")
        print(f"模型路径 (Python raw string 格式):")
        print(f'r"{model_path}"')
        print(f"{'='*60}")
    else:
        print("\n[ERROR] 未找到模型,请检查 Downloads 文件夹")
