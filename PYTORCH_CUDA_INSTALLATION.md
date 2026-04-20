# PyTorch CUDA 版本重装指南

## 环境信息
- **GPU**: NVIDIA GeForce RTX 4090
- **CUDA 驱动版本**: 13.2
- **当前 Python**: 3.13.9 (venv_grpo)
- **当前 PyTorch**: 2.11.0+cpu (仅 CPU)

## 推荐的 PyTorch 版本
由于你的 CUDA 驱动是 13.2，但 PyTorch 官方最高支持 CUDA 12.4，
我们将安装 **PyTorch 2.6.0 (CUDA 12.4)** 版本。

CUDA 驱动 13.2 完全向后兼容 CUDA 12.4，所以可以正常使用。

## 重装步骤

### 步骤 1: 卸载当前的 PyTorch (CPU 版本)

```bash
pip uninstall torch torchvision torchaudio -y
```

### 步骤 2: 安装 CUDA 12.4 版本的 PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**注意**: 这个命令会安装与 CUDA 12.4 兼容的 PyTorch 2.6.0 版本。

### 步骤 3: 验证安装

安装完成后，运行以下命令验证 CUDA 是否可用：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

预期输出类似：
```
PyTorch: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
GPU: NVIDIA GeForce RTX 4090 Laptop GPU
```

### 步骤 4: 重新测试模型加载

安装完成后，运行：

```bash
python test_local_model_loading.py
```

这次应该会使用 GPU 加载模型，速度会快很多。

## 备选方案

如果上述命令因网络问题失败，可以使用清华镜像源：

```bash
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple --index-url https://download.pytorch.org/whl/cu124
```

或者使用阿里云镜像：

```bash
pip install torch torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/ --index-url https://download.pytorch.org/whl/cu124
```

## 常见问题

### Q: 为什么不安装 CUDA 13.2 版本的 PyTorch?
A: PyTorch 官方尚未发布 CUDA 13.x 的预编译版本。CUDA 12.4 在你的 CUDA 13.2 驱动上可以完美运行。

### Q: 安装后还是显示 CUDA not available?
A: 检查以下几点：
1. 确认 nvidia-smi 能正常运行
2. 重启终端/Python 环境
3. 检查是否在虚拟环境中（`which python` 应该指向 venv_grpo）
4. 检查 NVIDIA 驱动是否最新

### Q: 安装很慢怎么办?
A: PyTorch CUDA 版本约 2-3GB，下载可能需要一些时间。可以：
1. 使用国内镜像源（见备选方案）
2. 等待下载完成，通常 10-30 分钟

## 安装后的测试

成功安装 CUDA 版 PyTorch 后，可以运行：

```bash
# 测试本地模型加载
python test_local_model_loading.py

# 测试 GRPO 奖励模型
python test_grpo_reward.py

# 运行烟雾测试
python run_smoke_test.py
```

所有 `.cuda()` 的硬编码已经被修复，模型会自动使用正确的设备（GPU 或 CPU）。
