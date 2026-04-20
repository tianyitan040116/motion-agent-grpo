# 快速开始指南 - 安装 CUDA 版 PyTorch 并运行

## 🚀 一键安装命令

在你的虚拟环境 `venv_grpo` 中，按顺序运行以下命令：

### 1. 卸载 CPU 版 PyTorch
```bash
pip uninstall torch torchvision torchaudio -y
```

### 2. 安装 CUDA 12.4 版 PyTorch（推荐）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**如果官方源太慢，使用清华镜像：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 验证安装
```bash
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**预期输出：**
```
✓ PyTorch: 2.6.0+cu124
✓ CUDA available: True
✓ GPU: NVIDIA GeForce RTX 4090 Laptop GPU
```

## 🧪 运行测试

安装完成后，按顺序运行以下测试：

### 测试 1: 本地模型加载（GPU）
```bash
python test_local_model_loading.py
```
应该显示 "使用设备: cuda:0" 并成功加载模型。

### 测试 2: GRPO 奖励模型
```bash
python test_grpo_reward.py
```
应该能加载 VQ-VAE、evaluator 并计算奖励。

### 测试 3: 烟雾测试
```bash
python run_smoke_test.py
```

## 📊 性能预期

使用 RTX 4090 GPU 后的改进：
- **模型加载**: ~5秒（vs CPU ~30-60秒）
- **推理速度**: 20-50x 加速
- **训练速度**: 50-100x 加速
- **显存占用**: ~4-6GB（2.6B 参数的 Gemma 模型）

## ❓ 常见问题

### Q1: 安装后还是显示 "CUDA available: False"？
**解决方案：**
```bash
# 1. 重启终端
# 2. 重新激活虚拟环境
venv_grpo/Scripts/activate  # Windows
# 3. 再次检查
python -c "import torch; print(torch.cuda.is_available())"
```

### Q2: 安装很慢或超时？
**解决方案：** 使用国内镜像
```bash
# 方案1: 清华镜像
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方案2: 阿里云镜像  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://mirrors.aliyun.com/pypi/simple/
```

### Q3: 显存不足错误？
**解决方案：**
```python
# 在脚本开头添加
import torch
torch.cuda.empty_cache()

# 或者减小 batch size
# 在命令行参数中添加 --batch-size 2
```

### Q4: 多 GPU 如何选择特定 GPU？
**解决方案：**
```bash
# 使用 GPU 0
export CUDA_VISIBLE_DEVICES=0
python your_script.py

# 使用 GPU 1
export CUDA_VISIBLE_DEVICES=1
python your_script.py
```

## 📝 修改总结

我已经完成了以下工作：

### ✅ 任务 1: 定位本地模型
- 找到模型路径：`C:\Users\tianyi\Downloads\gemma-2-2b-it`
- 验证文件完整性（config.json, safetensors, tokenizer等）

### ✅ 任务 2: 修复硬编码 .cuda()
修改了 **6 个文件**，**17 处**硬编码：
1. `models/quantize_cnn.py` - 2 处
2. `models/mllm.py` - 1 处
3. `dataset/evaluator.py` - 11 处
4. `demo.py` - 2 处
5. `utils/quaternion.py` - 1 处
6. `test_local_model_loading.py` - 1 处

所有修改遵循 PyTorch 最佳实践：
- ✓ 使用 `torch.device` 对象
- ✓ 使用 `.to(device)` 替代 `.cuda()`
- ✓ 从模型推断设备：`device = next(model.parameters()).device`
- ✓ 支持 GPU/CPU 自动降级

### ✅ 任务 3: 提供 PyTorch 安装命令
- 分析硬件：RTX 4090 + CUDA 13.2
- 推荐版本：PyTorch 2.6.0 + CUDA 12.4
- 提供详细安装步骤和备选方案

## 🎯 下一步

1. **立即执行**：运行上面的安装命令
2. **测试验证**：运行三个测试脚本
3. **开始训练**：运行 GRPO 训练
   ```bash
   python train_grpo.py --batch-size 4 --epochs 100
   ```

## 📚 相关文档

- [CUDA_FIX_SUMMARY.md](./CUDA_FIX_SUMMARY.md) - 详细修改总结
- [PYTORCH_CUDA_INSTALLATION.md](./PYTORCH_CUDA_INSTALLATION.md) - PyTorch 安装指南
- [LOCAL_MODEL_SETUP.md](./LOCAL_MODEL_SETUP.md) - 本地模型配置说明

---

**提示**：安装过程大约需要 10-30 分钟（取决于网络速度），PyTorch CUDA 版本约 2-3GB。建议使用稳定的网络连接。
