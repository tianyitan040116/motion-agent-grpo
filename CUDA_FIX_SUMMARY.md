# 硬编码 .cuda() 修复总结

## 修复概述
已将项目中所有硬编码的 `.cuda()` 调用替换为更灵活的设备管理方式，支持自动检测 GPU/CPU。

## 修改的文件清单

### 1. **models/quantize_cnn.py** ✅
- **修改位置**: 第18行，第313行
- **修改内容**: 
  - `QuantizeEMAReset.reset_codebook()`: 移除 `.cuda()`
  - `QuantizeEMA.reset_codebook()`: 移除 `.cuda()`
- **说明**: 使用 `register_buffer` 注册的张量会自动跟随模型的 `.to(device)` 移动到正确设备

### 2. **models/mllm.py** ✅
- **修改位置**: 第156行
- **修改内容**: 
  - 从 `input_ids = self.tokenizer.encode(input, return_tensors="pt").cuda()`
  - 改为 `input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.device)`
- **说明**: 使用类的 `self.device` 属性动态选择设备

### 3. **dataset/evaluator.py** ✅
- **修改位置**: 
  - `generate_one_motion_IG_rvq`: 第23, 25, 33行
  - `generate_one_motion_IG_rvq_v1_1`: 第79, 81, 89行
  - `generate_one_motion_IG_rvq_v3`: 第135, 137, 145, 157行
  - `EvaluationDataset.__iter__`: 第219行
- **修改内容**: 
  - 在每个函数开头添加 `device = next(gemma_model.parameters()).device`
  - 所有 `.cuda()` 改为 `.to(device)`
- **说明**: 从传入的模型参数中推断设备，确保数据与模型在同一设备

### 4. **demo.py** ✅
- **修改位置**: 第28, 36行
- **修改内容**:
  - 添加设备检测：`device = torch.device(args.device if torch.cuda.is_available() else 'cpu')`
  - `model.llm.cuda()` → `model.llm.to(device)`
  - `torch.from_numpy(motion).float().cuda()` → `torch.from_numpy(motion).float().to(device)`
- **说明**: 在演示代码中使用智能设备选择

### 5. **utils/quaternion.py** ✅
- **修改位置**: 第142行
- **修改内容**:
  - 添加设备检测：`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
  - `.cuda()` → `.to(device)`
- **说明**: 仅在 `use_gpu=True` 时才尝试使用 GPU，且会自动降级到 CPU

### 6. **test_local_model_loading.py** ✅
- **修改位置**: 第67行
- **修改内容**: `v.cuda()` → `v.to(device)`
- **说明**: 测试脚本中的设备一致性

## 未修改的文件

以下文件包含 `.cuda()` 但不需要修改：
- 下载脚本（download_*.py）- 不涉及运行时
- 其他注释掉的代码

## 修复后的优势

### 1. **自动设备检测**
代码现在会自动检测是否有可用的 GPU：
- 有 GPU → 使用 GPU
- 无 GPU → 自动降级到 CPU
- 不会因为缺少 CUDA 而崩溃

### 2. **更好的可移植性**
代码可以在以下环境中无缝运行：
- 有 NVIDIA GPU 的机器（CUDA）
- 没有 GPU 的机器（CPU）
- Apple Silicon Mac（MPS，如果需要可以扩展）
- 服务器环境

### 3. **遵循 PyTorch 最佳实践**
- 使用 `torch.device` 对象
- 使用 `.to(device)` 而非硬编码 `.cuda()`
- Buffer 通过 `register_buffer` 自动管理

## 测试验证

修复后已测试：
- ✅ CPU 环境下模型加载成功（test_local_model_loading.py）
- ✅ 代码不再因缺少 CUDA 而在初始化时崩溃
- ⏳ GPU 环境测试（需要重装 CUDA 版 PyTorch）

## 下一步操作

1. **安装 CUDA 版 PyTorch**
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

2. **验证 GPU 可用性**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **运行完整测试**
   ```bash
   python test_grpo_reward.py
   ```

## 技术细节

### register_buffer 自动设备迁移
```python
# 修改前
self.register_buffer('codebook', torch.zeros(...).cuda())  # 硬编码到 GPU

# 修改后
self.register_buffer('codebook', torch.zeros(...))  # 在 CPU 创建
# 当调用 model.to('cuda') 时，buffer 会自动迁移
```

### 从模型推断设备
```python
# 获取模型当前所在设备
device = next(model.parameters()).device
# 确保数据在同一设备
tensor = torch.tensor(data).to(device)
```

这种方式确保数据和模型始终在同一设备上，避免设备不匹配错误。

## 参考文档
- [PyTorch 设备管理最佳实践](https://pytorch.org/docs/stable/notes/cuda.html)
- [PYTORCH_CUDA_INSTALLATION.md](./PYTORCH_CUDA_INSTALLATION.md) - PyTorch CUDA 版本安装指南
