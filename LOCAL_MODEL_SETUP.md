# 本地模型配置说明

## 完成的修改

### 1. 模型路径定位
- 本地模型位置: `C:\Users\tianyi\Downloads\gemma-2-2b-it`
- 已验证包含所有必需文件:
  - config.json
  - model-00001-of-00002.safetensors
  - model-00002-of-00002.safetensors
  - tokenizer.json
  - tokenizer.model
  - generation_config.json
  - special_tokens_map.json

### 2. 修改的文件

已将以下文件中的模型路径从 `'google/gemma-2-2b-it'` 修改为本地路径 `r"C:\Users\tianyi\Downloads\gemma-2-2b-it"`:

1. **options/option_llm.py** (第12行)
   - 修改了 `--llm-backbone` 的默认值

2. **options/option_train.py** (第21行)
   - 修改了 `--llm-backbone` 的默认值

3. **train_grpo.py** (第94行)
   - 修改了 `--llm-backbone` 的默认值

4. **run_smoke_test.py** (第323行)
   - 修改了 `llm_backbone` 变量

### 3. 已有的离线配置

`models/mllm.py` 已经正确设置了离线模式:
```python
self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_backbone, local_files_only=True)
self.llm = AutoModelForCausalLM.from_pretrained(self.args.llm_backbone, local_files_only=True)
```

## 测试验证

✅ 运行 `test_local_model_loading.py` - 成功
- Tokenizer 加载成功 (词汇表大小: 256000)
- 模型加载成功 (参数量: 2.61B)
- 简单推理测试通过

## 使用说明

现在所有脚本都会默认使用本地模型，无需联网下载。

### 运行示例:

```bash
# 测试本地模型加载
python test_local_model_loading.py

# 运行 GRPO 奖励模型测试
python test_grpo_reward.py

# 运行 GRPO 训练
python train_grpo.py [其他参数]

# 运行烟雾测试
python run_smoke_test.py
```

### 如果需要使用不同的模型路径:

可以在运行时通过命令行参数覆盖默认值:
```bash
python train_grpo.py --llm-backbone "C:\path\to\your\model"
```

## 注意事项

1. **Windows 路径**: 使用了 raw string (`r"C:\..."`) 来避免反斜杠转义问题
2. **完全离线**: 设置了 `local_files_only=True`，确保不会尝试联网下载
3. **模型位置**: 模型必须保持在 `C:\Users\tianyi\Downloads\gemma-2-2b-it` 或更新上述配置文件中的路径

## 故障排除

如果遇到模型加载错误:
1. 运行 `python locate_model.py` 确认模型路径
2. 检查模型文件夹中是否包含所有必需文件
3. 确保路径中没有多余的嵌套文件夹
