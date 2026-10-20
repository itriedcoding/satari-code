---
license: Apache License 2.0
tasks:
  - text-generation
frameworks:
  - PyTorch
tags:
  - code
  - satari-code
  - qwen2
  - qlora
---

# Satari Code 1.5B

> **Status:** training pipeline and model card are published; weight files are
> uploaded after the first GPU fine-tune completes. No benchmark numbers are
> claimed before measurement.

Satari Code is a **1.5B-parameter coding model** produced by QLoRA supervised
fine-tuning of Qwen2.5-Coder-1.5B on public code-instruction data.

- Base: `Qwen/Qwen2.5-Coder-1.5B` (Apache-2.0)
- Method: 4-bit QLoRA, r=32, alpha=32, response-only loss
- Data: `ise-uiuc/Magicoder-OSS-Instruct-75K`
- Context: 2,048 tokens during fine-tuning
- Licence: Apache-2.0

## Training

| Setting | Value |
|---|---|
| Precision | 4-bit NF4 + bf16/fp16 compute |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| Epochs | 2 |
| Effective batch | 8 |
| Learning rate | 2e-4 (cosine) |
| Max sequence | 2,048 |
| Hardware | 1× 16 GB GPU (T4 / A10G) |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "NeuraAI/Satari-Code-1.5B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [
    {"role": "system", "content": "You are Satari Code, a precise coding assistant."},
    {"role": "user", "content": "Write a Python LRU cache with tests."},
]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.95)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

A 4-bit GGUF (`Q4_K_M`) will be published alongside the merged weights for CPU
inference with `llama.cpp`.

## Limitations

- 1.5B parameters: weaker than 7B+ models on long-horizon reasoning.
- Fine-tuned on 2,048-token samples.
- Can produce incorrect or insecure code. Review before running.
- Primarily English prompts.

## Licence

Apache-2.0, inherited from the base model.
