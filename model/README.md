---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B
library_name: transformers
language:
  - en
tags:
  - code
  - qwen2
  - qlora
  - sft
  - satari-code
pipeline_tag: text-generation
---

# Satari Code 1.5B

> **Status:** training pipeline and model card are published; weight files are
> uploaded after the first GPU fine-tune completes. Check the
> [training repository](https://huggingface.co/itriedcoding/Satari-Code-Training)
> for the exact recipe. No benchmark numbers are claimed before measurement.

Satari Code is a **1.5B-parameter coding model** produced by QLoRA supervised
fine-tuning of [Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B)
on public code-instruction data.

- Base: `Qwen/Qwen2.5-Coder-1.5B` (Apache-2.0)
- Method: 4-bit QLoRA, r=32, alpha=32, response-only loss
- Data: [`ise-uiuc/Magicoder-OSS-Instruct-75K`](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)
- Context: 2,048 tokens during fine-tuning
- Licence: Apache-2.0

## Training

| Setting | Value |
|---|---|
| Precision | 4-bit NF4 + bf16/fp16 compute |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down |
| Epochs | 2 |
| Effective batch | 8 |
| Learning rate | 2e-4 (cosine) |
| Max sequence | 2,048 |
| Hardware | 1× 16 GB GPU (T4 / A10G) |

Reproduce with [`training/train_sft.py`](https://huggingface.co/itriedcoding/Satari-Code-Training):
see the
[`Satari_Code_SFT.ipynb`](https://huggingface.co/itriedcoding/Satari-Code-Training/blob/main/Satari_Code_SFT.ipynb)
notebook.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "itriedcoding/Satari-Code-1.5B"
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

A 4-bit GGUF (`Q4_K_M`) is available at
[`itriedcoding/Satari-Code-1.5B-GGUF`](https://huggingface.co/itriedcoding/Satari-Code-1.5B-GGUF)
for CPU inference with `llama.cpp`.

## Evaluation

Benchmark numbers are **not** claimed here. Run them yourself with your preferred
harness once the weights exist; this card will be updated with measured results
and the exact evaluation command.

## Limitations

- 1.5B parameters: weaker than 7B+ models on long-horizon reasoning and large refactors.
- Fine-tuned on 2,048-token samples; very large files will be truncated.
- Can produce insecure or subtly incorrect code. Review all output before running it.
- Primarily English prompts.

## Licence

Apache-2.0, inherited from the base model.
