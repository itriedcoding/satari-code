---
domain:
  - text
tags:
  - code
  - qwen2
  - gguf
  - gradio
license: Apache License 2.0
---

# Satari Code Studio

ModelScope Studio for **Satari Code 1.5B**, a QLoRA fine-tune of
Qwen2.5-Coder-1.5B.

The app downloads the `Q4_K_M` GGUF checkpoint and streams responses with
`llama-cpp-python`. If the ModelScope model repo is unavailable it falls back to
the Hugging Face Hub.
