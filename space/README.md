---
title: Satari Code Demo
emoji: 🛠️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: apache-2.0
models:
  - itriedcoding/Satari-Code-1.5B-GGUF
short_description: Chat with the 1.5B Satari Code model on CPU.
---

# Satari Code Demo

A CPU demo for **Satari Code 1.5B**, a QLoRA fine-tune of
[Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B).

The Space downloads the `Q4_K_M` GGUF from
[`itriedcoding/Satari-Code-1.5B-GGUF`](https://huggingface.co/itriedcoding/Satari-Code-1.5B-GGUF)
and streams tokens with `llama-cpp-python`.

- Weights: 1.5B parameters, 4-bit GGUF (~1.0 GB)
- Context: 4,096 tokens at inference
- Licence: Apache-2.0
