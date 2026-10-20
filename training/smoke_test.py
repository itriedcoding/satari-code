#!/usr/bin/env python3
"""CPU smoke test: verifies data formatting and a real forward/backward pass.

This does not train the 1.5B model (impossible on CPU in reasonable time).
It proves the pipeline's code path runs end to end on a tiny random model.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from train_sft import SYSTEM_PROMPT, build_text


def main() -> None:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    text = build_text(tok, "Write a Python factorial.", "def fact(n):\n    return 1 if n < 2 else n * fact(n - 1)")
    assert "<|im_start|>assistant" in text, "chat template did not render"
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128)
    print(f"[ok] chat template rendered ({ids['input_ids'].shape[1]} tokens)")

    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = Qwen2ForCausalLM(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    labels = ids["input_ids"].clone()
    before = None
    for step in range(3):
        out = model(**ids, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        if before is None:
            before = out.loss.item()
        print(f"[ok] step {step + 1}: loss={out.loss.item():.4f}")
    assert out.loss.item() < before, "loss did not decrease"
    print("[ok] loss decreased; pipeline is functional")


if __name__ == "__main__":
    main()
