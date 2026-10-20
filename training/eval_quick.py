#!/usr/bin/env python3
"""Quick, honest evaluation: generate answers and save them for inspection.

This is not a benchmark. It produces a reproducible transcript you can read,
and leaves pass@k measurement to a dedicated harness (e.g. bigcode-evaluation-harness).

    python eval_quick.py --model itriedcoding/Satari-Code-1.5B --out eval_results.json
"""

from __future__ import annotations

import argparse
import json

PROMPTS = [
    "Write a Python function that returns the k most frequent words in a text, "
    "ignoring case and punctuation. Include a short explanation.",
    "Given a list of intervals, merge all overlapping intervals. "
    "Return the result sorted by start time. Use TypeScript.",
    "This function is supposed to debounce a callback but fires every time. "
    "Explain why and fix it:\n\n```js\nfunction debounce(fn, ms) {\n  return function (...args) {\n    setTimeout(() => fn(...args), ms);\n  };\n}\n```",
    "Write a SQL query that finds the second highest salary per department "
    "from employees(id, department_id, salary). Explain the NULL edge case.",
    "Add type hints and pytest tests to a function that parses ISO-8601 dates "
    "and returns timezone-aware datetimes.",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="itriedcoding/Satari-Code-1.5B")
    p.add_argument("--out", default="eval_results.json")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device
    )
    model.eval()

    system = (
        "You are Satari Code, a precise coding assistant. "
        "You write correct, runnable code and explain it briefly."
    )
    results = []
    for prompt in PROMPTS:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs.shape[1] :], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": text})
        print("=" * 80)
        print(prompt)
        print("-" * 80)
        print(text)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
