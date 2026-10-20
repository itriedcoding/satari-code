"""Satari Code 1.5B — ModelScope Studio (Gradio).

Tries ModelScope first for the GGUF, then falls back to the Hugging Face Hub.
"""

from __future__ import annotations

import glob
import os

import gradio as gr

MS_MODEL_ID = os.environ.get("MS_MODEL_ID", "itriedcoding/Satari-Code-1.5B-GGUF")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "itriedcoding/Satari-Code-1.5B-GGUF")
GGUF_HINT = os.environ.get("GGUF_HINT", "Q4_K_M")
SYSTEM = (
    "You are Satari Code, a precise coding assistant. "
    "You write correct, runnable code and explain it briefly. "
    "Prefer complete solutions over fragments, state assumptions, and point out "
    "edge cases. Use fenced code blocks with the language tag."
)


def via_modelscope() -> str | None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        d = snapshot_download(MS_MODEL_ID, allow_file_pattern=f"*{GGUF_HINT}*.gguf")
        hits = glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True)
        return hits[0] if hits else None
    except Exception as exc:
        print(f"[modelscope] {exc}")
        return None


def via_hf() -> str | None:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list_repo_files(HF_MODEL_ID)
        hits = [f for f in files if f.endswith(".gguf") and GGUF_HINT in f]
        if not hits:
            hits = [f for f in files if f.endswith(".gguf")]
        return hf_hub_download(HF_MODEL_ID, hits[0]) if hits else None
    except Exception as exc:
        print(f"[hf] {exc}")
        return None


path = via_modelscope() or via_hf()
if not path:
    raise RuntimeError("Could not locate a GGUF checkpoint.")

print(f"[boot] loading {path}")
from llama_cpp import Llama  # noqa: E402

llm = Llama(
    model_path=path,
    n_ctx=4096,
    n_threads=max(1, os.cpu_count() or 4),
    n_batch=256,
    verbose=False,
)


def to_messages(history) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    for item in history or []:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]:
                messages.append({"role": "user", "content": str(item[0])})
            if item[1]:
                messages.append({"role": "assistant", "content": str(item[1])})
    return messages


def chat(message: str, history):
    messages = to_messages(history)
    messages.append({"role": "user", "content": message})
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=768,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.05,
        stream=True,
    )
    out = ""
    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {}).get("content", "")
        if delta:
            out += delta
            yield out


demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="Satari Code 1.5B",
    description=(
        "A 1.5B coding model fine-tuned from Qwen2.5-Coder-1.5B. "
        "4-bit GGUF, CPU friendly."
    ),
    examples=[
        "Write a Python function that merges two sorted lists in O(n).",
        "Explain this SQL and rewrite it to avoid a full scan.",
        "Add type hints and tests to a function that parses ISO-8601 dates.",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.queue(max_size=24).launch()
