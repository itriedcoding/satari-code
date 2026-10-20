"""Satari Code 1.5B — Hugging Face Space (CPU, GGUF).

Loads the Q4_K_M GGUF and streams tokens so the UI stays responsive.
"""

from __future__ import annotations

import os

import gradio as gr

MODEL_REPO = os.environ.get("MODEL_REPO", "itriedcoding/Satari-Code-1.5B-GGUF")
GGUF_HINT = os.environ.get("GGUF_HINT", "Q4_K_M")
SYSTEM = (
    "You are Satari Code, a precise coding assistant. "
    "You write correct, runnable code and explain it briefly. "
    "Prefer complete solutions over fragments, state assumptions, and point out "
    "edge cases. Use fenced code blocks with the language tag."
)


def find_gguf() -> str:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(MODEL_REPO, repo_type="model")
    matches = [f for f in files if f.endswith(".gguf") and GGUF_HINT in f]
    if not matches:
        matches = [f for f in files if f.endswith(".gguf")]
    if not matches:
        raise RuntimeError(f"No GGUF file found in {MODEL_REPO}")
    return hf_hub_download(MODEL_REPO, matches[0])


print(f"[boot] fetching {MODEL_REPO}")
from llama_cpp import Llama  # noqa: E402

llm = Llama(
    model_path=find_gguf(),
    n_ctx=4096,
    n_threads=max(1, (os.cpu_count() or 4)),
    n_batch=256,
    verbose=False,
)
print("[boot] model ready")


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
        "Runs locally as a 4-bit GGUF. Expect a short warm-up on free CPU hardware."
    ),
    examples=[
        "Write a Python function that merges two sorted lists in O(n).",
        "Why does this React effect run twice, and how do I fix it?",
        "Add type hints and tests to a function that parses ISO-8601 dates.",
    ],
    cache_examples=False,
    theme=gr.themes.Base(
        primary_hue="lime",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(body_background_fill="#08090b", block_background_fill="#0e1014"),
)

if __name__ == "__main__":
    demo.queue(max_size=24).launch()
