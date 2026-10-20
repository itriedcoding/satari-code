#!/usr/bin/env python3
"""Publish Satari Code to the Hugging Face Hub.

Uploads (all optional):
  * merged 16-bit model      -> itriedcoding/Satari-Code-1.5B
  * GGUF quantisations       -> itriedcoding/Satari-Code-1.5B-GGUF
  * LoRA adapter             -> itriedcoding/Satari-Code-1.5B-LoRA
  * Gradio Space             -> itriedcoding/Satari-Code-Demo
  * training code            -> itriedcoding/Satari-Code-Training

Usage:
    $env:HF_TOKEN = "hf_..."
    python deploy/publish_hf.py --merged-dir outputs/satari-code/merged \
        --gguf-dir outputs/satari-code/gguf --space-dir space --training-dir training
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="itriedcoding/Satari-Code-1.5B")
    p.add_argument("--gguf-id", default="itriedcoding/Satari-Code-1.5B-GGUF")
    p.add_argument("--adapter-id", default="itriedcoding/Satari-Code-1.5B-LoRA")
    p.add_argument("--training-id", default="itriedcoding/Satari-Code-Training")
    p.add_argument("--space-id", default="itriedcoding/Satari-Code-Demo")
    p.add_argument("--merged-dir", default=None)
    p.add_argument("--gguf-dir", default=None)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--training-dir", default=str(ROOT / "training"))
    p.add_argument("--space-dir", default=str(ROOT / "space"))
    p.add_argument("--model-card", default=str(ROOT / "model" / "README.md"))
    p.add_argument("--private", action="store_true")
    p.add_argument("--skip-space", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    return p.parse_args()


def upload_folder(api, folder: str | None, repo_id: str, repo_type: str, private: bool, ignore=None):
    if not folder:
        return
    path = Path(folder)
    if not path.exists():
        print(f"[skip] {path} does not exist")
        return
    print(f"[push] {path} -> {repo_id} ({repo_type})")
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=ignore or ["__pycache__/*", "*.pyc", ".ipynb_checkpoints/*"],
    )


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN first.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()
    print(f"[auth] {who.get('name')}")

    # 1. Model card + merged weights
    api.create_repo(args.model_id, private=args.private, exist_ok=True)
    card = Path(args.model_card)
    if card.exists():
        print(f"[push] model card -> {args.model_id}")
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.model_id,
        )
    upload_folder(api, args.merged_dir, args.model_id, "model", args.private)

    # 2. Adapter
    upload_folder(api, args.adapter_dir, args.adapter_id, "model", args.private)

    # 3. GGUF
    upload_folder(api, args.gguf_dir, args.gguf_id, "model", args.private)

    # 4. Training code
    if not args.skip_training:
        upload_folder(api, args.training_dir, args.training_id, "model", args.private)

    # 5. Space
    if not args.skip_space:
        print(f"[push] Space -> {args.space_id}")
        api.create_repo(
            args.space_id,
            repo_type="space",
            space_sdk="gradio",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(Path(args.space_dir)),
            repo_id=args.space_id,
            repo_type="space",
            ignore_patterns=["__pycache__/*", "*.pyc"],
        )

    print("\nDone:")
    print(f"  model : https://huggingface.co/{args.model_id}")
    print(f"  gguf  : https://huggingface.co/{args.gguf_id}")
    if not args.skip_space:
        print(f"  space : https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
