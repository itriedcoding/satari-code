#!/usr/bin/env python3
"""Publish Satari Code to ModelScope (modelscope.ai) and create a Studio.

Model upload uses the Python SDK (endpoint https://modelscope.ai).
Studio creation/deploy uses the ModelScope CLI, which knows about Studio repos.

    $env:MODELSCOPE_TOKEN = "ms-..."
    python deploy/publish_modelscope.py `
        --merged-dir outputs/satari-code/merged `
        --gguf-dir   outputs/satari-code/gguf `
        --studio-dir msstudio

Studio deploy is skipped unless --only studio or --with-studio is passed, so you
can publish weights before the demo exists.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://modelscope.ai"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="NeuraAI/Satari-Code-1.5B")
    p.add_argument("--studio-id", default="NeuraAI/Satari-Code-Demo")
    p.add_argument("--merged-dir", default=None)
    p.add_argument("--gguf-dir", default=None)
    p.add_argument("--studio-dir", default=str(ROOT / "msstudio"))
    p.add_argument("--with-studio", action="store_true")
    p.add_argument("--deploy-studio", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd[:4]), "...")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode})")


def main() -> int:
    args = parse_args()
    token = os.environ.get("MODELSCOPE_TOKEN")
    if not token:
        print("Set MODELSCOPE_TOKEN first.", file=sys.stderr)
        return 2

    from modelscope.hub.api import HubApi

    api = HubApi(endpoint=ENDPOINT)
    api.login(access_token=token)

    api.create_model(args.model_id, visibility=5, license="Apache License 2.0")
    for folder in (args.merged_dir, args.gguf_dir):
        if not folder or not Path(folder).exists():
            print(f"[skip] {folder}")
            continue
        print(f"[push] {folder} -> {args.model_id}")
        api.upload_folder(
            repo_id=args.model_id,
            folder_path=folder,
            repo_type="model",
            commit_message=f"Upload {Path(folder).name}",
        )

    if args.with_studio:
        if not Path(args.studio_dir).exists():
            print(f"Studio directory missing: {args.studio_dir}", file=sys.stderr)
            return 2
        base = ["modelscope", "--token", token, "--endpoint", ENDPOINT]
        run(base + ["create", args.studio_id, "--repo-type", "studio",
                    "--sdk-type", "gradio", "--exist-ok"])
        print(f"[push] {args.studio_dir} -> {args.studio_id}")
        api.upload_folder(
            repo_id=args.studio_id,
            folder_path=args.studio_dir,
            repo_type="space",
            commit_message="Add Studio app",
        )
        if args.deploy_studio:
            run(base + ["deploy", args.studio_id, "--repo-type", "studio"])

    print("\nDone:")
    print(f"  model  : {ENDPOINT}/models/{args.model_id}")
    if args.with_studio:
        print(f"  studio : {ENDPOINT}/studios/{args.studio_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
