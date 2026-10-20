#!/usr/bin/env python3
"""Generate the Google Colab / Kaggle notebook from one source of truth."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/itriedcoding/satari-code/blob/main/training/Satari_Code_SFT.ipynb)"
)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


cells = [
    md(
        f"""
# Satari Code 1.5B - QLoRA SFT (Colab)

{COLAB_BADGE}

Fine-tunes **Qwen2.5-Coder-1.5B** into **Satari Code 1.5B** on
[Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K).

### Before you run

1. **Runtime -> Change runtime type -> T4 GPU**
2. Add a Hugging Face token:
   - click the key icon in the left sidebar, add a secret named `HF_TOKEN`, or
   - just paste it when the third cell asks
   - the token needs **write** access and your account needs **~$15 of HF credits**
     for the model/GGUF uploads at the end
3. **Runtime -> Run all**

### If it disconnects

Colab free sessions get cut off. Checkpoints are pushed to the Hugging Face Hub
every `--save-steps` steps, and the training cell resumes automatically.
Just run the cells again - it continues, it does not restart.
"""
    ),
    code(
        """
%%capture
import importlib.util, os, subprocess, sys

def pip(*args):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args]).returncode

IN_COLAB = importlib.util.find_spec("google.colab") is not None
IN_KAGGLE = os.path.isdir("/kaggle/input")

if IN_COLAB:
    rc = pip("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
elif IN_KAGGLE:
    rc = pip("unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git")
else:
    rc = pip("-U", "unsloth", "trl", "peft", "accelerate", "bitsandbytes", "datasets")

assert rc == 0, "unsloth install failed - rerun this cell"
"""
    ),
    code(
        """
import torch, unsloth, trl, peft, transformers, datasets

print("platform:", "colab" if IN_COLAB else ("kaggle" if IN_KAGGLE else "local"))
print("cuda    :", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("torch   :", torch.__version__)
print("unsloth :", unsloth.__version__)
print("trl     :", trl.__version__)
assert torch.cuda.is_available(), "No GPU - set runtime to T4 GPU and rerun."
"""
    ),
    code(
        """
import os

token = None
if IN_COLAB:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception as e:
        print("Colab secret not readable:", type(e).__name__)
elif IN_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("Kaggle secret not readable:", type(e).__name__)

if not token:
    from getpass import getpass
    token = getpass("Hugging Face token (hf_...): ")

assert token and token.startswith("hf_"), "Token must start with hf_"
os.environ["HF_TOKEN"] = token

from huggingface_hub import whoami
me = whoami(token=token).get("name")
print("logged in as", me)
"""
    ),
    code(
        """
from huggingface_hub import hf_hub_download
import shutil

script = hf_hub_download(
    repo_id="itriedcoding/Satari-Code-Training",
    filename="train_sft.py",
    repo_type="model",
)
shutil.copy(script, "train_sft.py")
print("trainer ->", script)
"""
    ),
    code(
        """
from huggingface_hub import HfApi
api = HfApi()
try:
    files = api.list_repo_files("itriedcoding/Satari-Code-1.5B-LoRA", repo_type="model")
    ckpts = sorted({f.split("/")[0] for f in files if "/" in f and f.startswith("checkpoint-")})
except Exception:
    ckpts = []

if ckpts:
    print("Resuming from an earlier session. Found checkpoints:")
    print("  " + ", ".join(ckpts))
else:
    print("No checkpoints yet - this is a fresh run.")
"""
    ),
    code(
        """
!python train_sft.py \\
  --output-repo  itriedcoding/Satari-Code-1.5B \\
  --adapter-repo itriedcoding/Satari-Code-1.5B-LoRA \\
  --gguf-repo    itriedcoding/Satari-Code-1.5B-GGUF \\
  --epochs 2 --batch-size 2 --grad-accum 4 \\
  --save-steps 250 \\
  --push-adapter --push-checkpoints --push-gguf \\
  --resume auto
"""
    ),
    md(
        """
## Done

The cell above publishes:

| Repo | Contents |
|---|---|
| `itriedcoding/Satari-Code-1.5B` | merged 16-bit weights |
| `itriedcoding/Satari-Code-1.5B-LoRA` | adapter + checkpoints |
| `itriedcoding/Satari-Code-1.5B-GGUF` | Q4_K_M for CPU |

If Colab cut out before the final cell finished, run the training cell again -
it resumes from the newest checkpoint in the Hub, then publishes the merged
model and GGUF.

Then from your PC:

```powershell
python training/eval_quick.py --model itriedcoding/Satari-Code-1.5B
python deploy/publish_hf.py --merged-dir outputs/satari-code/merged --gguf-dir outputs/satari-code/gguf
python deploy/publish_modelscope.py --with-studio --deploy-studio
```
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "Satari_Code_SFT.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out}")
