# Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itriedcoding/satari-code/blob/main/training/Satari_Code_SFT.ipynb)

## Why not on this PC?

The target is a 1.5B-parameter model. Training state in FP32 is roughly:

```
weights 6 GB + gradients 6 GB + Adam m/v 12 GB = 24 GB
```

This machine has 16 GB RAM (≈13 GB usable), no CUDA, and measures ~66 GFLOPS.
Even a minimal 10B-token run would take ≈43 years at peak throughput. Use a GPU.

## Free path: Google Colab (recommended)

Click the badge above, then:

1. **Runtime → Change runtime type → T4 GPU**
2. Add a secret named `HF_TOKEN` (key icon, left sidebar) with a **write** token.
   Your account also needs ~$15 of HF credits for the uploads at the end.
3. **Runtime → Run all**

2 epochs / 75k rows takes roughly **5–9 hours**. Colab free sessions get cut off,
so checkpoints are pushed to the Hub every 250 steps and the training cell
resumes automatically on re-run — you can finish across several sessions.

## Free path: Kaggle T4 (alternative)

1. Download [`Satari_Code_SFT.ipynb`](Satari_Code_SFT.ipynb).
2. On [kaggle.com](https://www.kaggle.com/code) → **New Notebook → File → Import
   Notebook**.
3. **Accelerator:** GPU T4 ×1, **Internet:** On. Add secret `HF_TOKEN`. Run all.

Kaggle allows 30 GPU hours/week, so a full run fits.

## Command line

```bash
pip install -U unsloth trl peft accelerate bitsandbytes datasets
export HF_TOKEN=hf_...

python train_sft.py \
  --output-repo  itriedcoding/Satari-Code-1.5B \
  --adapter-repo itriedcoding/Satari-Code-1.5B-LoRA \
  --gguf-repo    itriedcoding/Satari-Code-1.5B-GGUF \
  --epochs 2 --batch-size 2 --grad-accum 4 \
  --push-adapter --push-gguf
```

Outputs:

| Artifact | Destination |
|---|---|
| merged 16-bit model | `Satari-Code-1.5B` |
| LoRA adapter | `Satari-Code-1.5B-LoRA` |
| GGUF `Q4_K_M` | `Satari-Code-1.5B-GGUF` |

## Sanity checks (CPU, no GPU)

```bash
python smoke_test.py        # tiny model: real forward + backward, loss must fall
python build_notebook.py    # regenerate the notebook
```

`smoke_test.py` is the same data formatter the trainer uses, including the real
Qwen ChatML template.

## Evaluation

`eval_quick.py` generates a fixed transcript for manual review:

```bash
python eval_quick.py --model itriedcoding/Satari-Code-1.5B
```

It intentionally does not invent benchmark numbers. Use
`bigcode-evaluation-harness` for pass@k once the weights exist.
