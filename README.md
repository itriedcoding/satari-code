# Satari Code

A 1.5B-parameter coding model, built on
[Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) and tuned
with QLoRA on public code instructions.

- **Model:** `itriedcoding/Satari-Code-1.5B` (merged), `-LoRA`, `-GGUF` (Q4_K_M)
- **License:** Apache-2.0
- **Website:** https://satari-code.vercel.app
- **Training code:** https://huggingface.co/itriedcoding/Satari-Code-Training

## Status

| Item | State |
|---|---|
| Data pipeline (75k public instructions) | done, verified |
| Resumable QLoRA SFT trainer | done, CPU smoke test passed |
| Colab one-click training notebook | published |
| 1.5B GPU fine-tune | **waiting on you to hit Run All in Colab** |
| Merged weights + GGUF | pending training |
| HF Space demo | code ready, pending GGUF |
| ModelScope model + Studio | code ready, **token invalid** |
| Website | live on Vercel |

This machine (Ryzen 3 3300U, 16 GB RAM, no CUDA, ~66 GFLOPS) cannot train a 1.5B
model: the FP32 training state alone is ~24 GB, and a minimal 10B-token run would
take ~43 years at peak throughput. The model is therefore trained on a GPU — a
free Kaggle/Colab T4 is enough.

## Layout

```
training/   SFT trainer, eval, smoke test, Kaggle/Colab notebook
space/      Hugging Face Space (Gradio + llama.cpp, CPU)
msstudio/   ModelScope Studio (Gradio + llama.cpp)
web/        static site deployed to Vercel
deploy/     Hub / ModelScope / Vercel publish scripts
model/      model card
```

## Train

[Open the notebook in Colab](https://colab.research.google.com/github/itriedcoding/satari-code/blob/main/training/Satari_Code_SFT.ipynb),
select a **T4 GPU**, add an `HF_TOKEN` secret, then **Runtime → Run All**.
If it disconnects, run the training cell again — it resumes from the last
checkpoint pushed to the Hub.

Command-line equivalent:

```bash
pip install -U unsloth trl peft accelerate bitsandbytes datasets
export HF_TOKEN=hf_...
python training/train_sft.py \
  --output-repo  itriedcoding/Satari-Code-1.5B \
  --adapter-repo itriedcoding/Satari-Code-1.5B-LoRA \
  --gguf-repo    itriedcoding/Satari-Code-1.5B-GGUF \
  --epochs 2 --push-adapter --push-gguf
```

## Verify locally (no GPU needed)

```bash
python training/smoke_test.py      # tiny model, real forward/backward
python training/build_notebook.py  # regenerate the notebook
```

## Publish

See [`deploy/README.md`](deploy/README.md). Tokens are read from
`HF_TOKEN`, `MODELSCOPE_TOKEN`, and `VERCEL_TOKEN`; none are stored in the repo.
