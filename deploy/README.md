# Deploy

Requires the tokens in environment variables — never commit them.

| Variable | Used by |
|---|---|
| `HF_TOKEN` | `publish_hf.py` |
| `MODELSCOPE_TOKEN` | `publish_modelscope.py` |
| `VERCEL_TOKEN` | `deploy_vercel.ps1` |

## 1. Train (GPU)

Run `training/Satari_Code_SFT.ipynb` on Kaggle (T4, free) or any 16 GB GPU.
It pushes the merged model, LoRA adapter and GGUF to the Hub.

## 2. Publish to Hugging Face

```powershell
$env:HF_TOKEN = "hf_..."
python deploy/publish_hf.py `
  --merged-dir outputs/satari-code/merged `
  --gguf-dir   outputs/satari-code/gguf `
  --adapter-dir outputs/satari-code
```

## 3. Publish to ModelScope

```powershell
$env:MODELSCOPE_TOKEN = "ms-..."
python deploy/publish_modelscope.py `
  --merged-dir outputs/satari-code/merged `
  --gguf-dir   outputs/satari-code/gguf
```

## 4. Deploy the website

```powershell
$env:VERCEL_TOKEN = "vcp_..."
./deploy/deploy_vercel.ps1
```

The site is a static bundle in `web/`; Vercel needs no build step.
