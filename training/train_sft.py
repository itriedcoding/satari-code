#!/usr/bin/env python3
"""
Satari-Code-1.5B: supervised fine-tuning (QLoRA) of Qwen2.5-Coder-1.5B.

Target hardware: one 16 GB GPU (Kaggle T4, Colab T4, HF Jobs a10g-small).
The script is resumable, uploads the merged model, the LoRA adapter and a
GGUF quantisation to the Hugging Face Hub.

Examples
--------
Full run (Kaggle / Colab / HF Jobs):
    python train_sft.py --output-repo itriedcoding/Satari-Code-1.5B --push-gguf

Quick 5k-sample sanity run:
    python train_sft.py --max-samples 5000 --epochs 1 --push-adapter

Resume after a Kaggle session reset (checkpoints pushed to the adapter repo):
    python train_sft.py --push-checkpoints --resume itriedcoding/Satari-Code-1.5B-LoRA

Requirements: see requirements.txt. Unsloth is mandatory for the QLoRA path.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are Satari Code, a precise coding assistant. "
    "You write correct, runnable code and explain it briefly. "
    "Prefer complete solutions over fragments, state assumptions, and point out "
    "edge cases. Use fenced code blocks with the language tag."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Satari-Code QLoRA SFT")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B")
    p.add_argument("--dataset", default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--output-repo", default="itriedcoding/Satari-Code-1.5B")
    p.add_argument("--adapter-repo", default="itriedcoding/Satari-Code-1.5B-LoRA")
    p.add_argument("--gguf-repo", default="itriedcoding/Satari-Code-1.5B-GGUF")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=0, help="0 = use every row")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--output-dir", default="outputs/satari-code")
    p.add_argument("--gguf-quant", default="q4_k_m")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--resume", default=None)
    p.add_argument("--push-adapter", action="store_true")
    p.add_argument("--push-checkpoints", action="store_true")
    p.add_argument("--push-gguf", action="store_true")
    p.add_argument("--skip-merge", action="store_true")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="prepare data + model, no training")
    return p.parse_args()


def pick(row: dict, names: tuple[str, ...], default=None):
    for n in names:
        if n in row and row[n]:
            return row[n]
    return default


def resolve_resume(args, token) -> str | None:
    """`--resume <path>` resumes a local checkpoint; `--resume auto` pulls the
    newest checkpoint pushed by --push-checkpoints (Colab session survives)."""
    if not args.resume:
        return None
    if args.resume != "auto":
        if not Path(args.resume).exists():
            raise SystemExit(f"--resume path not found: {args.resume}")
        return args.resume
    if not token:
        return None

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(args.adapter_repo, repo_type="model")
    except Exception:
        print("[resume] no adapter repo yet, starting fresh")
        return None

    best_step, best_dir = -1, None
    for f in files:
        for part in f.split("/")[:-1]:
            if part.startswith("checkpoint-"):
                try:
                    step = int(part.split("-", 1)[1])
                except ValueError:
                    continue
                if step > best_step:
                    best_step, best_dir = step, part
    if best_dir is None:
        print("[resume] no checkpoints in adapter repo, starting fresh")
        return None

    print(f"[resume] fetching {best_dir} from {args.adapter_repo}")
    base = api.snapshot_download(
        args.adapter_repo, repo_type="model", allow_patterns=[f"{best_dir}/*"]
    )
    path = str(Path(base) / best_dir)
    print(f"[resume] resuming from {path}")
    return path


def build_text(tokenizer, problem: str, solution: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_tokenizer(base_model: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def prepare_dataset(args, tokenizer):
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.dataset_split)
    if args.max_samples and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
        print(f"[data] subsampled to {len(ds)} rows")

    sample = ds[0]
    problem_key = "problem" if "problem" in sample else None
    solution_key = "solution" if "solution" in sample else None
    if problem_key is None:
        problem_key = next(
            (k for k in ("instruction", "prompt", "question", "input") if k in sample),
            None,
        )
    if solution_key is None:
        solution_key = next(
            (k for k in ("output", "response", "answer", "completion") if k in sample),
            None,
        )
    if not problem_key or not solution_key:
        raise SystemExit(f"Unrecognised dataset columns: {list(sample)}")
    print(f"[data] using columns: {problem_key!r} -> {solution_key!r}")

    def batch_fn(batch):
        texts = []
        for problem, solution in zip(batch[problem_key], batch[solution_key]):
            if not problem or not solution:
                continue
            texts.append(build_text(tokenizer, str(problem), str(solution)))
        return {"text": texts}

    ds = ds.map(
        batch_fn,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=1 if os.name == "nt" else 2,
        desc="formatting",
    )
    ds = ds.filter(lambda r: bool(r["text"].strip()))
    print(f"[data] {len(ds)} training rows ready")
    return ds


def make_config(args):
    from trl import SFTConfig

    params = inspect.signature(SFTConfig.__init__).parameters
    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_ratio": 0.03,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": args.seed,
        "report_to": "none",
        "dataset_text_field": "text",
        "packing": False,
        "dataset_num_proc": 1 if os.name == "nt" else 2,
    }
    if "max_length" in params:
        kwargs["max_length"] = args.max_seq_length
    else:
        kwargs["max_seq_length"] = args.max_seq_length
    if "dataset_kwargs" in params:
        kwargs["dataset_kwargs"] = {"skip_prepare_dataset": False}
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return SFTConfig(**kwargs)


def make_trainer(model, tokenizer, dataset, config):
    from trl import SFTTrainer

    trainer_kwargs = {"model": model, "train_dataset": dataset, "args": config}
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**trainer_kwargs)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(args.output_dir, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] HF login failed: {exc}")

    try:
        import torch
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError as exc:
        print(
            "Unsloth is required for this trainer.\n"
            "  pip install -U unsloth trl peft accelerate bitsandbytes datasets\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    # Tokenizer first so data preparation works even if the model download fails.
    tokenizer = load_tokenizer(args.base_model)
    dataset = prepare_dataset(args, tokenizer)

    print(f"[model] loading {args.base_model} in 4-bit")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=True,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if args.dry_run:
        print("[dry-run] model + data are valid, exiting before training")
        return 0

    config = make_config(args)
    if args.push_checkpoints and token:
        config.hub_model_id = args.adapter_repo
        config.hub_strategy = "every_save"
        config.push_to_hub = True
        config.hub_token = token
        config.save_strategy = "steps"
    if is_bfloat16_supported():
        config.bf16 = True
        config.fp16 = False
    else:
        config.bf16 = False
        config.fp16 = True

    trainer = make_trainer(model, tokenizer, dataset, config)

    # Train only on assistant tokens. Qwen uses ChatML.
    try:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("[train] response-only masking enabled")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] response masking unavailable, training on full sequence: {exc}")

    if args.push_checkpoints and token:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.adapter_repo, private=args.private, exist_ok=True)

    resume_path = resolve_resume(args, token)
    if resume_path:
        print(f"[train] resuming, not restarting")

    print("[train] starting")
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train] adapter saved to {args.output_dir}")

    if args.push_adapter and token:
        print(f"[push] adapter -> {args.adapter_repo}")
        model.push_to_hub(args.adapter_repo, token=token, private=args.private)
        tokenizer.push_to_hub(args.adapter_repo, token=token, private=args.private)

    if not args.skip_merge:
        merged_dir = os.path.join(args.output_dir, "merged")
        print(f"[merge] writing merged 16-bit model to {merged_dir}")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        if token:
            print(f"[push] merged model -> {args.output_repo}")
            model.push_to_hub_merged(
                args.output_repo,
                tokenizer,
                save_method="merged_16bit",
                token=token,
                private=args.private,
            )

    if args.push_gguf and token:
        print(f"[gguf] {args.gguf_quant} -> {args.gguf_repo}")
        model.push_to_hub_gguf(
            args.gguf_repo,
            tokenizer,
            quantization_method=args.gguf_quant,
            token=token,
            private=args.private,
        )

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
