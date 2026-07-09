"""
train_lora_adapter_a_vlm.py

Production VLM LoRA Trainer

Model:    Qwen/Qwen2-VL-7B-Instruct
Purpose:  Planner Agent fine-tuning (RGB + FFT + metadata -> dispatch JSON)

RUN THIS ON THE TEAM'S SHARED GPU BOX OVER SSH -- NOT ON A LAPTOP CPU.
LoRA still runs a full forward+backward pass through the whole base model
every step; a 7B VLM on 2 images/sample is not viable on CPU.

Expects a dataset in the {"messages": [...]} multimodal chat format that
prepare_planner_dataset.py produces (image content items point at RGB and
FFT file paths on disk, not embedded image bytes).

Usage (on the GPU box, inside the project venv):
    python training/train_lora_adapter_a_vlm.py \
        --dataset training/data/adapter_a_training_samples.jsonl \
        --output planner_adapter_qwen

CORRECTIONS vs. previous version
---------------------------------
1. Added an explicit CUDA check before loading anything, matching the
   safeguard the old text-only script had -- previously this script would
   attempt to load a 7B model on CPU and hang/OOM with no clear message.
2. Added dataset existence / non-empty / row-count checks before loading
   the model, so a missing or empty JSONL fails in under a second instead
   of after minutes of model download + GPU allocation.
3. Validates that the image paths in the first few rows actually exist on
   disk before training starts -- catches relative-path issues (this
   script must be run from the project root, since prepare_planner_dataset.py
   writes image paths relative to it) early rather than mid-epoch.
4. Checks the installed trl version supports vision SFTTrainer (the
   multimodal `messages` + `processing_class` API is only available on
   newer trl releases) and fails with an actionable message if not.
"""

import argparse
import sys
from pathlib import Path

import torch

# --------------------------------------------------
# CUDA check -- do this before importing/loading anything heavy
# --------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA device found. VLM LoRA fine-tuning for Adapter A must run "
        "on GPU. SSH into the shared GPU box before running this script."
    )

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MIN_TRL_VERSION = (0, 12, 0)


def _check_trl_version() -> None:
    import trl

    version_str = getattr(trl, "__version__", "0.0.0")
    parts = version_str.split(".")[:3]
    try:
        version_tuple = tuple(int(p) for p in parts)
    except ValueError:
        print(f"[warn] Could not parse trl version '{version_str}'; proceeding anyway.")
        return
    if version_tuple < MIN_TRL_VERSION:
        raise RuntimeError(
            f"trl {version_str} is installed, but multimodal SFTTrainer "
            f"(messages + processing_class) needs trl >= "
            f"{'.'.join(map(str, MIN_TRL_VERSION))}. "
            "Run: pip install -U trl"
        )


def _check_dataset(path: Path) -> int:
    if not path.exists():
        raise RuntimeError(
            f"Dataset not found at {path}. Run prepare_planner_dataset.py "
            "first (and confirm organize_raw_datasets.py / prepare_dataset.py "
            "produced non-empty output before that)."
        )
    n_rows = 0
    sample_row = None
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            n_rows += 1
            if sample_row is None:
                import json

                sample_row = json.loads(line)
    if n_rows == 0:
        raise RuntimeError(
            f"Dataset at {path} exists but has zero rows. Check the output "
            "counts printed by prepare_planner_dataset.py, and the manifest "
            "row counts from the steps before it."
        )

    # Spot-check that the image paths in the first row actually resolve.
    # Catches "script run from the wrong directory" and "paths were never
    # real to begin with" before a 7B model gets loaded onto the GPU.
    if sample_row is not None:
        for msg in sample_row.get("messages", []):
            for item in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
                if item.get("type") == "image":
                    img_path = Path(item["image"])
                    if not img_path.exists():
                        raise RuntimeError(
                            f"Image path '{img_path}' from the first dataset row "
                            "doesn't exist. Image paths in the JSONL are relative "
                            "to wherever prepare_planner_dataset.py was run -- "
                            "make sure you're launching this script from the same "
                            "directory (the project root), not from training/."
                        )
    return n_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training/data/adapter_a_training_samples.jsonl")
    parser.add_argument("--output", default="planner_adapter_qwen")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    _check_trl_version()
    n_rows = _check_dataset(Path(args.dataset))
    print(f"Dataset OK: {n_rows} rows at {args.dataset}")

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    print("=" * 70)
    print("Planner VLM Fine-Tuning")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    dataset = load_dataset("json", data_files=args.dataset)["train"]

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_length=None,
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=lora_cfg,
    )

    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)

    print()
    print("=" * 70)
    print("Planner Adapter Saved")
    print(args.output)
    print("=" * 70)


if __name__ == "__main__":
    main()