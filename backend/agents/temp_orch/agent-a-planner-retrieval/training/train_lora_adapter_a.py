"""
train_lora_adapter_a.py  --  DEPRECATED, DO NOT USE

This was the original text-only trainer (Phi-4-mini-instruct,
AutoModelForCausalLM). It is incompatible with the current pipeline for
two separate reasons:

1. The Planner is a vision-language model (Qwen2-VL-7B-Instruct). This
   script has no image-handling path at all.
2. The current training data (training/data/adapter_a_training_samples.jsonl,
   produced by prepare_planner_dataset.py) is written as
   {"messages": [...]} multimodal rows. This script's SFTTrainer call
   expects dataset_text_field="text", which doesn't exist in that file
   -- it will fail immediately on a KeyError, not train on garbage, but
   there's no reason to hit that error path at all.

Use train_lora_adapter_a_vlm.py instead:

    python training/train_lora_adapter_a_vlm.py \
        --dataset training/data/adapter_a_training_samples.jsonl \
        --output planner_adapter_qwen

This file is kept only so old invocations fail fast with a clear message
instead of a confusing stack trace.
"""

import sys

if __name__ == "__main__":
    sys.exit(
        "train_lora_adapter_a.py is deprecated -- use "
        "train_lora_adapter_a_vlm.py instead. See this file's docstring "
        "for why the text-only path no longer applies."
    )