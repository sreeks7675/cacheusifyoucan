"""LoRA configuration for Adapter A — orchestration & tool-calling.

Shared by the Planner Agent (1) and Retrieval Agent (4), per design_document.md
Section 2. `to_peft_config()` lazily imports peft so this module can be imported
in tests/CI without peft installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AdapterAConfig:
    r: int = 16  # rank — 16 balances quality vs. VRAM/training-time for a one-day timeline
    lora_alpha: int = 32  # 2x rank is a safe, common default
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention — routing decisions
            "gate_proj", "up_proj", "down_proj",  # MLP — helps strict JSON formatting fidelity
        ]
    )
    task_type: str = "CAUSAL_LM"
    bias: str = "none"

    def to_peft_config(self):
        from peft import LoraConfig

        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            task_type=self.task_type,
            bias=self.bias,
        )
