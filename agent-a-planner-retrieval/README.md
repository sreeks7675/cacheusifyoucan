# Adapter A workspace — Planner Agent (1) + Retrieval Agent (4)

Your scope: Agents 1 and 4, both driven by Adapter A (orchestration & tool-calling),
plus the LangGraph orchestration/backend API work that comes after this. This repo
is self-contained enough to build, test, and demo those two agents independently
of your teammates' branches, then plug into `dev` per `design_document.md`.

## Core design principle: the Planner is a shallow generalist

The Planner Agent takes the **image itself**, not a pre-summarized text string. It
scans across 14 surface-level categories (6 forensic, 5 semantic, 3 retrieval — see
`src/agent_a/schemas/task_plan.py`'s `IssueCategory` enum) and decides which of
exactly three agents to call: `forensic_analysis`, `semantic_context`,
`retrieval_comparison`. It never runs deep analysis itself — that's the whole point
of the "jack of all trades, surface only" principle. Evidence Fusion & Debate is
**not** a planner choice; it's a fixed step the orchestrator runs after whichever
of the three actually ran. This is enforced at the schema level: `TaskPlan.agents_to_call`
is typed as `PlannerDispatchableAgent`, which only contains those three values, and
a validator rejects any output where a flagged category's owning agent got left out.

Because the planner now reasons over pixels, the backbone changed from a text-only
SLM to a small **vision-language model** (Qwen2-VL-7B-Instruct or Phi-3.5-vision-instruct
— see `configs/adapter_a.yaml`). The Retrieval Agent still runs text-only calls
against the same shared backbone.

## Folder structure

```
agent-a-planner-retrieval/
├── README.md
├── requirements.txt              # split: always-needed vs GPU-only vs vector-store-only
├── configs/
│   └── adapter_a.yaml            # single source of truth for model + LoRA + training config
├── src/agent_a/
│   ├── backbone/
│   │   └── slm_client.py         # SLMClient interface + RealSLMClient (lazy torch/peft imports)
│   ├── adapters/
│   │   └── lora_config.py        # AdapterAConfig dataclass -> peft.LoraConfig
│   ├── schemas/
│   │   └── task_plan.py          # pydantic contracts: TaskPlan, QueryPlan, RetrievalResult...
│   ├── tools/
│   │   └── vector_store.py       # VectorStoreClient interface + FaissVectorStoreClient
│   ├── agents/
│   │   ├── planner_agent.py      # Agent 1
│   │   └── retrieval_agent.py    # Agent 4
│   └── prompts/
│       ├── planner_system_prompt.txt
│       └── retrieval_system_prompt.txt
├── training/
│   ├── prepare_dataset.py            # small hand-written Retrieval-only SFT examples
│   ├── prepare_planner_dataset.py    # image-grounded Planner dataset via distillation — see below
│   ├── PLANNER_TRAINING_DESIGN.md    # why + how: source dataset, labeling strategy, evaluation
│   ├── train_lora_adapter_a.py       # GPU-only LoRA fine-tuning script (text-only; VLM version TBD)
│   └── data/
├── tests/                        # runs with zero GPU/ML deps — see below
└── scripts/
    └── run_local_demo.py         # wires Planner -> Retrieval with a fake client, no GPU needed
```

**Why it's shaped this way:**
- `schemas/` has no dependency on `agents/` or `backbone/` — both import *from* it, never the
  reverse. This is what lets `tests/test_schemas.py` run instantly with only pydantic installed.
- `agents/` depend on the `SLMClient` and `VectorStoreClient` **interfaces**, never on
  `RealSLMClient`/`FaissVectorStoreClient` directly (dependency injection). That's what makes
  `PlannerAgent` and `RetrievalAgent` unit-testable without a GPU, torch, transformers, peft,
  or faiss installed — see `tests/conftest.py`'s `FakeSLMClient`.
- Heavy imports (`torch`, `transformers`, `peft`, `faiss`) are all deferred to inside methods
  (`_ensure_loaded`), not at module level. You can `import` every file in this repo on a laptop
  with just pydantic installed; you only need the real ML stack when you actually call
  `generate_structured()` for real or run the training script.

## GPU vs. CPU — where to do what

**LoRA fine-tuning (`training/train_lora_adapter_a.py`): GPU only.** LoRA reduces *trainable
parameters*, not the size of the forward/backward pass — you're still running full inference
through a 3.8B–8B parameter model every training step. On CPU, a single epoch that takes
roughly 10–20 minutes on a T4/RTX-class GPU (with 4-bit QLoRA, rank 16) would realistically
take 15–30+ hours on CPU. That's not viable in a 4-session hackathon, so the training script
**hard-fails** with a clear error if `torch.cuda.is_available()` is False — don't remove that
guard, it's there to stop someone accidentally burning hours on the wrong machine.

Run it over SSH on the shared GPU box set up in Session 1:
```bash
ssh <user>@<gpu-host>
cd ~/hackathon/agent-a-planner-retrieval
pip install -r requirements.txt   # uncomment the GPU-only block first
python training/prepare_dataset.py
python training/train_lora_adapter_a.py --base_model microsoft/Phi-4-mini-instruct
```

**Inference at demo time: GPU strongly preferred, CPU only as an emergency fallback.**
With 4-bit quantization, a small backbone like Phi-4-mini can technically run inference on
CPU, but expect 5–15 seconds per structured call versus well under a second on GPU. Keep
the demo on the GPU box; if it goes down mid-demo, a pre-warmed CPU fallback (model already
loaded, not loading from cold) is acceptable as insurance, not as the primary path.

**Everything else (unit tests, schema validation, agent logic, the local demo script):
CPU, your laptop, no GPU needed at all.** That's the whole point of the fake-client pattern —
you should be able to develop and test 90% of this repo without ever touching the GPU box.

## Running things

```bash
# One-time setup
pip install -r requirements.txt   # base install; uncomment GPU/vector-store blocks as needed

# Unit tests (fast, no GPU)
PYTHONPATH=src python -m pytest tests/ -v

# Local demo, no GPU (fake client)
PYTHONPATH=src python scripts/run_local_demo.py

# Build the Retrieval-only training set (hand-written seed examples)
python training/prepare_dataset.py

# Build the image-grounded Planner training set (distilled from deep classifiers —
# see training/PLANNER_TRAINING_DESIGN.md; runs with randomized scores until the
# real classifier hooks are wired in)
python training/prepare_planner_dataset.py

# Fine-tune Adapter A — GPU box only
python training/train_lora_adapter_a.py
```

## Wiring in the real backbone (once Adapter A is trained)

```python
from agent_a.backbone.slm_client import RealSLMClient
from agent_a.agents.planner_agent import PlannerAgent
from agent_a.agents.retrieval_agent import RetrievalAgent
from agent_a.tools.vector_store import FaissVectorStoreClient

llm_client = RealSLMClient(
    base_model_name="microsoft/Phi-4-mini-instruct",
    adapter_a_path="checkpoints/adapter_a",
)
planner = PlannerAgent(llm_client=llm_client)
retrieval = RetrievalAgent(llm_client=llm_client, vector_store=FaissVectorStoreClient(...))
```

Both agents share the **same** `RealSLMClient` instance — that's the point of Adapter A being
shared between them; you load the backbone + adapter once, not twice.

## Next: orchestration

This repo deliberately stops at the agent level. LangGraph orchestration (wiring Planner's
`TaskPlan` into actual parallel/sequential dispatch of all 8 agents, plus the backend API
layer) is the next piece — say the word when you're ready and we'll design the LangGraph
graph on top of these two agents.
