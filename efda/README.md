# Evidence Fusion & Debate Agent (Agent 5)

Part of the PS#09 deepfake investigation pipeline. This agent doesn't
analyze images directly — it takes the opinions of the upstream analysis
agents (Forensic Analysis, Semantic & Context, Retrieval & Comparison),
cross-examines them, resolves disagreement, and produces one trustworthy
intermediate verdict for the Decision & Confidence Agent.

## What it does

1. **Collects** evidence claims from upstream agents (`schemas.py`)
2. **Cross-examines** each claim against the others (`critic.py`) — a rule-based
   critic checks whether a claim's evidence is specific and strong enough to
   survive being challenged by opposing claims
3. **Fuses** surviving claims into one verdict, weighted by each agent's
   track record (`fusion.py`, `reliability_store.py`)
4. **Scores confidence and uncertainty separately** (`uncertainty.py`) — "92%
   fake, low uncertainty" and "92% fake, high uncertainty" are never
   collapsed into the same number
5. **Self-reflects** when agents disagree too much (`self_reflection.py`) —
   classifies *why* they disagree, and either requests re-analysis or flags
   the case for human review instead of forcing a guess
6. **Learns over time** — reliability scores update from feedback on
   resolved cases (`agent.py: record_feedback()`), so agents that turn out
   to be wrong more often get trusted less on similar future cases

## Running it

```bash
# Demo — two example scenarios (confident agreement, and genuine conflict)
python -m efda.demo

# Test suite — 31 tests, no installs required beyond Python 3.10+
python -m unittest discover -s efda/tests -v
```

## Structure

```
efda/
├── schemas.py           # shared data contracts (EvidenceClaim, FinalOutput, etc.)
├── reliability_store.py # per-agent trust scores, persisted to JSON
├── critic.py            # cross-examination / debate step
├── fusion.py            # reliability-weighted verdict combination
├── uncertainty.py       # confidence vs. uncertainty scoring
├── self_reflection.py   # disagreement classification + dissent reports
├── agent.py             # orchestrator wiring all of the above together
├── demo.py              # runnable example
└── tests/
    └── test_efda.py     # 31 unit tests covering the above
```

## Known limitations / next steps

- **Not yet wired to real upstream agents.** Tested only against
  hand-written `EvidenceClaim` data in `demo.py` and the test suite —
  Forensic Analysis, Semantic & Context, and Retrieval & Comparison agents
  don't exist yet elsewhere in the project.
- **Critic is rule-based, not LLM-based.** `critic.py` includes a scaffolded
  `LLMCritic` class ready to accept a real model call (Claude, GPT, etc.)
  for richer natural-language debate reasoning, but it's currently unused —
  the working `RuleBasedCritic` is what's actually wired into the agent.
- **Fusion/uncertainty thresholds are initial estimates.** The `0.15`
  fusion cutoff and `0.45` uncertainty cutoff haven't been calibrated
  against real labeled deepfake cases yet, since no real cases exist to
  calibrate against until the upstream agents are built.
