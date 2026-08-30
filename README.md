# Role-Discrepancy Detection

**Recovering Perceived Role from Text for Attack-Agnostic Prompt Injection
Defense Without Model Access.**

Source of truth for the project's research framing, scope, and rules is
`role-discrepancy-v1.pdf` (not tracked in this repo). This README is an
implementation-side pointer into it, not a replacement.

## The question

LLM applications assemble prompts from channels of differing trust (system
instructions, user turns, retrieved documents, tool outputs). Models assign
conversational roles by *stylistic register* rather than by structural tags
— a finding established by probing model activations, which requires access
a hosted-API deployment doesn't have. This project asks: can the same
quantity (**perceived role**) be estimated from text alone, and does its
divergence from the **declared** channel work as an attack-agnostic
injection signal — without ever training on attack data?

## Where the actual contribution lives

```
src/supervision/        cross-channel transplantation supervision construction
src/models/perceived_role/  encoder (stock DeBERTa-v3-base) trained on it
src/scoring/             discrepancy score + asymmetric channel-pair severity
```

Everything else (`src/ingestion`, `src/preprocessing`, `src/baselines`,
`src/evaluation`, `experiments/`, `service/`, `visualization/`) is
instrumentation that exists to measure whether the above works. Deletion
test (spec Section 3): if the service layer and demo were deleted, every
figure in the report would be unchanged.

## Status

Experiment 1 and a first cut of Experiment 3 (BIPIA) are done. The leave-one-dataset-out evaluation, the over-defense check and the adaptive attack are still to run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `transformers` is pinned to `4.46.3`, not the latest, because of a real bug in 5.16.1's DeBERTa-v3 tokenizer
conversion path.

## Running the pipeline so far

```bash
# C1: ingest register-source corpora (Category B, training data)
python -m src.ingestion.register_corpora

# C1: ingest injection corpora (Category A, test-only)
python -m src.ingestion.bipia
python -m src.ingestion.notinject
python -m src.ingestion.llmail

# C2: clean + segment + dedup
python -m src.preprocessing.build_clean_spans \
  --in data/interim/register_sources.jsonl \
  --out data/interim/clean_spans_register.jsonl

# C3: THE supervision construction (naive control, transplant set, swap pairs)
python -m src.supervision.build_supervision_sets \
  --in data/interim/clean_spans_register.jsonl \
  --out-dir data/processed \
  --transplant-rate 0.5

# E1: the gating experiment
python -m experiments.exp01_swap_validity.run_exp01
```

## Tests

```bash
pytest tests/
```
