# Role-Discrepancy Detection

**Recovering perceived role from text, for prompt injection defense that needs no access to the model.**

This is the code behind our capstone project at VIT Chennai. The research framing and rules come from `role-discrepancy-v1.pdf`, which is not in this repository. This README covers the implementation only.

## The question

An LLM application builds one prompt out of channels that deserve different levels of trust: system instructions, user turns, retrieved documents and tool outputs. Earlier work found that models decide which role a piece of text has mostly from how it sounds, not from the tags around it, and it showed this by probing model activations. That needs access a hosted-API deployment does not have.

We ask a narrower question. Can the same quantity, the **perceived role**, be estimated from the text alone? And if so, does a gap between it and the channel's **declared** role work as a prompt injection signal, without ever training on attack data?

## Where our contribution lives

```
src/supervision/            cross-channel transplant supervision
src/models/perceived_role/  the encoder (stock DeBERTa-v3-base) trained on it
src/scoring/                discrepancy score and channel-pair severity (default: continuous expected severity)
```

Everything else (`src/ingestion`, `src/preprocessing`, `src/baselines`, `src/evaluation`, `experiments/`, `service/`, `visualization/`) exists to measure whether those three parts work. If the demo and service were deleted, none of the report's figures would change.

## What we claim

1. Transplant supervision makes a text-only role encoder ignore carrier formatting. Swap consistency goes from 0.70 to 0.96 across three seeds.
2. A channel-pair severity score keeps false alarms manageable on realistic benign text.

We do not claim that transplant makes a better detector. A naively trained ensemble detects BIPIA attacks as well as the transplant one, and how we compare with third-party guardrails depends on the benchmark.

## Status

All planned experiments (E1 to E7, plus the follow-ups E5b and E7b) have been run and written up. The white-box probe from the prior work does not reproduce at 135M, 360M or 1.7B parameters. A register probe trained without tags (E5b) is no better than our encoder at that scale, and a feedback-driven attacker (E7b) found no rewrite that both evades the detector and still works.

Figures and tables for the report are generated into `paper/` by `visualization/make_report_assets.py`, and the local demo is in `service/`.

## Demo

Activate the virtual environment first (see Setup below). The system Python has no `transformers` and the server will fail with `ModuleNotFoundError`.

```bash
python -m service.server        # then open http://127.0.0.1:8765
```

You paste a text, pick the channel it supposedly came from, and the page shows whether it is flagged and why. Only one detector (three seeds) is kept in memory at a time, about 3.4 GB, so switching between them takes 10 to 16 seconds. The first score after start-up takes about 15 seconds while the models load.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A few things worth knowing:
- `transformers` is pinned to `4.46.3`, not the latest, because version 5.16.1 has a bug in its DeBERTa-v3 tokenizer conversion.
- Everything was developed on an 8 GB, CPU-only Apple Silicon laptop. Training on the Apple GPU (MPS) thrashed memory, so all runs use the CPU. Run one heavy job at a time. Training a single DeBERTa-v3-base checkpoint takes about 45 to 57 minutes there.
- Only E7 needs the network once the first downloads are done, plus an `ANTHROPIC_API_KEY` in a gitignored `.env` file. Never commit that file. After the first run the scripts set `HF_HUB_OFFLINE=1`, so the models and datasets must already be in the Hugging Face cache.
- `checkpoints/` is gitignored (4.1 GB) and is not in the repository. Back it up, or retrain it with steps 2 and 3 below.

## Reproducing the results

Run every command from the repository root with the virtual environment active. Result files are written next to each script and are read by the analysis scripts and by `visualization/make_report_assets.py`. Steps 1 to 3 are prerequisites for everything else.

**1. Data (C1, C2, C3)**
```bash
python scripts/build_dataset.py                       # ingest register corpora + BIPIA, NotInject, LLMail; clean, segment, dedup
python -m src.supervision.build_supervision_sets \
  --in data/interim/clean_spans_register.jsonl --out-dir data/processed --transplant-rate 0.5
python -m src.ingestion.benign_role_mixed             # E6 benign role-mixed corpus (1,710 texts)
python -m experiments.exp04_generalization.build_e4_sets   # E4 LLMail short set and BIPIA family map
```

**2. Seed-42 checkpoints** (about 1 hour each)
```bash
python -m experiments.exp01_swap_validity.train_production_checkpoint   # transplant, checkpoints/perceived_role_v1
python -m experiments.exp03_baselines.train_naive_checkpoint            # naive,      checkpoints/perceived_role_v1_naive
```

**3. Seed-study checkpoints** (seeds 123 and 456, both conditions; about 3.5 hours)
```bash
python -m experiments.exp03_baselines.run_exp03_seeds --keep
mkdir -p checkpoints/kept_seed_models
mv checkpoints/perceived_role_v1_seed123 checkpoints/perceived_role_v1_seed456 \
   checkpoints/perceived_role_v1_naive_seed123 checkpoints/perceived_role_v1_naive_seed456 checkpoints/kept_seed_models/
```
E4, E6, the E1 re-run and the demo read the six checkpoints from `checkpoints/` (seed 42) and `checkpoints/kept_seed_models/` (seeds 123 and 456).

**4. Experiments**
```bash
# E1 swap validity (gate)
python -m experiments.exp01_swap_validity.run_exp01                    # original run: throwaway models, 2 seeds
python -m experiments.exp01_swap_validity.run_exp01_kept_seeds         # re-run on the six kept checkpoints, no training

# E2 / E2b leave-one-corpus-out (config: configs/experiment/exp02_lodo*.yaml)
python -m experiments.exp02_supervision_compare.run_exp02
python -m experiments.exp02_supervision_compare.run_exp02b_carrier
python -m experiments.exp02_supervision_compare.analyze_b_carrier

# E3 detection against guardrails
python -m experiments.exp03_baselines.run_exp03                        # first cut (seed 42)
python -m experiments.exp03_baselines.run_exp03_naive
python -m experiments.exp03_baselines.analyze_exp03_naive_dedup
python -m experiments.exp03_baselines.run_exp03_wrapped
python -m experiments.exp03_baselines.analyze_exp03_wrapped
python -m experiments.exp03_baselines.analyze_exp03_seeds
python -m experiments.exp03_baselines.analyze_exp03_seeds_perm
python -m experiments.exp03_baselines.analyze_exp03_ensemble

# Severity ablation (the seeds script retrains and saves role probabilities: about 3.5 hours)
python -m experiments.ablation_severity.run_severity_ablation
python -m experiments.ablation_severity.analyze_severity_ablation
python -m experiments.ablation_severity.analyze_severity_cross
python -m experiments.ablation_severity.run_severity_seeds
python -m experiments.ablation_severity.analyze_severity_seeds

# E4 generalization and the pre-registered fusion test
python -m experiments.exp04_generalization.run_e4_scoring
python -m experiments.exp04_generalization.analyze_e4
python -m experiments.exp04_generalization.analyze_fusion

# E5 white-box sidecar (negative result). 135M, then 360M and 1.7B gates; the 1.7B run took hours on CPU.
python -m src.models.sidecar.validate
python -m src.models.sidecar.validate_native_template
python -m src.models.sidecar.layer_sweep
python -m experiments.exp05_oracle_gap.run_sidecar_gate HuggingFaceTB/SmolLM2-360M-Instruct 360m
python -m experiments.exp05_oracle_gap.run_sidecar_gate HuggingFaceTB/SmolLM2-1.7B-Instruct 1p7b bf16

# E6 over-defense
python -m experiments.exp06_overdefense.run_e6_scoring
python -m experiments.exp06_overdefense.run_e6_guardrails
python -m experiments.exp06_overdefense.analyze_e6
python -m experiments.exp06_overdefense.analyze_e6_pooled
python -m experiments.exp06_overdefense.analyze_e6_guardrails
python -m experiments.exp06_overdefense.analyze_continuous_severity

# E7 adaptive attack (paid API calls, Claude only; call cap is set in configs/experiment/exp07_adaptive.yaml)
python -m experiments.exp07_adaptive.run_multi_target                  # v1 (kept as a flawed first attempt; also builds the destyled variants v2 reuses)
python -m experiments.exp07_adaptive.run_multi_target_v2 --host-seed 7 # the reported run, 24 distinct host documents
python -m experiments.exp07_adaptive.analyze_e7_multi
```

The E7 scripts run the three Claude targets listed in the config by default, which matches the reported run.

**5. Report assets and demo**
```bash
python -m visualization.make_report_assets            # figures, tables, captions, provenance manifest in paper/
python -m service.calibrate_thresholds                # demo thresholds from the saved role probabilities
python -m service.build_demo_cases                    # demo cases, seeded random selection
python -m service.server                              # then open http://127.0.0.1:8765
```

`make_report_assets` builds every figure and table from the saved result files, with no hand-typed numbers, in about 20 seconds. It writes 11 figures to `paper/figures/` (PNG and vector PDF, with `CAPTIONS.md` giving each figure's main caveat) and 18 tables to `paper/tables/` (CSV, Markdown and LaTeX). `paper/ASSETS_MANIFEST.md` maps every asset to the result files it came from by sha256 prefix. It also reads `data/interim/`, so on a fresh checkout rebuild the data first (step 1).

## Tests

```bash
pytest tests/
```

There are 134 tests. They check the software (leakage controls, scoring arithmetic, harness logic, the demo core), not the science. No test trains the encoder or asserts an accuracy; the experiments do that.
