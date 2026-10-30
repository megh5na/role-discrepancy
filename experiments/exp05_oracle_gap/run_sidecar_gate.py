from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.models.sidecar import layer_sweep, validate, validate_native_template

HERE = Path(__file__).resolve().parent


def main(model_name: str, tag: str, dtype: str | None = None):
    t0 = time.time()
    if dtype == "bf16":  # the three sidecar modules bind load_sidecar_model at import time, so patch each
        import functools

        import torch

        patched = functools.partial(validate.load_sidecar_model, dtype=torch.bfloat16)
        for mod in (validate, validate_native_template, layer_sweep):
            mod.load_sidecar_model = patched
    print(f"=== sidecar gate: {model_name} ===", flush=True)
    r_syn = validate.run_validation(model_name=model_name)
    r_nat = validate_native_template.run_validation(model_name=model_name)
    sweep = layer_sweep.sweep(model_name=model_name)
    content_layers = [r["layer"] for r in sweep if r["follows_content"]]
    summary = {
        "model": model_name, "synthetic_tags": r_syn, "native_template": r_nat, "layer_sweep": sweep,
        "layers_following_content": content_layers, "n_layers_swept": len(sweep),
        "max_P_reasoning_any_layer": max(r["P_reasoning"] for r in sweep),
        "min_P_user_any_layer": min(r["P_user"] for r in sweep),
        "gate_reproduces": bool(r_syn["follows_content_register"] or r_nat["follows_content_register"] or content_layers),
        "elapsed_sec": round(time.time() - t0),
    }
    (HERE / f"gate_{tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nGATE SUMMARY ({model_name}): reproduces = {summary['gate_reproduces']}; layers following content = "
          f"{content_layers}; max P(reasoning) over layers = {summary['max_P_reasoning_any_layer']:.3f}; "
          f"min P(user) = {summary['min_P_user_any_layer']:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
