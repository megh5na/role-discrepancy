from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp06_overdefense.analyze_e6 import SEEDS, load
from src.ingestion.schema import Role
from src.scoring.asymmetry import expected_severity
from src.scoring.calibration import calibrate_threshold

HERE = Path(__file__).resolve().parent
FPR = 0.05


def score_rows(probs: np.ndarray, declared) -> np.ndarray:
    declared = np.asarray(declared)
    out = np.empty(len(probs))
    for ch in set(declared.tolist()):
        m = declared == ch
        out[m] = expected_severity(probs[m], Role(ch))
    return out


def main():
    meta, decl, control, cells, raw = load()
    out = {"rule": "one threshold per ensemble: 5% of pooled control cells flagged, default scorer (expected severity), mean over seeds 42/123/456",
           "target_fpr": FPR, "n_control_texts": int(control.sum()), "thresholds": {}, "behaviour_at_threshold": {}}
    for cond in ("naive", "transplant"):
        members = [f"{cond}_s{s}" for s in SEEDS]
        ens = lambda key, d: np.mean([score_rows(raw[m][key], d) for m in members], axis=0)  # noqa: E731
        corpus = ens("corpus", decl)
        thr = float(calibrate_threshold(corpus[control], FPR))
        n_ni, n_bb, n_bi = len(raw[members[0]]["notinject"]), len(raw[members[0]]["bipia_ben"]), len(raw[members[0]]["bipia_inj"])
        ni = ens("notinject", ["user"] * n_ni)
        bb = ens("bipia_ben", ["document"] * n_bb)
        bi = ens("bipia_inj", ["document"] * n_bi)
        mixed = ~control
        # per-channel thresholds: 5% of THAT channel's own control cell flagged (E6 secondary analysis)
        per_ch = {}
        for ch in ("user", "document", "tool", "reasoning", "system"):
            sel = control & (np.asarray(decl) == ch)
            per_ch[ch] = float(calibrate_threshold(corpus[sel], FPR))
        doc_thr = per_ch["document"]
        out["thresholds_per_channel"] = {**out.get("thresholds_per_channel", {}), cond: per_ch}
        out["behaviour_at_document_channel_threshold"] = {**out.get("behaviour_at_document_channel_threshold", {}), cond: {
            "threshold": round(doc_thr, 4),
            "document_control_flagged": round(float((corpus[control & (np.asarray(decl) == "document")] >= doc_thr).mean()), 3),
            "bipia_benign_flagged": round(float((bb >= doc_thr).mean()), 3),
            "bipia_injected_flagged (120 distinct strings, frequency-weighted)": round(float((bi >= doc_thr).mean()), 3)}}
        print(cond, "document-channel threshold", json.dumps(out["behaviour_at_document_channel_threshold"][cond]))
        out["thresholds"][cond] = thr
        out["behaviour_at_threshold"][cond] = {
            "control_flagged": round(float((corpus[control] >= thr).mean()), 3),
            "mixed_cells_flagged (benign, role-mixed)": round(float((corpus[mixed] >= thr).mean()), 3),
            "notinject_flagged (benign, trigger words)": round(float((ni >= thr).mean()), 3),
            "bipia_benign_flagged": round(float((bb >= thr).mean()), 3),
            "bipia_injected_flagged (120 distinct strings, frequency-weighted)": round(float((bi >= thr).mean()), 3)}
        print(cond, round(thr, 4), json.dumps(out["behaviour_at_threshold"][cond], indent=1))
    (HERE / "thresholds.json").write_text(json.dumps(out, indent=2))
    print("wrote", HERE / "thresholds.json")


if __name__ == "__main__":
    main()
