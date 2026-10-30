from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

C_NAIVE, C_TRANS, C_GUARD, C_INK, C_FAINT = "#3B6EA8", "#D98332", "#8A8A8A", "#222222", "#BBBBBB"
C_GUARD2 = "#B5B5B5"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "figure.dpi": 100, "savefig.bbox": "tight"})

SOURCES: dict[str, list[str]] = {}
CAPTIONS: dict[str, str] = {}


# ---------------------------------------------------------------- helpers
def load(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text())


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:10]


def track(asset: str, *rels: str):
    SOURCES.setdefault(asset, []).extend(f"{r} ({sha(r)})" if (ROOT / r).is_file() else r for r in rels)


def save_fig(fig, name: str, caption: str, *srcs: str):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    CAPTIONS[name] = caption
    track(f"figures/{name}", *srcs)


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_table(name: str, rows: list[dict], caption: str, *srcs: str):
    cols = list(rows[0].keys())
    cells = [[fmt(r[c]) for c in cols] for r in rows]
    (TAB / f"{name}.csv").write_text("\n".join([",".join(f'"{c}"' for c in cols)] +
                                               [",".join(f'"{x}"' for x in row) for row in cells]) + "\n")
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] + \
         ["| " + " | ".join(row) + " |" for row in cells]
    (TAB / f"{name}.md").write_text(f"**{caption}**\n\n" + "\n".join(md) + "\n")
    esc = lambda s: s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")  # noqa: E731
    tex = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule", " & ".join(esc(c) for c in cols) + " \\\\", "\\midrule"] + \
          [" & ".join(esc(x) for x in row) + " \\\\" for row in cells] + ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(tex) + "\n")
    CAPTIONS[name] = caption
    track(f"tables/{name}", *srcs)


def ci_str(v, ci):
    return f"{v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"


# ---------------------------------------------------------------- FIGURES
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.8, 3.3))
    ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 34)
    boxes = [("Register corpora\n8 public sets,\n5 role labels", False), ("C1 Ingestion\none schema", False),
             ("C2 Cleaning\nstrip delimiters,\nsplit, dedup", False),
             ("C3 Transplant\nsupervision\n(ours)", True), ("C4 Encoder\nDeBERTa-v3-base\n+ linear head (stock)", False),
             ("C5 Scorer (ours)\nbase discrepancy\n\u00d7 severity", True), ("Flag if score\n\u2265 benign-calibrated\nthreshold", False)]
    w, gap, x = 11.4, 2.7, 1.2
    for i, (txt, ours) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, 17), w, 12, boxstyle="round,pad=0.25,rounding_size=0.8", fc="#F4E3CC" if ours else "white",
                                    ec=C_TRANS if ours else "#666", lw=1.4 if ours else 1))
        ax.text(x + w / 2, 23, txt, ha="center", va="center", fontsize=7.0)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.2, 23), xytext=(x + w + 0.2, 23), arrowprops=dict(arrowstyle="->", color="#555"))
        x += w + gap
    ax.text(50, 12.5, "The encoder sees ONE string. The declared role is compared with the perceived role only inside the scorer (C5).",
            ha="center", fontsize=8.2, style="italic", color=C_INK)
    ax.text(50, 7.5, "Evaluation:  E1 swap validity   E2 / E2b leave-one-corpus-out   E3 detection vs guardrails   E4 attack families + 2nd corpus   "
            "E6 over-defense   E7 adaptive attack", ha="center", fontsize=7.6, color="#444")
    ax.text(50, 3.5, "Orange = the project's own components. Everything else is existing infrastructure. Attack data is never used in training.",
            ha="center", fontsize=7.6, color="#444")
    save_fig(fig, "fig01_architecture", "System architecture. The two components in orange are the project's contributions (transplant "
             "supervision construction, discrepancy scorer with channel-pair severity). Caveat: the transplant construction's measured benefit "
             "is carrier-robustness, not better register recognition (Figs. 2, 3, 5); the severity matrix is evaluated in Figs. 4, 7, 8.")


def fig_e1():
    rows = load("experiments/exp01_swap_validity/results.json")
    orig = [r for r in rows if r["model"] == "deberta_v3_base_encoder"]
    tf = {r["condition"]: r for r in rows if r["model"] == "tfidf_lr"}
    kept = load("experiments/exp01_swap_validity/results_kept_seeds.json")["per_model"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for k, (cond, col) in enumerate((("naive", C_NAIVE), ("transplant", C_TRANS))):
        ks = [kept[f"{cond}_s{s}"] for s in (42, 123, 456)]
        ys = [r["swap_consistency"] for r in ks]
        ax[0].scatter(k + np.array([-0.06, 0, 0.06]), ys, color=col, s=34, zorder=3, label="full-data, 3 seeds" if k == 0 else None)
        ax[0].hlines(np.mean(ys), k - 0.22, k + 0.22, color=col, lw=2.2)
        oy = [r["swap_consistency"] for r in orig if r["condition"] == cond]
        ax[0].scatter([k - 0.3] * len(oy), oy, facecolors="none", edgecolors=col, s=30, zorder=3, label="original run, 2 seeds" if k == 0 else None)
        ax[0].scatter([k + 0.3], [tf[cond]["swap_consistency"]], marker="x", color="#444", s=40, zorder=3)
        ax[1].bar(k - 0.18, np.mean([r["native_acc"] for r in ks]), 0.34, color=col, alpha=0.95)
        ax[1].bar(k + 0.18, np.mean([r["foreign_acc"] for r in ks]), 0.34, color=col, alpha=0.45)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["naive", "transplant"]); ax[0].set_ylim(0.5, 1.02); ax[0].set_xlim(-0.55, 1.55)
    ax[0].set_ylabel("swap consistency"); ax[0].set_title("Same text, bare vs foreign carrier")
    ax[0].scatter([], [], marker="x", color="#444", label="TF-IDF + LR"); ax[0].legend(frameon=False, loc="lower right", fontsize=7.5)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["naive", "transplant"]); ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("accuracy (mean of 3 seeds)"); ax[1].set_title("Native (dark) vs foreign-wrapped (light)")
    save_fig(fig, "fig02_e1_swap_validity", "E1 (gate). Left: swap consistency per seed (filled dots = six full-data checkpoints, 3 seeds per condition; hollow = the original "
             "2-seed subsample run; bar = mean of the filled dots; x = TF-IDF+LR). The naive encoder follows the carrier, the transplant encoder mostly does not, "
             "by +0.25 to +0.26 in every seed (3v3 permutation p = 0.05 one-sided, the minimum attainable). Right: accuracy on bare vs foreign-wrapped held-out spans. "
             "Caveats: 600 swap pairs; carriers are the ones the construction itself uses, so this shows robustness to KNOWN carriers only; spans are held out from the "
             "SAME corpora (E2 shows register does not generalise across corpora); TF-IDF moves the same way, so part of the gap reflects the data construction.",
             "experiments/exp01_swap_validity/results.json", "experiments/exp01_swap_validity/results_kept_seeds.json")


def fig_e2():
    rows = [r for r in load("experiments/exp02_supervision_compare/results.json") if r["model"] == "deberta_v3_base_encoder"]
    summ = load("experiments/exp02_supervision_compare/summary_b_carrier.json")
    by = defaultdict(list)
    lab = {}
    for r in rows:
        by[(r["held_out_corpus"], r["condition"])].append(r["accuracy"]); lab[r["held_out_corpus"]] = r["true_label"]
    corpora = list(dict.fromkeys(r["held_out_corpus"] for r in rows))
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, c in enumerate(corpora):
        n, t = np.mean(by[(c, "naive")]), np.mean(by[(c, "transplant")])
        ax[0].plot([n, t], [i, i], color=C_FAINT, lw=1.6, zorder=1)
        ax[0].scatter([n], [i], color=C_NAIVE, s=42, zorder=3); ax[0].scatter([t], [i], color=C_TRANS, s=42, zorder=3)
    ax[0].axvline(0.2, color="#999", ls="--", lw=1); ax[0].text(0.21, 4.5, "chance", fontsize=7.5, color="#777", va="center")
    ax[0].set_yticks(range(len(corpora))); ax[0].set_yticklabels([f"{c}  ({lab[c]})" for c in corpora]); ax[0].invert_yaxis()
    ax[0].set_xlabel("accuracy on the held-out corpus (bare), mean of 2 seeds"); ax[0].set_xlim(0, 1.04)
    ax[0].set_title("E2: leave-one-corpus-out")
    ax[0].scatter([], [], color=C_NAIVE, label="naive"); ax[0].scatter([], [], color=C_TRANS, label="transplant"); ax[0].legend(frameon=False, loc="upper right")
    m = ["bare_accuracy", "wrapped_accuracy", "bare_wrapped_agreement"]
    names = ["bare\naccuracy", "wrapped\naccuracy", "bare/wrapped\nagreement"]
    for k, (cond, col) in enumerate((("naive", C_NAIVE), ("transplant", C_TRANS))):
        v = [summ[f"deberta_v3_base_encoder|{cond}"][x] for x in m]
        ax[1].bar(np.arange(3) + (k - 0.5) * 0.36, v, 0.34, color=col, label=cond)
        for xi, vi in zip(np.arange(3) + (k - 0.5) * 0.36, v):
            ax[1].text(xi, vi + 0.012, f"{vi:.2f}", ha="center", fontsize=7)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names, fontsize=8); ax[1].set_ylim(0, 1); ax[1].set_title("E2b: unseen corpora, mean of 6 folds")
    ax[1].legend(frameon=False, loc="upper left", fontsize=7.5)
    save_fig(fig, "fig03_e2_lodo_and_carrier", "Left (E2): accuracy on a corpus never seen in training; user and document corpora sit at or below "
             "chance (0.20) for both conditions, so neither learns corpus-independent register for those labels. Right (E2b): the same models "
             "evaluated bare and carrier-wrapped; transplant loses almost no accuracy when wrapped and agrees with itself far more often. "
             "Caveats: 2 seeds, 1,000 training examples per condition, accuracy for a single-label fold is recall; E2b evaluates carriers that "
             "transplant trained on (E3-wrapped adds unseen wrappers).",
             "experiments/exp02_supervision_compare/results.json", "experiments/exp02_supervision_compare/summary_b_carrier.json")


def fig_severity_per_model():
    sev = load("experiments/ablation_severity/analysis_severity_seeds.json")["per_model"]
    cont = load("experiments/exp06_overdefense/analysis_continuous_severity.json")["auroc"]
    models = ["naive_s42", "naive_s123", "naive_s456", "transplant_s42", "transplant_s123", "transplant_s456"]
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for i, m in enumerate(models):
        a, n = sev["prob_deficit|asym"][m]["bare_auroc"], sev["prob_deficit|none"][m]["bare_auroc"]
        c = cont[f"{m}|cont_exp|bipia_benign"][0]
        col = C_NAIVE if m.startswith("naive") else C_TRANS
        ax.plot([i, i], [min(a, n, c), max(a, n, c)], color=C_FAINT, lw=1.4, zorder=1)
        ax.scatter([i], [a], color=col, s=48, zorder=3, label="shipped (asym)" if i == 0 else None)
        ax.scatter([i], [n], facecolor="white", edgecolor=col, s=48, lw=1.6, zorder=3, label="no severity" if i == 0 else None)
        ax.scatter([i], [c], marker="D", color=col, s=34, zorder=3, label="cont_exp" if i == 0 else None)
    ax.axhline(0.5, color="#999", ls="--", lw=1)
    ax.set_xticks(range(6)); ax.set_xticklabels([m.replace("_s", "\nseed ") for m in models])
    ax.set_ylabel("AUROC, BIPIA injected vs BIPIA benign"); ax.set_ylim(0.45, 1.0)
    ax.set_title("Detection quality varies with the training seed and with the severity rule")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    save_fig(fig, "fig04_seed_and_severity", "BIPIA AUROC for each of the six trained models under three scoring rules. Within one condition the "
             "AUROC ranges from 0.57 to 0.95 depending on the training seed; the asymmetric severity matrix costs two of the three transplant seeds "
             "0.17 to 0.25 AUROC on this benign pool, and is neutral for the others. Caveat: BIPIA-only benign negatives; the matrix's intended "
             "benefit shows only once benign traffic is role-mixed (Fig. 8). Single-run values, no intervals.",
             "experiments/ablation_severity/analysis_severity_seeds.json", "experiments/exp06_overdefense/analysis_continuous_severity.json")


def fig_wrapped():
    perm = load("experiments/exp03_baselines/analysis_exp03_seeds_perm.json")["per_model"]
    none = load("experiments/ablation_severity/analysis_severity_seeds.json")["per_model"]["prob_deficit|none"]
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for ax, (title, data) in zip(axs, (("shipped scorer (asymmetric severity)", perm), ("no severity", none))):
        for m, d in data.items():
            col = C_NAIVE if m.startswith("naive") else C_TRANS
            ys = [d["bare_auroc"], d["known_wrapped_auroc"], d["novel_wrapped_auroc"]]
            ax.plot(range(3), ys, "-o", color=col, alpha=0.8, ms=4, lw=1.3)
        ax.set_xticks(range(3)); ax.set_xticklabels(["bare", "known\ncarriers", "novel\ncarriers"]); ax.set_title(title); ax.set_xlim(-0.2, 2.2)
    axs[0].set_ylabel("AUROC (BIPIA injected vs benign, 5% FPR)")
    axs[0].plot([], [], color=C_NAIVE, label="naive (3 seeds)"); axs[0].plot([], [], color=C_TRANS, label="transplant (3 seeds)")
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    save_fig(fig, "fig05_wrapped_robustness", "BIPIA attacks wrapped in other channels' formatting (5 known carrier styles, 6 novel wrapper "
             "styles). Each line is one trained model. Naive models lose more AUROC when wrapped than transplant models in all 3x3 pairings "
             "(exact permutation p = 0.05 one-sided, the minimum attainable at 3 v 3); absolute detection quality is not separated. Caveats: 120 "
             "attacks, one corpus; novel wrappers are mild and hand-written; intervals over seeds not claimed.",
             "experiments/exp03_baselines/analysis_exp03_seeds_perm.json", "experiments/ablation_severity/analysis_severity_seeds.json")


def fig_detection_vs_guardrails():
    d = load("experiments/exp06_overdefense/analysis_e6_guardrails.json")["separability"]
    pools = [("bipia_benign", "BIPIA benign only"), ("same_channel", "same-channel benign"), ("all_benign", "all benign pooled")]
    dets = [("ens_naive|asym", "naive ensemble", C_NAIVE), ("ens_transplant|asym", "transplant ensemble", C_TRANS),
            ("protectai_v2", "ProtectAI v2", C_GUARD), ("third_party_deberta_v3_injection", "third-party DeBERTa", C_GUARD2)]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for j, (key, label, col) in enumerate(dets):
        for i, (p, _) in enumerate(pools):
            rec = d[f"{key}|{p}"]
            x = i + (j - 1.5) * 0.19
            ax.bar(x, rec["auroc"], 0.17, color=col, label=label if i == 0 else None, yerr=[[rec["auroc"] - rec["auroc_ci95"][0]],
                                                                                           [rec["auroc_ci95"][1] - rec["auroc"]]],
                   error_kw=dict(lw=1, capsize=2, ecolor="#333"))
    ax.axhline(0.5, color="#999", ls="--", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in pools]); ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUROC, BIPIA injected vs benign pool")
    ax.set_title("Role-discrepancy detector vs trained guardrails, by benign pool"); ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save_fig(fig, "fig06_detection_vs_guardrails", "AUROC separating BIPIA attacks from three benign pools, shipped (asymmetric) scorer, seed ensembles "
             "of 3. The naive ensemble beats both guardrails on every pool; the transplant ensemble does so only on the all-benign pool. Error bars: "
             "95% bootstrap CI over records (seed uncertainty not included). Caveats: BIPIA-style instruction injections in isolated spans only; "
             "LLMail (Fig. 9) reverses the ordering with ProtectAI; the E6 corpus is benign by construction.",
             "experiments/exp06_overdefense/analysis_e6_guardrails.json")


def fig_overdefense_cells():
    e6 = load("experiments/exp06_overdefense/analysis_e6.json")["by_cell"]
    man = load("data/interim/benign_role_mixed_manifest.json")["cells"]
    order = ["user", "document", "tool", "system", "reasoning"]
    cells = sorted(man, key=lambda c: (order.index(c["declared"]), c["control"], c["cell"]))
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    y = np.arange(len(cells))
    a = [e6[c["cell"]]["asym"] for c in cells]; n = [e6[c["cell"]]["none"] for c in cells]
    ax.barh(y - 0.2, n, 0.38, color="#C9C9C9", label="no severity")
    ax.barh(y + 0.2, a, 0.38, color=C_TRANS, label="asymmetric severity (shipped)")
    for yi, c in zip(y, cells):
        if c["control"]:
            ax.text(1.01, yi, "control", fontsize=7, va="center", color="#666")
    ax.set_yticks(y); ax.set_yticklabels([f"{c['declared']} <- {c['content_register']}  |  {c['cell'].replace('_', ' ')}" for c in cells], fontsize=7.2)
    ax.invert_yaxis(); ax.set_xlim(0, 1.08); ax.set_xlabel("fraction of benign texts flagged (mean over 6 models; threshold = 5% of matched controls)")
    ax.set_title("E6: where does over-defense occur?"); ax.legend(frameon=False, loc="lower right", fontsize=8)
    save_fig(fig, "fig07_overdefense_by_cell", "False-positive rate on benign role-mixed text, by cell (declared channel <- natural register of the "
             "content). The asymmetric severity matrix removes most false alarms for declared user/system/reasoning, but benign document text "
             "quoting requests or directives is flagged 93-99% by both scorers, and tool-wrapped text is almost never flagged by either. Caveats: "
             "corpus benign by construction, inherited and noisy register labels, 55-100 texts per cell; mean over six models with no intervals.",
             "experiments/exp06_overdefense/analysis_e6.json", "data/interim/benign_role_mixed_manifest.json")


def fig_scorers_pools():
    d = load("experiments/exp06_overdefense/analysis_continuous_severity.json")["auroc"]
    pools = [("bipia_benign", "BIPIA benign"), ("same_channel", "same-channel"), ("all_benign", "all benign")]
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)
    for ax, (ens, col, title) in zip(axs, (("ens_naive", C_NAIVE, "naive ensemble"), ("ens_transplant", C_TRANS, "transplant ensemble"))):
        for j, (sc, hatch, lab) in enumerate((("asym", None, "asymmetric (shipped)"), ("none", "///", "no severity"), ("cont_exp", "..", "cont_exp"))):
            vals = [d[f"{ens}|{sc}|{p}"][0] for p, _ in pools]
            ax.bar(np.arange(3) + (j - 1) * 0.26, vals, 0.24, color=col, alpha=[1, 0.35, 0.7][j], hatch=hatch, edgecolor="white", label=lab)
            for xi, v in zip(np.arange(3) + (j - 1) * 0.26, vals):
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.5)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in pools]); ax.set_ylim(0.5, 1.0); ax.set_title(title)
    axs[0].set_ylabel("AUROC, BIPIA injected vs benign pool"); fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    save_fig(fig, "fig08_severity_vs_benign_pool", "AUROC of three scoring rules as the benign pool becomes more realistic. Without the severity matrix "
             "the detector is competitive on BIPIA-only negatives but collapses once benign role-mixed traffic and NotInject are included; the "
             "asymmetric and continuous (cont_exp) scorers hold up. cont_exp matches the shipped scorer on the all-benign pool. Caveats: pool "
             "composition is a construction of ours (mostly role-mixed cells); point estimates (intervals for the asymmetric scorer in Fig. 6; paired differences in the source file); choice between asym "
             "and cont_exp was confirmed on LLMail (pre-registered).",
             "experiments/exp06_overdefense/analysis_continuous_severity.json")


def fig_families():
    r = load("experiments/exp04_generalization/analysis_fusion.json")["results"]
    sets = [("BIPIA_code", "BIPIA\ncode"), ("BIPIA_targeted", "BIPIA\ntargeted"), ("BIPIA_task_irrelevant", "BIPIA\ntask-irrelevant"),
            ("BIPIA_task_relevant", "BIPIA\ntask-relevant"), ("LLMail_all", "LLMail\n(short emails)")]
    dets = [("N", "naive ensemble", C_NAIVE), ("T", "transplant ensemble", C_TRANS), ("P", "ProtectAI v2", C_GUARD), ("G", "third-party DeBERTa", C_GUARD2)]
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    for j, (k, lab, col) in enumerate(dets):
        for i, (s, _) in enumerate(sets):
            rec = r[f"all_benign|{s}|{k}"]
            ax.bar(i + (j - 1.5) * 0.2, rec["auroc"], 0.18, color=col, label=lab if i == 0 else None,
                   yerr=[[rec["auroc"] - rec["auroc_ci95"][0]], [rec["auroc_ci95"][1] - rec["auroc"]]], error_kw=dict(lw=0.9, capsize=1.5, ecolor="#333"))
    ax.axhline(0.5, color="#999", ls="--", lw=1); ax.set_xticks(range(len(sets))); ax.set_xticklabels([l for _, l in sets], fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("AUROC vs all benign pooled"); ax.set_title("E4: detection depends on the attack style")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_fig(fig, "fig09_attack_families", "AUROC per BIPIA attack family and on LLMail short emails (cont_exp scorer for our ensembles). Our detectors "
             "are consistent across BIPIA families (transplant weakest on code); ProtectAI is erratic (0.15 on task-irrelevant, below chance) and "
             "wins on LLMail (0.93). Caveats: 25-45 distinct attacks per BIPIA family; the LLMail subset (426 short, English-leaning, level-1-heavy "
             "emails) has no benign emails, so its negatives are not genre-matched, and a TF-IDF model separates it from benign text at ~1.00. "
             "Intervals: bootstrap over records (500 resamples), no seed uncertainty.",
             "experiments/exp04_generalization/analysis_fusion.json")


def fig_fusion_coverage():
    v = load("experiments/exp04_generalization/analysis_fusion.json")["venn"]
    sets = ["BIPIA_code", "BIPIA_targeted", "BIPIA_task_irrelevant", "BIPIA_task_relevant", "BIPIA_all", "LLMail_all"]
    parts = [("ours_only", "ours only", C_TRANS), ("both", "both", "#6B8E6B"), ("guard_only", "ProtectAI only", C_GUARD), ("neither", "neither", "#E3E3E3")]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    left = np.zeros(len(sets))
    for k, lab, col in parts:
        vals = np.array([v[s][k] for s in sets])
        ax.barh(range(len(sets)), vals, left=left, color=col, label=lab, edgecolor="white")
        for i, (l, x) in enumerate(zip(left, vals)):
            if x >= 0.06:
                ax.text(l + x / 2, i, f"{x:.2f}", ha="center", va="center", fontsize=7.5)
        left += vals
    ax.set_yticks(range(len(sets))); ax.set_yticklabels([s.replace("_", " ") for s in sets]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of attacks caught at each detector's own 5% FPR"); ax.set_title("Fusion test: who catches what")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    save_fig(fig, "fig10_fusion_coverage", "Share of attacks flagged by our detector only, ProtectAI only, both, or neither, each at its own 5% "
             "false-positive threshold. The detectors cover different attacks (BIPIA: ours; LLMail: mostly ProtectAI), but the pre-registered "
             "max-rule fusion did NOT beat the stronger single detector on both corpora (tab09). Caveats: LLMail negatives not genre-matched; "
             "guardrail scores are near-binary; no seed uncertainty.",
             "experiments/exp04_generalization/analysis_fusion.json")


def fig_e7():
    d = load("experiments/exp07_adaptive/analysis_e7_multi.json")
    asr = d["v2b_diverse_hosts(24 documents)"]["asr"]
    det = d["detector"]
    levels = ["0.0", "0.5", "1.0"]
    tcol = {"claude-haiku-4-5": "#8E5BA8", "claude-sonnet-4-5": "#4C9A8B", "claude-sonnet-5": "#C2A23A"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw={"wspace": 0.32})
    # left: attack success, pooled over the three Claude targets (bar) with per-target points
    pooled = [sum(asr[t][L]["succeeded"] for t in asr) for L in levels]
    n_pool = [sum(asr[t][L]["n"] for t in asr) for L in levels]
    ax.bar(range(3), [s / n for s, n in zip(pooled, n_pool)], 0.5, color="#B35C5C", alpha=0.85, label="pooled, 3 targets")
    tops = [max([s / n] + [asr[t][L]["succeeded"] / asr[t][L]["n"] for t in asr]) for (s, n), L in zip(zip(pooled, n_pool), levels)]
    for j, t in enumerate(asr):
        ax.scatter(np.arange(3) + (j - 1) * 0.13, [asr[t][L]["succeeded"] / asr[t][L]["n"] for L in levels], s=22, color=tcol[t],
                   edgecolor="white", linewidth=0.6, zorder=3, label=t.replace("claude-", ""))
    for i, (s, n) in enumerate(zip(pooled, n_pool)):
        ax.text(i, tops[i] + 0.022, f"{s}/{n}", ha="center", fontsize=8.5, color="#7A3C3C", fontweight="bold")
    clean = sum(asr[t]["clean"]["succeeded"] for t in asr)
    ax.axhline(0, color=C_INK, linewidth=0.8)
    ax.text(1.5, 0.105, f"control, attack removed:\n{clean}/72 judged successes", ha="center", fontsize=7.5, color="#555")
    ax.set_xticks(range(3)); ax.set_xticklabels(["0.0\n(original)", "0.5", "1.0\n(most destyled)"])
    ax.set_ylim(0, 0.34); ax.set_ylabel("attack success rate"); ax.set_xlabel("destyling level")
    ax.set_title("Does the attack still work?")
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=1)
    # right: detector, fraction of the 24 attacks flagged at the 5%-FPR threshold
    for key, lab, col in (("ens_naive|cont_exp", "naive ensemble", C_NAIVE), ("ens_transplant|cont_exp", "transplant ensemble", C_TRANS)):
        vals = [det[key][L]["detected"] / det[key][L]["n"] for L in levels]
        ax2.plot(range(3), vals, "-o", color=col, label=lab, linewidth=1.8)
        for i, L in enumerate(levels):
            ax2.text(i + 0.05, vals[i] + 0.03, f"{det[key][L]['detected']}/{det[key][L]['n']}", fontsize=8, color=col, ha="left", va="bottom")
    ax2.set_xticks(range(3)); ax2.set_xticklabels(["0.0\n(original)", "0.5", "1.0\n(most destyled)"])
    ax2.set_ylim(0, 1); ax2.set_xlim(-0.25, 2.45); ax2.set_ylabel("fraction of attacks flagged"); ax2.set_xlabel("destyling level")
    ax2.set_title("Does the detector still catch it? (5% FPR)"); ax2.legend(frameon=False, fontsize=8, loc="upper right")
    save_fig(fig, "fig11_e7_adaptive", "E7, multi-target. Left: attack success after the injected instruction is rewritten to sound less like a command "
             "(24 BIPIA attacks, each in its own real host document; bar = pooled over three Claude targets, dots = each target; the clean control, with "
             "the attack removed, gives 0/72 judged successes). Right: share of the same 24 variants our seed-ensemble detectors flag at the threshold "
             "giving 5% false positives on BIPIA benign text (scorer cont_exp). Destyling lowers detection AND removes the attack: only 2 of 144 destyled "
             "trials both evaded the naive ensemble and succeeded. Caveats: 24 attacks and counts of 0-5; this figure's rewriter is single-shot with no "
             "feedback (a four-round feedback attacker, E7b, is in tab14 and also found no attack that both evades and works); targets and judge are all Claude; the detector scores the bare span, not the span inside "
             "its host document; the transplant ensemble is already weak on these attacks at level 0.",
             "experiments/exp07_adaptive/analysis_e7_multi.json")


# ---------------------------------------------------------------- TABLES
def tab_corpora():
    cnt = Counter()
    with open(ROOT / "data/interim/clean_spans_register.jsonl") as f:
        for line in f:
            r = json.loads(line); cnt[(r["declared_role"], r["source_dataset"])] += 1
    rows = [{"role": k[0], "source": k[1], "clean spans": v} for k, v in sorted(cnt.items())]
    rows.append({"role": "TOTAL", "source": "register-source corpora (training pool)", "clean spans": sum(cnt.values())})
    man = load("data/interim/benign_role_mixed_manifest.json")
    rows.append({"role": "eval only", "source": f"benign role-mixed corpus ({man['n_mixed']} mixed + {man['n_control']} control cells)", "clean spans": man["n_records"]})
    fam = load("experiments/exp04_generalization/e4_sets_meta.json")
    rows.append({"role": "eval only", "source": "BIPIA unique injected strings (after delimiter stripping)", "clean spans": sum(fam["bipia_family_counts"].values())})
    rows.append({"role": "eval only", "source": "LLMail-Inject short emails (5-45 words, ASCII >= 0.9)", "clean spans": fam["llmail_n"]})
    write_table("tab01_datasets", rows, "Datasets. Register-source corpora train the encoder; everything labelled 'eval only' is never used in training. "
                "Caveat: BIPIA's 800 injected spans in E3 are only 120 distinct strings.",
                "data/interim/clean_spans_register.jsonl", "data/interim/benign_role_mixed_manifest.json", "experiments/exp04_generalization/e4_sets_meta.json")


def tab_e1_e2():
    rows = []
    for r in load("experiments/exp01_swap_validity/results.json"):
        rows.append({"experiment": "E1", "model": r["model"], "condition": r["condition"], "seed": r["seed"],
                     "swap consistency": r["swap_consistency"], "native acc": r["native_acc"], "foreign acc": r["foreign_acc"]})
    for m, r in load("experiments/exp01_swap_validity/results_kept_seeds.json")["per_model"].items():
        cond, seed = m.split("_s")
        rows.append({"experiment": "E1 re-run (full-data, kept checkpoints)", "model": "deberta_v3_base_encoder", "condition": cond, "seed": int(seed),
                     "swap consistency": r["swap_consistency"], "native acc": r["native_acc"], "foreign acc": r["foreign_acc"]})
    write_table("tab02_e1_swap_validity", rows, "E1 swap validity (the gate). Same held-out spans shown bare and wrapped in a foreign carrier. First rows: original run (2 seeds, "
                "1,275-example subsample, plus TF-IDF+LR); last six rows: re-run on the six full-data kept checkpoints (3 seeds), which passed the pre-registered replication rule. "
                "Caveat: 600 swap pairs from the training corpora; known carriers only.", "experiments/exp01_swap_validity/results.json", "experiments/exp01_swap_validity/results_kept_seeds.json")
    e2 = [r for r in load("experiments/exp02_supervision_compare/results.json") if r["model"] == "deberta_v3_base_encoder"]
    by = defaultdict(list); lab = {}
    for r in e2:
        by[(r["held_out_corpus"], r["condition"])].append(r["accuracy"]); lab[r["held_out_corpus"]] = r["true_label"]
    rows2 = [{"held-out corpus": c, "label": lab[c], "naive (mean of 2 seeds)": float(np.mean(by[(c, 'naive')])),
              "transplant (mean of 2 seeds)": float(np.mean(by[(c, 'transplant')]))} for c in dict.fromkeys(r["held_out_corpus"] for r in e2)]
    rows2.append({"held-out corpus": "MEAN of 6 folds", "label": "", "naive (mean of 2 seeds)": float(np.mean([r["naive (mean of 2 seeds)"] for r in rows2])),
                  "transplant (mean of 2 seeds)": float(np.mean([r["transplant (mean of 2 seeds)"] for r in rows2]))})
    write_table("tab03_e2_lodo", rows2, "E2 leave-one-corpus-out, encoder accuracy on the held-out corpus (bare). Chance is 0.20. Caveat: 2 seeds, 1,000 "
                "examples/condition; single-label folds, so accuracy is recall; system and tool are single-sourced and cannot be held out.",
                "experiments/exp02_supervision_compare/results.json")
    s = load("experiments/exp02_supervision_compare/summary_b_carrier.json"); a = load("experiments/exp02_supervision_compare/analysis_b_carrier.json")
    rows3 = [{"model": k.split("|")[0], "condition": k.split("|")[1], "bare acc": v["bare_accuracy"], "wrapped acc": v["wrapped_accuracy"],
              "bare/wrapped agreement": v["bare_wrapped_agreement"]} for k, v in s.items()]
    rows3.append({"model": "agreement: transplant - naive (encoder)", "condition": f"{a['agreement']['folds_positive']} folds positive",
                  "bare acc": "", "wrapped acc": "", "bare/wrapped agreement": f"{a['agreement']['mean_diff_transplant_minus_naive']:.3f} (sign test p = {a['agreement']['sign_test_p_over_folds']:.3f})"})
    write_table("tab04_e2b_carrier", rows3, "E2b: unseen corpora scored bare and carrier-wrapped (mean over 6 folds, seeds averaged within fold). Caveat: carriers "
                "are templates transplant trained on; 2 seeds; span-level tests would overstate certainty, the fold-level sign test is the conservative one.",
                "experiments/exp02_supervision_compare/summary_b_carrier.json", "experiments/exp02_supervision_compare/analysis_b_carrier.json")


def tab_seeds_scorers():
    cont = load("experiments/exp06_overdefense/analysis_continuous_severity.json")
    sev = load("experiments/ablation_severity/analysis_severity_seeds.json")
    rows = []
    for m in ["naive_s42", "naive_s123", "naive_s456", "transplant_s42", "transplant_s123", "transplant_s456", "ens_naive", "ens_transplant"]:
        rows.append({"model": m, "BIPIA AUROC asym": cont["auroc"][f"{m}|asym|bipia_benign"][0], "none": cont["auroc"][f"{m}|none|bipia_benign"][0],
                     "cont_exp": cont["auroc"][f"{m}|cont_exp|bipia_benign"][0],
                     "all-benign AUROC asym": cont["auroc"][f"{m}|asym|all_benign"][0], "all-benign none": cont["auroc"][f"{m}|none|all_benign"][0],
                     "all-benign cont_exp": cont["auroc"][f"{m}|cont_exp|all_benign"][0]})
    for cond in ("naive", "transplant"):
        sp = {s: cont["seed_spread"][f"bipia_benign|{cond}|{s}"] for s in ("asym", "none", "cont_exp")}
        spa = {s: cont["seed_spread"][f"all_benign|{cond}|{s}"] for s in ("asym", "none", "cont_exp")}
        rows.append({"model": f"{cond}: mean +/- sd over 3 seeds", "BIPIA AUROC asym": f"{sp['asym'][0]:.3f} +/- {sp['asym'][1]:.3f}",
                     "none": f"{sp['none'][0]:.3f} +/- {sp['none'][1]:.3f}", "cont_exp": f"{sp['cont_exp'][0]:.3f} +/- {sp['cont_exp'][1]:.3f}",
                     "all-benign AUROC asym": f"{spa['asym'][0]:.3f} +/- {spa['asym'][1]:.3f}", "all-benign none": f"{spa['none'][0]:.3f} +/- {spa['none'][1]:.3f}",
                     "all-benign cont_exp": f"{spa['cont_exp'][0]:.3f} +/- {spa['cont_exp'][1]:.3f}"})
    write_table("tab05_seeds_and_scorers", rows, "Per-model AUROC under three scorers on two benign pools (BIPIA benign only; all benign pooled). Training seed "
                "dominates detection quality within a condition; removing the severity matrix helps transplant on BIPIA-only negatives and hurts both "
                "on the all-benign pool. Caveat: point estimates; 3 seeds per condition (seed-level tests cannot reach p < 0.05 two-sided).",
                "experiments/exp06_overdefense/analysis_continuous_severity.json", "experiments/ablation_severity/analysis_severity_seeds.json")
    perm = sev["permutation"]["prob_deficit|none"]; permA = sev["permutation"]["prob_deficit|asym"]
    rows = []
    for k in ("bare_auroc", "known_wrapped_auroc", "novel_wrapped_auroc", "auroc_drop_known", "auroc_drop_novel", "fpr_inflation_known", "fpr_inflation_novel"):
        rows.append({"quantity": k, "naive - transplant (asym)": permA[k]["observed_naive_minus_transplant"], "p two-sided (asym)": permA[k]["p_two_sided"],
                     "naive - transplant (no severity)": perm[k]["observed_naive_minus_transplant"], "p two-sided (no severity)": perm[k]["p_two_sided"]})
    write_table("tab05b_seed_level_tests", rows, "Seed-level exact permutation tests, naive vs transplant (3 v 3 trained models; smallest attainable p is 0.10 "
                "two-sided). Caveat: no absolute detection difference is demonstrated; the degradation-when-wrapped effects reach the minimum p.",
                "experiments/ablation_severity/analysis_severity_seeds.json")


def tab_e3():
    dd = load("experiments/exp03_baselines/analysis_exp03_naive_dedup.json")["detectors"]
    rows = []
    for k, lab in (("ours_naive", "role detector, naive, seed 42"), ("ours_transplant", "role detector, transplant, seed 42"),
                   ("protectai_v2", "ProtectAI v2"), ("third_party_deberta_v3_injection", "third-party DeBERTa")):
        v = dd[k]
        rows.append({"detector": lab, "TPR@1% FPR": ci_str(v["tpr_0.01"], v["tpr_0.01_ci95"]), "TPR@5% FPR": ci_str(v["tpr_0.05"], v["tpr_0.05_ci95"]),
                     "AUROC": ci_str(v["auroc"], v["auroc_ci95"])})
    write_table("tab06_e3_deduplicated", rows, "E3 on BIPIA, deduplicated: the 800 injected spans are 120 distinct strings and each distinct string counts once "
                "(bootstrap over strings, threshold re-derived per resample). Caveat: single seed 42 checkpoints (the weakest seeds); see Fig. 4 and tab05 for the "
                "3-seed picture. The frequency-weighted numbers first reported (TPR@5% 0.547) are inflated by duplicate attack strings.",
                "experiments/exp03_baselines/analysis_exp03_naive_dedup.json")


def tab_e6():
    g = load("experiments/exp06_overdefense/analysis_e6_guardrails.json")
    rows = []
    for k, lab in (("protectai_v2", "ProtectAI v2"), ("third_party_deberta_v3_injection", "third-party DeBERTa"),
                   ("ens_naive|asym", "naive ensemble, asym"), ("ens_transplant|asym", "transplant ensemble, asym"),
                   ("ens_naive|none", "naive ensemble, no severity"), ("ens_transplant|none", "transplant ensemble, no severity")):
        v = g["operating_point"][k]
        rows.append({"detector": lab, "control FPR": v["ctrl_fpr"], "mixed-cell FPR": v["mixed_fpr"], "NotInject FPR": v["notinject_fpr"],
                     "BIPIA benign FPR": v["bipia_benign_fpr"], "BIPIA TPR": v["bipia_tpr"]})
    write_table("tab07_e6_operating_point", rows, "E6 over-defense at a deployment-like operating point: one threshold per detector, set so 5% of matched "
                "clean control cells are flagged. Caveats: benign role-mixed corpus is benign by construction with noisy inherited labels; a "
                "control-calibrated threshold does not transfer to BIPIA benign content for our detectors (high 'BIPIA benign FPR'), so this table is not a ranking; "
                "intervals in the source file.", "experiments/exp06_overdefense/analysis_e6_guardrails.json")
    cs = load("experiments/exp06_overdefense/analysis_continuous_severity.json")
    rows = []
    for u in ("ens_naive", "ens_transplant"):
        for s in ("asym", "none", "cont_exp"):
            v = cs["operating_point"][f"{u}|{s}"]
            rows.append({"detector": f"{u}|{s}", "mixed FPR": v["mixed_fpr"], "NotInject FPR": v["notinject_fpr"],
                         "BIPIA benign FPR": v["bipia_benign_fpr"], "BIPIA TPR": v["bipia_tpr"]})
    write_table("tab07b_scorer_operating_point", rows, "Operating point of the three scorers (seed ensembles). cont_exp keeps the over-defense benefit of the "
                "asymmetric matrix and lowers NotInject false alarms. Caveat: same corpus caveats as tab07.",
                "experiments/exp06_overdefense/analysis_continuous_severity.json")


def tab_e4_fusion():
    f = load("experiments/exp04_generalization/analysis_fusion.json"); e4 = load("experiments/exp04_generalization/analysis_e4.json")
    r = f["results"]
    rows = []
    for s in ("BIPIA_all", "BIPIA_code", "BIPIA_targeted", "BIPIA_task_irrelevant", "BIPIA_task_relevant", "LLMail_all", "LLMail_level1", "LLMail_level2", "LLMail_level3"):
        row = {"attack set": s}
        for k, lab in (("N", "naive ens"), ("T", "transplant ens"), ("P", "ProtectAI"), ("G", "3rd-party")):
            v = r[f"all_benign|{s}|{k}"]; row[lab] = ci_str(v["auroc"], v["auroc_ci95"])
        rows.append(row)
    write_table("tab08_e4_families", rows, "E4: AUROC vs all-benign pool by attack set (our ensembles use cont_exp). Caveats: 25-45 distinct attacks per BIPIA family; "
                "LLMail is a supplementary short-email subset with no benign emails (negatives not genre-matched); bootstrap over records only.",
                "experiments/exp04_generalization/analysis_fusion.json")
    rows = [{"attack set vs benign pool": k.replace("|", " vs "), "TF-IDF + LR 5-fold CV AUROC": v} for k, v in e4["tfidf_style_auc"].items()]
    write_table("tab08b_style_diagnostic", rows, "Source-style diagnostic: a bag-of-words classifier trained ON the attacks separates both attack sets from the benign "
                "pools almost perfectly, so these benchmarks offer little protection against lexical shortcuts.", "experiments/exp04_generalization/analysis_e4.json")
    c = e4["preregistered_check"]; fc = f["criterion"]
    rows = [{"pre-registered check": "cont_exp vs asym on LLMail (non-inferior if CI lower bound > -0.03)",
             "result": f"naive ens {c['ensembles']['ens_naive']['diff']:+.3f} {c['ensembles']['ens_naive']['ci95']}, transplant ens "
                       f"{c['ensembles']['ens_transplant']['diff']:+.3f} {c['ensembles']['ens_transplant']['ci95']}; singles ok {c['singles_ok']}/6",
             "verdict": "CONFIRMED" if c["confirmed"] else "NOT confirmed"}]
    for ps in ("BIPIA_all", "LLMail_all"):
        v = fc[ps]
        rows.append({"pre-registered check": f"fusion (N+P, max) beats better single detector on {ps}",
                     "result": f"vs {v['better_component']}: {v['diff']:+.3f} {v['ci95']}", "verdict": "pass" if v["pass"] else "FAIL"})
    rows.append({"pre-registered check": "fusion complementary on both sets", "result": "", "verdict": "YES" if fc["verdict_complementary"] else "NO"})
    write_table("tab09_preregistered_checks", rows, "Pre-registered checks (criteria written to docs/decisions.md before the data were scored). Both are reported "
                "whatever they showed.", "experiments/exp04_generalization/analysis_e4.json", "experiments/exp04_generalization/analysis_fusion.json")


def tab_misc():
    s1, s2 = load("experiments/sidecar_validation/results.json"), load("experiments/sidecar_validation/results_native_template.json")
    g360, g17 = load("experiments/exp05_oracle_gap/gate_360m.json"), load("experiments/exp05_oracle_gap/gate_1p7b.json")
    rows = []
    for size, syn, nat, layers in (("135M", s1, s2, "31 swept, none follow content"),
                                   ("360M", g360["synthetic_tags"], g360["native_template"], f"{g360['n_layers_swept']} swept, none follow content"),
                                   ("1.7B", g17["synthetic_tags"], g17["native_template"], f"{g17['n_layers_swept']} swept, none follow content")):
        rows.append({"sidecar model (SmolLM2)": size, "variant": "synthetic XML tags, 5 roles", "in-distribution accuracy": syn["in_distribution_wrapper_accuracy"],
                     "P(reasoning) in conflict": syn["conflict_mean_reasoningness"], "P(user) in conflict": syn["conflict_mean_userness"],
                     "reproduces?": syn["follows_content_register"], "layer sweep": layers})
        rows.append({"sidecar model (SmolLM2)": size, "variant": "native ChatML template, 3 roles", "in-distribution accuracy": nat["in_distribution_accuracy"],
                     "P(reasoning) in conflict": nat["conflict_mean_reasoningness"], "P(user) in conflict": nat["conflict_mean_userness"],
                     "reproduces?": nat["follows_content_register"], "layer sweep": layers})
    write_table("tab10_sidecar", rows, "Sidecar reproduction of the white-box probe at three scales (SmolLM2-Instruct 135M, 360M, 1.7B; 1.7B in bf16). "
                "Reasoning-register text under a user tag reads as user at every size, the opposite of Ye et al.'s finding. Caveat: all far smaller than their "
                "7B-class models, which do not fit on this machine; one model family; this does not affect E1-E6.",
                "experiments/sidecar_validation/results.json", "experiments/sidecar_validation/results_native_template.json",
                "experiments/exp05_oracle_gap/gate_360m.json", "experiments/exp05_oracle_gap/gate_1p7b.json")


def tab_e7_multi():
    d = load("experiments/exp07_adaptive/analysis_e7_multi.json")
    a = d["v2b_diverse_hosts(24 documents)"]["asr"]
    rows = []
    for tgt, cells in a.items():
        row = {"target": tgt}
        for c in ("clean", "0.0", "0.5", "1.0"):
            row[f"level {c}" if c != "clean" else "clean control"] = f"{cells[c]['succeeded']}/{cells[c]['n']} (refused {cells[c]['refused']})"
        rows.append(row)
    det = d["detector"]
    for name in ("ens_naive|cont_exp", "ens_transplant|cont_exp"):
        rows.append({"target": f"detector: {name} (detected at 5% FPR)", "clean control": "",
                     **{f"level {L}": f"{det[name][L]['detected']}/{det[name][L]['n']}" for L in ("0.0", "0.5", "1.0")}})
    write_table("tab13_e7_multi_target", rows, "E7 multi-target: attack success (succeeded/24 per cell; API refusals in brackets, counted as not succeeded) by destyling level, "
                "with the clean (attack-removed) control as the judge false-positive floor, and our detector's detection counts on the same variants. "
                "Caveats: n = 24 attacks, so counts of 0-5; all three targets and the judge are Claude (OpenAI dropped by decision); single-shot rewriter with no "
                "feedback, so a weak adaptive attacker; host documents are real BIPIA contexts (24 distinct).", "experiments/exp07_adaptive/analysis_e7_multi.json")


def tab_e7b():
    d = load("experiments/exp07_adaptive/analysis_e7_adaptive.json")
    a = load("experiments/exp07_adaptive/adaptive_judge_audit.json")
    cand = [json.loads(l) for l in open(ROOT / "experiments/exp07_adaptive/adaptive_candidates.jsonl") if l.strip()]
    rew = [c for c in cand if c["round"] > 0 and not c["refused"]]
    cell = lambda fl, fo: sum(1 for c in rew if c["flagged"] is fl and c["followed"] is fo)  # noqa: E731
    rows = [
        {"measure": "Joint success, adaptive attacker (evades and works, of 24 attacks)", "value": d["adaptive_joint_success"]},
        {"measure": "Joint success, non-adaptive reference (same detector, threshold and target)", "value": d["reference_joint_success"]},
        {"measure": "Gain (pre-registered rule: >= 4 clear, <= 1 none, else inconclusive)", "value": f"+{d['gain']}: {d['verdict_preregistered'].lower()}"},
        {"measure": "Round at which the one adaptive win occurred (an unmodified original)", "value": "0"},
        {"measure": f"Rewrites (rounds 1 to 4, n = {len(rew)}): evaded and ignored by the target", "value": cell(False, False)},
        {"measure": "Rewrites: flagged and ignored", "value": cell(True, False)},
        {"measure": "Rewrites: flagged but followed", "value": cell(True, True)},
        {"measure": "Rewrites: evaded AND followed", "value": cell(False, True)},
        {"measure": "Evaded-but-not-followed candidates re-judged by a second judge that flipped", "value": f"{len(a['flipped'])} of {a['n_evaded_not_followed']}"},
        {"measure": "Rewriter refusals (candidates)", "value": d["refusals_total"]},
    ]
    write_table("tab14_e7b_feedback_attacker", rows, "E7b: an attacker with feedback from the detector (numeric score) and the target (followed or not), four rounds, against the naive ensemble at its "
                "document-channel threshold. Pre-registered rule applied mechanically. The single win is an unmodified original attack; no rewrite both evaded and worked. Caveats: 24 attacks "
                "and counts of 0 or 1; Claude-only rewriter, target and judge; a numeric-score oracle a real attacker may lack; the provider blocked the rewriter on cipher-style attacks.",
                "experiments/exp07_adaptive/analysis_e7_adaptive.json", "experiments/exp07_adaptive/adaptive_judge_audit.json", "experiments/exp07_adaptive/adaptive_candidates.jsonl")


def tab_e5b():
    rows = []
    for tag, size in (("360m", "360M (primary)"), ("135m", "135M (size check)")):
        d = load(f"experiments/exp05_oracle_gap/e5b_{tag}.json")
        for cond, x in d["conditions"].items():
            ref = d["encoder_reference_3_seed_mean"][cond]
            rows.append({"sidecar model": size, "probe trained on": cond, "layer (by validation acc)": x["selected_layer"], "validation acc": x["val_acc_selected"],
                         "layer-0 validation acc": x["val_acc_by_layer"][0], "probe swap consistency": x["probe_swap_metrics"]["swap_consistency"],
                         "encoder swap consistency (3-seed mean)": round(ref["swap_consistency"], 3), "gap": x["gap_probe_minus_encoder"]["swap_consistency"],
                         "verdict (band +/-0.05)": x["interpretation_swap_consistency"].split(" (")[0]})
    write_table("tab15_e5b_oracle_gap", rows, "E5b: white-box register probe vs the black-box encoder on the 600 swap pairs (pre-registered; not a reproduction of the prior work). The probe is trained on register-labelled "
                "text without role tags; the layer is chosen by validation accuracy only. Gap = probe minus encoder. Caveats: a probe on layer 0 alone (mean token embeddings) already reaches 0.82 to 0.84 validation accuracy, so the "
                "oracle is weak; one small model and one probe seed; lbfgs did not fully converge on unscaled activations; 7B-class models do not fit on this machine.",
                "experiments/exp05_oracle_gap/e5b_360m.json", "experiments/exp05_oracle_gap/e5b_135m.json", "experiments/exp01_swap_validity/results_kept_seeds.json")


def tab_tests():
    out = subprocess.run([str(ROOT / ".venv/bin/python3"), "-m", "pytest", "tests", "-q"], capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"(\d+) passed", out.stdout)
    coll = subprocess.run([str(ROOT / ".venv/bin/python3"), "-m", "pytest", "tests", "--collect-only", "-q"], capture_output=True, text=True, cwd=ROOT)
    per = Counter(line.split("::")[0] for line in coll.stdout.splitlines() if "::" in line)
    cover = {"test_delimiters.py": "delimiter stripping (security-critical)", "test_leakage.py": "no id overlap between train and swap pools",
             "test_schema.py": "record schema and ids", "test_supervision.py": "transplant construction, balance, swap-pair reservation",
             "test_scoring.py": "discrepancy and severity scoring", "test_lodo.py": "leave-one-dataset-out splitting",
             "test_matched_fpr.py": "threshold calibration, matched-FPR comparison", "test_over_defense.py": "FPR by channel pair",
             "test_significance.py": "paired bootstrap, McNemar", "test_guardrail_harness.py": "guardrail harness (offline)",
             "test_sidecar_probe.py": "linear probe readout", "test_sidecar_wrapper.py": "wrapper construction, stripping collapse",
             "test_asr_harness.py": "E7 harness logic with a fake client", "test_exp02b_carrier.py": "E2b evaluation sets and summary",
             "test_exp03_naive.py": "paired detector comparison helpers", "test_exp03_wrapped.py": "wrapped conditions and bootstrap",
             "test_severity_ablation.py": "scoring variants", "test_benign_role_mixed.py": "E6 corpus builder and leakage checks",
             "test_continuous_severity.py": "continuous severity variants"}
    rows = [{"test file": k.replace("tests/", ""), "tests": v, "covers": cover.get(k.replace("tests/", ""), "")} for k, v in sorted(per.items())]
    rows.append({"test file": "TOTAL", "tests": sum(per.values()), "covers": f"{m.group(1) if m else '?'} passed in the latest run"})
    write_table("tab11_tests", rows, "Automated tests. They verify software and data-pipeline correctness (leakage, schema, scoring arithmetic, harness logic). They do "
                "NOT establish that the model learned register, or that any result is scientifically valid; the experiments do that, with their stated limits.",
                "tests/ (collected and run at generation time)")


def tab_ledger():
    cont = load("experiments/exp06_overdefense/analysis_continuous_severity.json")
    e1 = [r for r in load("experiments/exp01_swap_validity/results.json") if r["model"] == "deberta_v3_base_encoder"]
    sc = {c: np.mean([r["swap_consistency"] for r in e1 if r["condition"] == c]) for c in ("naive", "transplant")}
    g = load("experiments/exp06_overdefense/analysis_e6_guardrails.json")["separability"]
    f = load("experiments/exp04_generalization/analysis_fusion.json")
    rows = [
        {"experiment": "E1 swap validity", "question": "Does the encoder read register or carrier formatting?", "setup": "naive vs transplant labels, same spans, held-out swap pairs",
         "key result": f"swap consistency naive {sc['naive']:.3f} vs transplant {sc['transplant']:.3f} (2 seeds); re-run on 6 full-data checkpoints: 0.701 vs 0.957, +0.25 to +0.26 in all 3 seeds (replicated)", "main limitation": "known carriers only; spans from training corpora; TF-IDF moves the same way"},
        {"experiment": "E2 / E2b leave-one-corpus-out", "question": "Does it generalise to an unseen corpus; is transplant robust there?", "setup": "6 folds, 2 seeds, 1000 ex/condition; bare and carrier-wrapped",
         "key result": "bare accuracy tied; wrapped drop 0.010 (transplant) vs 0.102 (naive)", "main limitation": "near chance on user/document; known carriers only"},
        {"experiment": "Seed study + severity ablation", "question": "How much is the training seed and the severity matrix?", "setup": "3 seeds x 2 conditions; asym / none / cont_exp from saved probabilities",
         "key result": "bare AUROC 0.57-0.95 within a condition; matrix costs 2 of 3 transplant seeds 0.17-0.25 on BIPIA-only", "main limitation": "3 seeds; hand-set matrix"},
        {"experiment": "E3 detection vs guardrails", "question": "Does the discrepancy score beat trained guardrails?", "setup": "BIPIA, unique strings, matched FPR, seed ensembles",
         "key result": f"AUROC all-benign: naive ens {g['ens_naive|asym|all_benign']['auroc']:.3f}, transplant ens {g['ens_transplant|asym|all_benign']['auroc']:.3f}, ProtectAI {g['protectai_v2|all_benign']['auroc']:.3f}",
         "main limitation": "BIPIA-style attacks only; 120 distinct attacks"},
        {"experiment": "E4 generalization", "question": "Do results survive other attack families and a second corpus?", "setup": "BIPIA families; LLMail short emails; pre-registered scorer check",
         "key result": "consistent across BIPIA families; ProtectAI better on LLMail (0.93); cont_exp confirmed", "main limitation": "LLMail genre-confounded; small families"},
        {"experiment": "E6 over-defense", "question": "How often are benign role-mixed texts flagged?", "setup": "1,710-text corpus + NotInject; threshold = 5% of matched controls",
         "key result": "mixed-cell FPR 0.18-0.83 (asym); matrix essential on realistic pools; quoted requests in documents flagged 93-99%", "main limitation": "benign by construction; noisy labels"},
        {"experiment": "Fusion (pre-registered)", "question": "Are the role detector and ProtectAI complementary?", "setup": "benign-calibrated percentile fusion, max / mean",
         "key result": "criterion NOT met (fused < better single on both corpora)", "main limitation": "unsupervised fusion only"},
        {"experiment": "E7 adaptive (multi-target)", "question": "Does destyling evade detection while still working?", "setup": "24 BIPIA attacks x 3 levels x 3 Claude targets, host-document embedding, clean control",
         "key result": "score falls with destyling and so does attack success (12/72 -> 2/72 -> 0/72)", "main limitation": "n=24, all-Claude targets and judge; single-shot rewriter here, feedback attacker in E7b"},
        {"experiment": "E7b feedback attacker", "question": "Does a rewriter with detector and target feedback find an attack that evades and works?", "setup": "24 attacks, 4 rounds, numeric-score oracle, naive ensemble document threshold, pre-registered rule",
         "key result": "no meaningful gain: 1 vs 0 of 24 (the win is an unmodified original); 0 of 80 rewrites evaded and worked", "main limitation": "n=24, Claude only, four rounds, rewriter blocked on cipher-style attacks"},
        {"experiment": "Sidecar (C7)", "question": "Does Ye et al.'s probe result reproduce?", "setup": "SmolLM2 135M, 360M, 1.7B; 3 variants incl. all-layer sweep", "key result": "does not reproduce at any layer or size (P(user) 0.990-0.9996)",
         "main limitation": "7B-class (their setting) does not fit; one family"},
        {"experiment": "E5b oracle gap", "question": "Is a white-box register probe better than our black-box encoder?", "setup": "SmolLM2-360M and 135M, probe trained on register text without tags, 600 swap pairs, pre-registered band +/-0.05",
         "key result": "no oracle advantage: gaps +0.017 and -0.037 (360M); same transplant jump as the encoder", "main limitation": "weak oracle (layer 0 alone is close), one small model, one probe seed"},
    ]
    write_table("tab12_experiment_ledger", rows, "Experiment ledger: question, setup, headline result and main limitation for each experiment. All numbers are read from the "
                "result files listed in ASSETS_MANIFEST.md.", "experiments/exp01_swap_validity/results.json", "experiments/exp06_overdefense/analysis_e6_guardrails.json",
                "experiments/exp04_generalization/analysis_fusion.json", "experiments/exp06_overdefense/analysis_continuous_severity.json")


def main():
    for fn in (fig_architecture, fig_e1, fig_e2, fig_severity_per_model, fig_wrapped, fig_detection_vs_guardrails, fig_overdefense_cells,
               fig_scorers_pools, fig_families, fig_fusion_coverage, fig_e7,
               tab_corpora, tab_e1_e2, tab_seeds_scorers, tab_e3, tab_e6, tab_e4_fusion, tab_misc, tab_e7_multi, tab_e7b, tab_e5b, tab_tests, tab_ledger):
        fn()
        print("ok", fn.__name__)
    (FIG / "CAPTIONS.md").write_text("# Figure and table captions (with the main caveat of each)\n\n" +
                                     "\n\n".join(f"**{k}** — {v}" for k, v in sorted(CAPTIONS.items())) + "\n")
    lines = ["# Report assets manifest", "", "Generated by `python -m visualization.make_report_assets` from saved result files only.", ""]
    for k in sorted(SOURCES):
        lines.append(f"- `{k}`  <-  " + "; ".join(f"`{s}`" for s in dict.fromkeys(SOURCES[k])))
    (ROOT / "paper" / "ASSETS_MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"{len(list(FIG.glob('*.png')))} figures, {len(list(TAB.glob('*.csv')))} tables")


if __name__ == "__main__":
    main()
