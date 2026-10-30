import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.benign_role_mixed import (  # noqa: E402
    MANIFEST_PATH,
    OUT_PATH,
    apply_frame,
    build,
    candidate_pools,
    load_config,
    training_used_ids,
)
from src.ingestion.io_utils import read_records  # noqa: E402
from src.ingestion.schema import DataCategory, Record, Role  # noqa: E402
from src.supervision import carriers  # noqa: E402


def _span(i, source, role, words=10):
    return Record(record_id=f"{source}-{i}", text=" ".join(f"w{i}x{j}" for j in range(words)),
                  declared_role=role, source_dataset=source, category=DataCategory.REGISTER_SOURCE)


def _cfg(n=4):
    return {
        "seed": 1, "min_words": 5, "max_words": 40, "default_n_per_cell": n,
        "sources_by_register": {"document": ["docs"], "user": ["asks"]},
        "cells": [
            {"name": "user_pastes_document", "declared": "user", "content": "document",
             "frames": ["A: {span}", "{span} B?"]},
            {"name": "document_quotes_request", "declared": "document", "content": "user",
             "frames": ["Q: {span}"]},
            {"name": "control_user", "declared": "user", "content": "user", "control": True},
        ],
    }


def _spans():
    return ([_span(i, "docs", Role.DOCUMENT) for i in range(40)]
            + [_span(i, "asks", Role.USER) for i in range(40)])


def test_apply_frame_substitutes_literally_and_handles_braces():
    assert apply_frame("hello", "{\"k\": \"{span}\"}") == '{"k": "hello"}'
    assert apply_frame("say \"hi\"", "{\"k\": \"{span}\"}", quote_swap=True) == "{\"k\": \"say 'hi'\"}"
    assert apply_frame("plain", None) == "plain"


def test_build_is_deterministic_and_independent_of_input_order():
    spans = _spans()
    a, _ = build(spans, _cfg(), set())
    b, _ = build(list(reversed(spans)), _cfg(), set())
    assert [r.text for r in a] == [r.text for r in b]


def test_no_span_reused_across_cells_and_texts_unique():
    recs, _ = build(_spans(), _cfg(), set())
    src_ids = [r.meta["source_record_id"] for r in recs]
    assert len(src_ids) == len(set(src_ids))
    assert len({r.text for r in recs}) == len(recs)


def test_used_spans_are_excluded():
    spans = _spans()
    used = {r.record_id for r in spans if r.source_dataset == "docs"}
    pools = candidate_pools(spans, _cfg(), used)
    assert pools["document"] == [] and len(pools["user"]) > 0


def test_length_filter_applies():
    spans = [_span(0, "docs", Role.DOCUMENT, words=2), _span(1, "docs", Role.DOCUMENT, words=60),
             _span(2, "docs", Role.DOCUMENT, words=10)]
    assert [r.record_id for r in candidate_pools(spans, _cfg(), set())["document"]] == ["docs-2"]


def test_raises_when_a_pool_is_too_small():
    with pytest.raises(ValueError, match="unused candidates"):
        build(_spans()[:6], _cfg(n=4), set())


def test_frames_cycle_evenly_and_controls_are_bare():
    recs, _ = build(_spans(), _cfg(n=4), set())
    mixed = [r for r in recs if r.meta["cell"] == "user_pastes_document"]
    assert sorted(r.meta["frame_id"] for r in mixed) == [0, 0, 1, 1]
    ctrl = [r for r in recs if r.meta["control"]]
    assert all(r.text == r.meta["source_record_id"].replace("asks-", "") or r.meta["frame_id"] is None for r in ctrl)
    assert all(r.meta["frame_id"] is None for r in ctrl)


def test_all_records_are_benign_category_c():
    recs, manifest = build(_spans(), _cfg(), set())
    assert all(r.is_injected is False and r.category == DataCategory.BENIGN_ROLE_MIXED for r in recs)
    assert manifest["n_records"] == len(recs) == 12


# ---------------- config + real-corpus checks ----------------

def test_config_frames_are_not_the_transplant_training_templates():
    cfg = load_config()
    trained = {t for ts in carriers._TEMPLATES_BY_CHANNEL.values() for t in ts}
    frames = {f for c in cfg["cells"] for f in c.get("frames", [])}
    assert frames and not (frames & trained)


def test_every_frame_has_exactly_one_placeholder_and_cells_are_well_formed():
    cfg = load_config()
    for c in cfg["cells"]:
        assert c["declared"] in {r.value for r in Role} and c["content"] in cfg["sources_by_register"]
        if c.get("control"):
            assert c["declared"] == c["content"] and "frames" not in c
        else:
            assert c["declared"] != c["content"] and len(c["frames"]) >= 3
            assert all(f.count("{span}") == 1 for f in c["frames"])


corpus_built = OUT_PATH.exists() and MANIFEST_PATH.exists()


@pytest.mark.skipif(not corpus_built, reason="corpus not built; run python -m src.ingestion.benign_role_mixed")
def test_real_corpus_is_disjoint_from_training_and_swap_data():
    used = training_used_ids()
    recs = list(read_records(OUT_PATH))
    leaked = [r.meta["source_record_id"] for r in recs if r.meta["source_record_id"] in used]
    assert not leaked, f"{len(leaked)} corpus spans were used in training/swap construction"


@pytest.mark.skipif(not corpus_built, reason="corpus not built")
def test_real_corpus_shape_matches_manifest_and_covers_all_channels():
    recs = list(read_records(OUT_PATH))
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert len(recs) == manifest["n_records"]
    assert {r.declared_role for r in recs} == set(Role)
    assert len({r.text for r in recs}) == len(recs)
    assert all(r.is_injected is False for r in recs)
    # every mixed cell really is a mismatch between content register and channel
    assert all(r.meta["content_register"] != r.declared_role.value for r in recs if not r.meta["control"])
