"""tests/test_sidecar_wrapper.py -- offline tests for C7's wrapper
construction, including the empirical demonstration of Section 5's core
argument (no model needed for this part)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.schema import Role  # noqa: E402
from src.models.sidecar.wrapper_construction import (  # noqa: E402
    build_wrapper_probe_set,
    evaporates_under_stripping,
    wrap,
)


def test_wrap_covers_all_roles():
    text = "hello"
    for role in Role:
        wrapped = wrap(text, role)
        assert text in wrapped
        assert role.value in wrapped


def test_build_wrapper_probe_set_is_content_constant_across_labels():
    neutral = ["Sentence one.", "Sentence two."]
    texts, labels = build_wrapper_probe_set(neutral)
    assert len(texts) == len(neutral) * len(list(Role))
    assert len(set(labels)) == len(list(Role))
    # Content is identical across every label for a given base sentence.
    for base in neutral:
        variants = [t for t in texts if base in t]
        assert len(variants) == len(list(Role))


def test_evaporates_under_stripping_produces_byte_identical_strings():
    """*** THE CORE EMPIRICAL ARGUMENT OF SECTION 5, DEMONSTRATED *** --
    five wrapped versions of the same neutral text, once run through the
    SAME delimiter stripper used everywhere else in the project, must
    collapse to byte-identical strings. If this test ever fails, either the
    stripper regressed (stopped removing these tags) or the wrapper tags
    changed to something the stripper doesn't recognise -- either way, the
    argument this file exists to demonstrate would be silently broken."""
    results = evaporates_under_stripping("This is a neutral sentence.")
    unique_strings = set(results.values())
    assert len(unique_strings) == 1, (
        f"Expected all 5 wrapped-then-stripped versions to be byte-identical, "
        f"got {len(unique_strings)} distinct strings: {results}"
    )
