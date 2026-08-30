"""
src/models/sidecar/wrapper_construction.py

REPRODUCTION of Ye et al. 2026's probe-training construction (Section 4.2:
"identical neutral text from pretraining corpora... wrapped in each of five
role tags, so content is constant across classes and the probe cannot learn
semantics"). Architecture component: C7.

Uses literal XML-style role tags (<system>...</system>, <user>...</user>,
etc.) -- the SAME tag vocabulary src/preprocessing/delimiters.py strips
(_XML_ROLE_TAGS). This is not a coincidence: it is the empirical
demonstration of Section 5's argument for why this exact construction
CANNOT be ported to text space. Run `evaporates_under_stripping()` below to
see it happen -- five wrapped versions of the SAME neutral text, stripped of
tags, collapse to five byte-identical strings.
"""

from __future__ import annotations

from src.ingestion.schema import Role

_WRAPPERS: dict[Role, str] = {
    Role.SYSTEM: "<system>{text}</system>",
    Role.USER: "<user>{text}</user>",
    Role.DOCUMENT: "<document>{text}</document>",
    Role.TOOL: "<tool>{text}</tool>",
    Role.REASONING: "<reasoning>{text}</reasoning>",
}


def wrap(text: str, role: Role) -> str:
    return _WRAPPERS[role].format(text=text)


def build_wrapper_probe_set(neutral_texts: list[str]) -> tuple[list[str], list[Role]]:
    """Ye et al.'s construction: EVERY neutral text wrapped in EVERY role
    tag. Content is identical across classes -- the resulting probe can only
    be picking up the TAG, never semantics, by construction."""
    wrapped_texts: list[str] = []
    labels: list[Role] = []
    for text in neutral_texts:
        for role in Role:
            wrapped_texts.append(wrap(text, role))
            labels.append(role)
    return wrapped_texts, labels


def evaporates_under_stripping(text: str = "This is a neutral sentence.") -> dict[str, str]:
    """Demonstrates Section 5's argument directly: strip the SAME delimiter
    stripper (src/preprocessing/delimiters.py) used everywhere else in this
    project from all five wrapped versions of one neutral text, and show
    they become byte-identical -- i.e. carry five contradictory labels for
    the exact same string. This is WHY the wrapper construction cannot
    supervise a text-only encoder, demonstrated empirically rather than
    just asserted."""
    from src.preprocessing.delimiters import strip_delimiters

    results = {}
    for role in Role:
        wrapped = wrap(text, role)
        results[role.value] = strip_delimiters(wrapped)
    return results
