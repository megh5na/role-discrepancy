"""
src/supervision/carriers.py

*** OUR CONTRIBUTION (NEW) *** — Section 6 NEW-1, Section 17 (src/supervision/
is one of the three directories the spec names as where the actual
contribution lives).

Architecture component: C3, Supervision Construction (Section 8).

WHAT THIS FILE IS FOR AND WHY IT HAS TO EXIST

The encoder (C4) never sees channel metadata, delimiters, or context (Section
27, Absolute Prohibition A). It sees exactly one string. That means the ONLY
way a channel's identity can leave any trace for the model to (mis)use is
through surface FORMATTING that is typical of that channel — JSON braces for
tool output, short imperative/interrogative phrasing for a user turn,
multi-clause connective prose for a document excerpt, first-person
step-by-step framing for a reasoning trace, persona/identity-directive
framing for a system prompt.

In NATURAL data (Section 5's confound argument) that formatting is perfectly
correlated with the text's actual register, because of course a JSON tool
response is also machine-register content, and a Wikipedia excerpt is also
document-register content. A naively-trained classifier can hit high
accuracy by keying on the formatting alone (length, punctuation, braces)
without ever reading mood/person/modality -- the corpus-identity /
shortcut-learning failure this whole project exists to avoid.

`carriers.py` provides the mechanism for BREAKING that correlation: given a
span of text with a KNOWN origin register (from Category B ingestion,
declared_role == true register, per src/ingestion/register_corpora.py), wrap
it in a carrier template belonging to a DIFFERENT channel's typical surface
formatting. The wrapped text now carries channel-typical formatting cues
pointing one way, while its actual content -- mood, person, modality --
still points the other way. `transplant.py` (next file) uses this to label
such examples by ORIGIN register, forcing a model that wants to do well on
both natural AND transplanted examples to actually read register instead of
formatting.

Deterministic and template-based on purpose (not LLM-generated): keeps the
construction reproducible (same seed -> same training set, Section 27 rule
9), fast (no API calls, no cost, works fully offline), and avoids the
"generator fingerprint" risk flagged in Section 10's cross-cutting data
risks (an LLM asked to "make this sound like a document" would imprint its
own generation style, which is itself a new shortcut).

LIMITATION, STATED HONESTLY: hand-written templates are a coarser
approximation of "channel-typical formatting" than what a large, diverse
natural corpus of transplanted-in-the-wild examples would give. This is a
scope decision forced by the compressed timeline (Section 24) and the
absence of any existing corpus of naturally-transplanted role-mismatched
text (which is precisely category C, and is scarce -- Section 10). Recorded
here rather than glossed over.
"""

from __future__ import annotations

import random

from src.ingestion.schema import Role

# Each template is a format string with one {span} slot. Applying a template
# changes the text's LENGTH, PUNCTUATION STRUCTURE, and/or LEXICAL FRAME to
# match what's typical of that channel -- exactly the surface cues a
# shortcut-learning model would key on.

_USER_TEMPLATES = [
    "{span}",  # bare -- user turns are often presented with no wrapper at all
    "Can you help me with this? {span}",
    "{span} Please let me know.",
    "Quick question: {span}",
    "I was wondering: {span}",
]

_DOCUMENT_TEMPLATES = [
    "{span} This is described in further detail in the following section.",
    "According to the record, {span}",
    "{span} The account continues from this point.",
    "As documented elsewhere, {span}",
    "{span} Further context is provided below.",
]

_TOOL_TEMPLATES = [
    '{{"result": "{span}"}}',
    '{{"response": "{span}", "status": "ok"}}',
    '{{"data": {{"value": "{span}"}}}}',
    '{{"output": "{span}"}}',
    '{{"payload": "{span}", "timestamp": "2026-01-01T00:00:00Z"}}',
]

_REASONING_TEMPLATES = [
    "First, {span} Therefore, this follows.",
    "Let me work through this: {span} So that settles it.",
    "Step 1: {span}",
    "Thinking this through, {span} That must be the answer.",
    "Consider the following: {span} This leads to the conclusion above.",
]

_SYSTEM_TEMPLATES = [
    "You must always ensure that {span}",
    "Your role requires the following: {span}",
    "As a core instruction, {span}",
    "Remember at all times: {span}",
    "{span} This is a standing directive.",
]

_TEMPLATES_BY_CHANNEL: dict[Role, list[str]] = {
    Role.USER: _USER_TEMPLATES,
    Role.DOCUMENT: _DOCUMENT_TEMPLATES,
    Role.TOOL: _TOOL_TEMPLATES,
    Role.REASONING: _REASONING_TEMPLATES,
    Role.SYSTEM: _SYSTEM_TEMPLATES,
}


def available_channels() -> list[Role]:
    return list(_TEMPLATES_BY_CHANNEL.keys())


def embed_in_channel(span_text: str, target_channel: Role, rng: random.Random) -> str:
    """Wrap `span_text` in a carrier template typical of `target_channel`'s
    surface formatting. Deterministic given `rng`'s state.

    JSON carriers (TOOL) need the span's own quotes/braces escaped so the
    result is well-formed-looking JSON-ish text; we do a light escape
    (only double quotes) rather than a full JSON encode, because a full
    json.dumps() would also escape newlines etc. in a way that changes the
    span's visible content more than a carrier transplant should.
    """
    templates = _TEMPLATES_BY_CHANNEL[target_channel]
    template = rng.choice(templates)
    text = span_text
    if target_channel == Role.TOOL:
        text = text.replace('"', "'")
    return template.format(span=text)
