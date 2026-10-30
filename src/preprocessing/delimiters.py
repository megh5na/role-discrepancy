from __future__ import annotations

import re
import unicodedata

# --- Pattern groups -----------------------------------------------------

# OpenAI / HuggingFace chat-template special tokens.
_OPENAI_HF_TOKENS = re.compile(
    r"<\|(?:im_start|im_end|endoftext|user|assistant|system|tool|"
    r"start_header_id|end_header_id|eot_id|begin_of_text|"
    r"start_of_role|end_of_role|channel|constrain|message)\|>",
    re.IGNORECASE,
)

# Llama-style instruction wrappers.
_LLAMA_TOKENS = re.compile(
    r"\[/?INST\]|<<SYS>>|<</SYS>>|<<\s*/?\s*SYS\s*>>", re.IGNORECASE
)

# BOS/EOS-ish bare tags.
_BOS_EOS_TAGS = re.compile(r"</?s>", re.IGNORECASE)

# Alpaca / "### Role:" markdown-header style role markers, at line start.
_MARKDOWN_ROLE_HEADER = re.compile(
    r"^\s*#{1,4}\s*(Instruction|Response|Input|System|Context|Output)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Raw transcript role markers ("Human:", "Assistant:", "System:", "User:",
# "Tool:") at the start of a line, optionally bold-markdown-wrapped.
_TRANSCRIPT_ROLE_MARKER = re.compile(
    r"^\s*\**\s*(Human|User|Assistant|System|Tool|AI|Bot)\s*\**\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Ye et al.-style XML role wrapper tags (open and close), used in their
# activation-probe construction and reproduced by our sidecar (C7). Must be
# stripped from anything that reaches the text-only encoder.
_XML_ROLE_TAGS = re.compile(
    r"</?(?:system|user|document|tool|reasoning|assistant)>", re.IGNORECASE
)

# Generic ChatML-adjacent angle-bracket role tags with attributes, e.g.
# <|role:tool|> or <role=tool>.
_GENERIC_ROLE_TAG = re.compile(
    r"<\|?\s*role\s*[:=]\s*\w+\s*\|?>", re.IGNORECASE
)

_ALL_STRIP_PATTERNS = [
    _OPENAI_HF_TOKENS,
    _LLAMA_TOKENS,
    _BOS_EOS_TAGS,
    _MARKDOWN_ROLE_HEADER,
    _TRANSCRIPT_ROLE_MARKER,
    _XML_ROLE_TAGS,
    _GENERIC_ROLE_TAG,
]

# Patterns used only for DETECTION (validator), not stripping: things that
# should never appear in clean text at all, used to fail loudly rather than
# silently strip-and-hope. Currently the same set; kept as a separate name
# so detection logic can diverge from stripping logic later without
# confusion about which list is authoritative for which purpose.
_ALL_DETECT_PATTERNS = _ALL_STRIP_PATTERNS


def strip_delimiters(text: str) -> str:
    """Remove known chat-template / role-marker artifacts from `text`.

    Idempotent-ish: applies each pattern repeatedly is unnecessary since
    patterns don't nest across groups, but we do run a second pass to catch
    a header marker exposed by removing surrounding whitespace-only
    artifacts (e.g. "### System:\\n<|im_start|>" -> after first pass yields
    a leading blank line before "user"-role residue).
    """
    cleaned = unicodedata.normalize("NFKC", text)
    for _ in range(2):
        for pattern in _ALL_STRIP_PATTERNS:
            cleaned = pattern.sub("", cleaned)
    # Collapse whitespace left behind by removed markers.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def contains_delimiter_artifact(text: str) -> bool:
    """True if `text` still contains a recognised delimiter/role-marker
    artifact. Used by:
      - tests/test_delimiters.py (must be False on every cleaned sample)
      - the training loop (asserted on every batch, per Section 27 rule 10 /
        the "assert on every batch" instruction in C2's spec entry)
    """
    for pattern in _ALL_DETECT_PATTERNS:
        if pattern.search(text):
            return True
    return False


def assert_clean_batch(texts: list[str]) -> None:
    """Raise loudly if any text in a training batch still carries a
    delimiter artifact. Cheap (regex over already-short spans); called every
    batch specifically because a silent leak here invalidates results
    without any visible symptom (Section 8, C2: "SECURITY-CRITICAL").
    """
    for t in texts:
        if contains_delimiter_artifact(t):
            raise ValueError(
                "Delimiter artifact survived preprocessing into a training "
                f"batch. This must never happen. Offending text: {t!r}"
            )
