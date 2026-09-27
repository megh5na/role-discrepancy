"""
src/preprocessing/dedup.py

NEW (standard MinHash/LSH technique applied to project-specific requirements)
Section 8, C2. "public injection corpora overlap heavily" (spec) is the
stated motivation, but near-duplicates matter for register sources too: many
short Dolly instructions or gsm8k answer-openings are near-identical
templates ("What is the capital of X?"), and letting dozens of copies of
the same sentence into training silently overweights whatever surface
pattern that one template happens to have.

Uses datasketch MinHash + LSH for approximate near-duplicate detection
(exact-hash dedup would miss paraphrase-level duplicates; full pairwise
comparison is O(n^2) and too slow past a few thousand records).
"""
# run separately per role

from __future__ import annotations

import re

from datasketch import MinHash, MinHashLSH

from src.ingestion.schema import Record

_WORD_RE = re.compile(r"\w+")


def _shingles(text: str, k: int = 3) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def dedup_records(
    records: list[Record], threshold: float = 0.8, num_perm: int = 64
) -> list[Record]:
    """Remove near-duplicates (Jaccard similarity >= threshold on 3-word
    shingles) via MinHash-LSH, keeping the first occurrence of each cluster.

    Deduplication is scoped WITHIN a call -- callers deduplicate within and
    across sources by passing all records they want compared together. The
    pipeline (build_dataset.py) calls this within each declared_role class
    to avoid an O(n^2)-ish LSH index across the whole corpus mixing spans
    that were never going to collide anyway.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Record] = []
    for i, r in enumerate(records):
        mh = _minhash(r.text, num_perm=num_perm)
        key = f"r{i}"
        if lsh.query(mh):
            continue  # near-duplicate of something already kept
        lsh.insert(key, mh)
        kept.append(r)
    return kept
