"""
src/supervision/balance.py

*** OUR CONTRIBUTION (NEW) *** — Section 6 NEW-1, Section 8 C3(b), Section 17.

Enforces (and then VERIFIES) the three balance conditions the spec requires
of the supervision construction:

    (i)   every label appears in every position (native + every foreign
          carrier channel)
    (ii)  every label draws from multiple source corpora, where a second
          real corpus exists (Section 10; the two
          registers -- tool, system -- currently single-sourced)
    (iii) span length is balanced across labels

Why "enforce, then verify" and not just "enforce": Section 10's own
cross-cutting risk list warns that the corpus-identity shortcut can "enter
through our own pipeline" -- i.e. balancing code can have bugs, or can
balance the wrong axis, and LOOK correct while still leaving a shortcut
available. `corpus_prediction_probe` below is the verification instrument
named explicitly in that section: "verify it with a corpus-prediction
probe."
"""

from __future__ import annotations

import random
from collections import defaultdict

from src.ingestion.schema import Record, Role
from src.supervision.types import TrainingExample


# --- (i) count + position balancing ----------------------------------------


def balance_by_count(
    spans_by_label: dict[Role, list[Record]], seed: int = 42
) -> dict[Role, list[Record]]:
    """Downsample every label to the size of the smallest label class.

    This is the blunt instrument for condition (iii) as well as a
    precondition for (i): if one label has 10x the examples of another, no
    amount of position-balancing fixes the resulting class imbalance in
    training. We do NOT upsample the minority class (duplicating spans would
    itself become a shortcut -- a model could memorise the small set of
    repeated tool spans instead of generalising).
    """
    min_n = min(len(v) for v in spans_by_label.values())
    out = {}
    for label, spans in spans_by_label.items():
        rng = random.Random(seed)
        shuffled = spans[:]
        rng.shuffle(shuffled)
        out[label] = shuffled[:min_n]
    return out


def length_bucket(text: str, n_buckets: int = 3) -> int:
    """Coarse length bucket (by whitespace-token count) used to balance
    length across labels rather than just counts -- a label with all-short
    spans and another with all-long spans would let the model use length as
    a proxy even after count-balancing.
    """
    n = len(text.split())
    if n <= 10:
        return 0
    if n <= 25:
        return 1
    return n_buckets - 1


def balance_by_length(
    spans_by_label: dict[Role, list[Record]], n_buckets: int = 3, seed: int = 42
) -> dict[Role, list[Record]]:
    """Within each label, resample so the length-bucket distribution matches
    the distribution pooled across ALL labels (condition iii). Uses the
    pooled distribution as the common target so no single label's natural
    length profile is privileged.
    """
    rng = random.Random(seed)

    pooled_bucket_counts: dict[int, int] = defaultdict(int)
    for spans in spans_by_label.values():
        for s in spans:
            pooled_bucket_counts[length_bucket(s.text, n_buckets)] += 1
    total_pooled = sum(pooled_bucket_counts.values())
    target_frac = {b: c / total_pooled for b, c in pooled_bucket_counts.items()}

    out: dict[Role, list[Record]] = {}
    for label, spans in spans_by_label.items():
        by_bucket: dict[int, list[Record]] = defaultdict(list)
        for s in spans:
            by_bucket[length_bucket(s.text, n_buckets)].append(s)
        n_target = len(spans)
        selected: list[Record] = []
        for b, frac in target_frac.items():
            pool = by_bucket.get(b, [])
            rng.shuffle(pool)
            take = min(len(pool), round(n_target * frac))
            selected.extend(pool[:take])
        rng.shuffle(selected)
        out[label] = selected
    return out


def ensure_position_coverage(
    examples: list[TrainingExample], min_per_position: int = 5
) -> dict[Role, dict[Role, int]]:
    """Diagnostic (not a filter): counts how many examples of each label
    appear at each position_channel. Returns the coverage table so callers
    can check condition (i) -- "every label appears in every position" --
    was actually satisfied by the construction, not just assumed.
    """
    coverage: dict[Role, dict[Role, int]] = {
        label: {pos: 0 for pos in Role} for label in Role
    }
    for ex in examples:
        coverage[ex.label][ex.position_channel] += 1
    return coverage


# --- (ii) verification: corpus-prediction probe -----------------------------


def corpus_prediction_probe(
    spans_by_label: dict[Role, list[Record]], seed: int = 42
) -> dict[str, float]:
    """For every label with >= 2 source corpora, train a cheap TF-IDF +
    logistic-regression classifier to predict SOURCE CORPUS from TEXT ALONE
    (never the label, which is circular by construction -- see the long
    note in this module's callers). High accuracy here means a given
    label's corpora are stylistically very separable, i.e. there is a lot
    of corpus-specific surface signal available for a model to (mis)use
    INSTEAD of general register, even though it would still get the right
    label either way. This is the same diagnostic Fomin (arXiv 2602.14161)
    ran to show datasets are trivially separable (96.6% source-prediction
    accuracy) -- we run it on our OWN register-source corpora as a
    pre-training sanity check, exactly as Section 10 instructs.

    Reported, not gated: the spec sets no pass/fail threshold, and "no
    experiment may be designed to make results look good" (Section 14) --
    this is a diagnostic to report honestly, not a metric to optimise.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    results: dict[str, float] = {}
    for label, spans in spans_by_label.items():
        by_corpus: dict[str, list[str]] = defaultdict(list)
        for s in spans:
            by_corpus[s.source_dataset].append(s.text)
        if len(by_corpus) < 2:
            continue  # single-sourced label -- probe is undefined, skip
        texts, corpora = [], []
        for corpus, corpus_texts in by_corpus.items():
            texts.extend(corpus_texts)
            corpora.extend([corpus] * len(corpus_texts))

        X_train, X_test, y_train, y_test = train_test_split(
            texts, corpora, test_size=0.25, random_state=seed, stratify=corpora
        )
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, y_train)
        acc = clf.score(Xte, y_test)
        results[label.value] = acc
    return results
