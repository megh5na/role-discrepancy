"""
src/baselines/simple/tfidf_baseline.py

EXISTING models (TF-IDF, logistic regression), NEW harness applying them to
OUR role-classification task. Architecture component: C8(a) (Section 8/13).

*** THIS IS THE DEEP-LEARNING FALSIFICATION TEST, NOT DECORATION ***
(Section 11: "Component C8(a) exists to falsify our justification. If
TF-IDF + logistic regression matches the encoder on the swap test AND on
LODO, then register is a surface-feature problem, the transformer is
unjustified, and WE REPORT THAT.") Trained on the EXACT SAME TrainingExample
sets (naive_control / transplant_train) as the encoder, evaluated with the
exact same metrics (src/evaluation/metrics.py), so the comparison is
apples-to-apples.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ingestion.schema import ROLE_ORDER, Role
from src.supervision.types import TrainingExample


def train_tfidf_lr(examples: list[TrainingExample], seed: int = 42) -> Pipeline:
    texts = [ex.text for ex in examples]
    labels = [ex.label.value for ex in examples]
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


def make_predict_fn(pipe: Pipeline):
    def predict_fn(texts: list[str]) -> list[Role]:
        preds = pipe.predict(texts)
        return [Role(p) for p in preds]

    return predict_fn
