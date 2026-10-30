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
