from __future__ import annotations

import numpy as np

from src.ingestion.schema import ROLE_ORDER, Role
from src.scoring.discrepancy import DEFAULT_FORMULATION, base_discrepancy, perceived_role_argmax

COMMAND_LIKE = {Role.USER, Role.SYSTEM}
CONTENT_CHANNELS = {Role.DOCUMENT, Role.TOOL}


def _build_asymmetric_matrix() -> dict[tuple[Role, Role], float]:
    matrix: dict[tuple[Role, Role], float] = {}
    for perceived in Role:
        for declared in Role:
            if perceived == declared:
                matrix[(perceived, declared)] = 0.0
            elif perceived in COMMAND_LIKE and declared in CONTENT_CHANNELS:
                matrix[(perceived, declared)] = 1.0
            elif perceived == Role.REASONING and declared in CONTENT_CHANNELS:
                matrix[(perceived, declared)] = 0.7
            elif perceived == Role.SYSTEM and declared == Role.USER:
                matrix[(perceived, declared)] = 0.6
            elif perceived == Role.USER and declared == Role.SYSTEM:
                matrix[(perceived, declared)] = 0.3
            elif perceived in CONTENT_CHANNELS and declared in COMMAND_LIKE:
                matrix[(perceived, declared)] = 0.15
            else:
                matrix[(perceived, declared)] = 0.4
    return matrix


ASYMMETRIC_SEVERITY: dict[tuple[Role, Role], float] = _build_asymmetric_matrix()

SYMMETRIC_SEVERITY: dict[tuple[Role, Role], float] = {
    (p, d): (0.0 if p == d else 1.0) for p in Role for d in Role
}


def severity(perceived_role: Role, declared_role: Role, use_asymmetry: bool = True) -> float:
    matrix = ASYMMETRIC_SEVERITY if use_asymmetry else SYMMETRIC_SEVERITY
    return matrix[(perceived_role, declared_role)]


def severity_matrix() -> np.ndarray:
    """M[perceived_index, declared_index] (ROLE_ORDER indexing) from ASYMMETRIC_SEVERITY."""
    M = np.zeros((len(ROLE_ORDER), len(ROLE_ORDER)))
    for i, p in enumerate(ROLE_ORDER):
        for j, d in enumerate(ROLE_ORDER):
            M[i, j] = ASYMMETRIC_SEVERITY[(p, d)]
    return M


_SEVERITY_M = severity_matrix()


def expected_severity(perceived_probs: np.ndarray, declared_role: Role) -> np.ndarray | float:
    """sum_r P(r) * S[r, declared]. Accepts one probability vector (returns a float)
    or a (n, 5) batch (returns shape (n,)). Probabilities are in ROLE_ORDER."""
    out = np.asarray(perceived_probs) @ _SEVERITY_M[:, ROLE_ORDER.index(declared_role)]
    return float(out) if out.ndim == 0 else out


def combined_score(
    perceived_probs: np.ndarray,
    declared_role: Role,
    formulation: str = DEFAULT_FORMULATION,
    use_asymmetry: bool = True,
    severity_mode: str | None = None,
) -> dict:
    """THE final discrepancy score (Section 8 C5 output).

    DEFAULT SCORER (since 2026-10-04, user decision): continuous expected severity,
    score = sum_r P(r) * S[r, declared] (`expected_severity`). `severity_mode=None`
    resolves to "expected" when use_asymmetry=True and to "argmax" when use_asymmetry=False
    (the symmetric ablation switches severity off, which the expected form cannot express).

    "argmax" is the ORIGINAL scorer, kept as a named baseline and used by the historical
    E3 scripts so their saved results reproduce: base mismatch magnitude
    (discrepancy.py) multiplied by the severity of the argmax perceived role.
    Multiplicative, not additive: a channel pair we've judged
    near-benign (severity ~0.15) should stay low-scoring even under a large
    probability deficit, and a high-severity pair should scale up with the
    magnitude of the mismatch rather than saturating to a constant flag.
    In "expected" mode the returned `severity` field is still the ARGMAX cell's
    severity (informational); `score` is the expected severity.

    Returns a dict (not just a float) so callers -- the demo, error
    analysis, Experiment 6's per-channel-pair breakdown -- have the
    perceived-role argmax and the raw base score available without
    recomputing them.
    """
    if severity_mode is None:
        severity_mode = "expected" if use_asymmetry else "argmax"
    if severity_mode not in ("argmax", "expected"):
        raise ValueError(f"severity_mode must be 'argmax' or 'expected', got {severity_mode!r}")
    base = base_discrepancy(perceived_probs, declared_role, formulation=formulation)
    perceived_role = perceived_role_argmax(perceived_probs)
    sev = severity(perceived_role, declared_role, use_asymmetry=use_asymmetry)
    if severity_mode == "expected":
        # continuous expected severity (module docstring); the matrix is the asymmetric one by
        # definition, so combining it with use_asymmetry=False would be a silent contradiction
        if not use_asymmetry:
            raise ValueError("severity_mode='expected' uses the asymmetric matrix; use_asymmetry=False is not supported with it")
        score = expected_severity(perceived_probs, declared_role)
    else:
        score = base * sev
    return {
        "score": score,
        "base_discrepancy": base,
        "severity": sev,
        "perceived_role": perceived_role,
        "declared_role": declared_role,
    }

# it returns a dictionary containing the final discrepancy score, the base discrepancy score, the severity of the perceived-declared role pair, and both the perceived and declared roles themselves.
# base is a scalar value derived from the perceived probabilities and declared role, while perceived role is the role with the highest probability according to the model's output. The severity is a weight based on how critical it is for the perceived role to match the declared role, which can be asymmetric depending on the roles involved.