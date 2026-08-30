"""
src/scoring/asymmetry.py

*** OUR CONTRIBUTION (NEW) *** — Section 6 NEW-2, Section 8 C5.

WHY ASYMMETRY EXISTS AT ALL (restated from the spec, Section 6): a SYMMETRIC
discrepancy score treats "perceived=DOCUMENT, declared=USER" (a user pasting
a document excerpt into their own message -- extremely common, benign) as
equally severe as "perceived=USER, declared=TOOL" (a command-register
sentence sitting inside a tool response -- exactly the shape of an
injection). Section 6: "a symmetric comparison would over-flag benign cases
(a user pasting document text is not an attack)." Using a symmetric score
would make the detector useless in practice, which is why this file exists
as a SEPARATE, independently-ablatable component from discrepancy.py.

THE SEVERITY MATRIX AND ITS REASONING (explicit, not a black box, so a panel
can audit every cell -- Section 27 rule 15):

  COMMAND-LIKE registers (USER, SYSTEM): both are, structurally, someone
  telling the model what to do -- a task request or a standing directive.
  CONTENT registers (DOCUMENT, TOOL): both are, structurally, information
  the model is meant to read/use, not obey.

  perceived COMMAND-LIKE, declared CONTENT  -> SEVERITY 1.0 (maximum)
      This is the injection shape itself: text that reads as an instruction,
      sitting in a channel the application marked as passive content.

  perceived REASONING, declared CONTENT     -> SEVERITY 0.7
      Matches Ye et al.'s CoT-forgery finding (Section 4.2): reasoning-style
      text is what drives chain-of-thought hijacking. Slightly below the
      command-like case because reasoning register is a step removed from a
      direct imperative.

  perceived SYSTEM, declared USER           -> SEVERITY 0.6
      A user-turn span that reads like a standing system directive --
      plausible privilege-escalation phrasing ("you must always...", "act
      as..."), worth flagging but far more ambiguous than the content-channel
      cases above (users legitimately ask the assistant to role-play).

  perceived USER, declared SYSTEM           -> SEVERITY 0.3
      The reverse direction: unusual, but not the attack direction the
      threat model (Section 5) is built around.

  perceived CONTENT, declared COMMAND-LIKE  -> SEVERITY 0.15 (near-floor)
      Section 6's own worked example: a user pasting document/email text
      into their turn. Common, legitimate, explicitly why symmetric scoring
      fails.

  everything else (e.g. TOOL vs REASONING mismatches, any pair not covered
  above)                                     -> SEVERITY 0.4 (moderate default)
      Genuinely ambiguous cases we have no strong prior about; moderate
      rather than 0 or 1 so they neither vanish nor dominate.

  perceived == declared                      -> SEVERITY 0.0 (no mismatch)

Ablation hook: `SYMMETRIC_SEVERITY` (all off-diagonal cells = 1.0) is
provided so Experiment 6's ablation ("symmetric vs asymmetric scoring",
Section 14) is a one-line swap, not new code.
"""

from __future__ import annotations

import numpy as np

from src.ingestion.schema import Role
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


def combined_score(
    perceived_probs: np.ndarray,
    declared_role: Role,
    formulation: str = DEFAULT_FORMULATION,
    use_asymmetry: bool = True,
) -> dict:
    """THE final discrepancy score (Section 8 C5 output): base mismatch
    magnitude (discrepancy.py) multiplied by channel-pair severity (this
    file). Multiplicative, not additive: a channel pair we've judged
    near-benign (severity ~0.15) should stay low-scoring even under a large
    probability deficit, and a high-severity pair should scale up with the
    magnitude of the mismatch rather than saturating to a constant flag.

    Returns a dict (not just a float) so callers -- the demo, error
    analysis, Experiment 6's per-channel-pair breakdown -- have the
    perceived-role argmax and the raw base score available without
    recomputing them.
    """
    base = base_discrepancy(perceived_probs, declared_role, formulation=formulation)
    perceived_role = perceived_role_argmax(perceived_probs)
    sev = severity(perceived_role, declared_role, use_asymmetry=use_asymmetry)
    return {
        "score": base * sev,
        "base_discrepancy": base,
        "severity": sev,
        "perceived_role": perceived_role,
        "declared_role": declared_role,
    }
