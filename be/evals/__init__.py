"""Text-to-SQL evaluation harness for the structured-data query agent.

Two layers, deliberately separated (see evals/README.md for the full rationale):

- **Ground truth** (`run_ground_truth`): executes a hand-written golden SQL per
  case through the same read-only SQL tool the agent uses, and checks the answer
  against a value measured from the live database. No API key, deterministic —
  it pins ground truth and doubles as a regression test that the data + curated
  semantics are what we assume.
- **Agent grading** (`run_agent`, opt-in): runs the real structured-query agent
  on the natural-language question and grades its prose answer against the same
  ground truth, producing a per-semantic-category scorecard. This is the signal
  for *which* semantics are worth promoting from curated tool-notes into a
  governed, machine-readable model first — a category the agent fails is a
  semantic the prose hints don't reliably convey.
"""

from __future__ import annotations

from evals.cases import CASES, EvalCase
from evals.harness import (
    AgentResult,
    GroundTruthResult,
    run_agent,
    run_ground_truth,
)

__all__ = [
    "CASES",
    "EvalCase",
    "AgentResult",
    "GroundTruthResult",
    "run_agent",
    "run_ground_truth",
]
