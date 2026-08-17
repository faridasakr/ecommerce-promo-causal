"""LLM explanation layer, with a hard constraint on causal overclaiming.

The design decision that matters here:

    The LLM NEVER sees the raw customer data, and never sees the planted
    ground truth. It sees only results/stakeholder_summary.json -- the numbers
    the causal pipeline actually computed.

That is deliberate. If you hand an LLM the raw table and ask "did the promo
work?", it will happily run a mental difference-in-means and report $23. The
whole point of the project is that $23 is wrong. So the statistical reasoning
stays in Python, where it is testable, and the LLM is restricted to translating
computed results into stakeholder language.

The ground truth is excluded for a second reason. The scoring figures live in
results/evaluation_summary.json, which this module must not read. A model that
can see the planted effect can quote it, and on real data that number does not
exist -- the assistant would be leaning on knowledge the analysis could never
have. Keeping the answer key out is what makes its output an honest rehearsal
of the real-data case.

Three layers of defence, in the order they run:

  1. SYSTEM PROMPT -- forbids causal claims not present in the summary, requires
     refusal of questions the analysis cannot answer, and forbids upgrading an
     "economically uncertain" segment into a recommendation to target.

  2. SCHEMA VALIDATION -- the model must fill `StakeholderAnswer`, and every
     structured field is compared against the summary BEFORE any prose is
     shown. Numbers must match to the cent; the three segment lists must equal
     the pipeline's verdict groups exactly; assumptions and limitations must be
     selected from the summary rather than invented. If any field fails, the
     prose is never rendered -- `ask()` returns rendered=False with field-level
     errors.

  3. REGEX BACKSTOP -- `validate_response` and `validate_numeric_fidelity` then
     run over the free-text `answer` field, catching overclaims that live in
     prose and so cannot be schema-checked.

Layer 2 exists because layer 3 is not a design. Regexes inspect prose after the
fact and catch only what a pattern anticipates; they cannot parse negation or
hedging, and a fluent wrong answer passes. Moving the load-bearing claims into
structured fields makes them checkable rather than merely inspectable. Layer 3
is kept because prose can still overclaim in ways no schema constrains.

None of this is a safety guarantee. It narrows the failure surface; it does not
close it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """You are a careful analytics assistant reporting the results \
of a causal study on an e-commerce free-shipping promotion.

You will be given a JSON summary of a completed causal analysis. Answer the \
user's question using ONLY the numbers in that summary.

Hard rules:
1. NEVER state or imply a causal effect for any variable the analysis did not \
estimate. The study estimated the effect of the free-shipping promo on revenue. \
It did NOT estimate the effect of channel, region, device, tenure, or engagement \
on anything. Those are CONFOUNDERS that were adjusted for, not treatments.
2. If asked about a relationship the analysis did not estimate, say plainly that \
the study cannot answer it, and explain what would be required to answer it.
3. Always report the headline estimate WITH its confidence interval. Never give \
a point estimate alone.
4. When the naive and adjusted estimates differ, explain the gap as selection \
bias in plain language -- the people who used the promo were already the \
higher-spending customers.
5. State the key assumption (no unmeasured confounding) whenever you give a \
causal number to a decision-maker.
6. Use dollars and percentages from the summary verbatim. Do not recompute, \
extrapolate, or round differently.
7. Segment targeting verdicts are INTERVAL-based and appear verbatim in \
recommendation.verdicts. Report a segment's verdict as written. Never upgrade \
"economically uncertain; recommend a controlled test" into a recommendation to \
target, however positive that segment's point estimate looks -- an interval \
spanning zero means the data cannot resolve the sign, and a controlled test is \
the recommended next step.

Write for a business stakeholder: short paragraphs, no jargon without a gloss."""


# Variables the study did NOT estimate causal effects for. If the model attaches
# a causal verb to one of these, that is an overclaim.
# Includes the *levels* of each categorical, not just the column name -- a model
# is far more likely to write "paid search drove higher spend" than
# "acquisition channel drove higher spend". Caught by tests/test_pipeline.py.
NON_TREATMENT_VARS = [
    "channel",
    "organic",
    "paid search",
    "social",
    "referral",
    "region",
    "northeast",
    "midwest",
    "south",
    "west",
    "device",
    "mobile",
    "desktop",
    "tenure",
    "engagement",
    "email",
    "prior orders",
    "prior spend",
]
CAUSAL_VERBS = [
    r"caused?",
    r"causes",
    r"drives?",
    r"drove",
    r"led to",
    r"resulted in",
    r"increases?",
    r"decreases?",
    r"boosts?",
    r"lifts?",
]


def load_summary() -> dict:
    """Load the stakeholder artefact -- the only results file this layer may read.

    Never point this at evaluation_summary.json. That file holds the planted
    ground truth, and feeding it to the model would defeat the separation this
    module exists to enforce.
    """
    path = ROOT / "results" / "stakeholder_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            "results/stakeholder_summary.json not found -- run "
            "`python src/run_analysis.py` first."
        )
    with open(path) as f:
        return json.load(f)


def validate_response(text: str) -> list[str]:
    """Flag sentences that attach a causal verb to a non-treatment variable.

    Deliberately conservative: it produces warnings for a human to review rather
    than silently rewriting the model's output. A regex cannot truly parse
    causal claims -- this catches the common failure, not every failure, and the
    README says so.
    """
    warnings = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    verb_pat = re.compile("|".join(CAUSAL_VERBS), re.I)

    for sent in sentences:
        if not verb_pat.search(sent):
            continue
        for var in NON_TREATMENT_VARS:
            if re.search(rf"\b{var}\b", sent, re.I):
                warnings.append(
                    f"Possible causal overclaim about '{var}' (not a treatment "
                    f"in this study): {sent.strip()!r}"
                )
                break
    return warnings


def _collect_numbers(obj, out: set[float] | None = None) -> set[float]:
    """Every numeric value anywhere in the summary, as floats.

    Kept as floats rather than formatted strings: string matching let "$6.40"
    pass against a true value of 6.42, because both render as "6.4" at one
    decimal place. Tolerance-based comparison is the correct primitive here.
    """
    out = set() if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(float(obj))
        out.add(abs(float(obj)))
    return out


def validate_numeric_fidelity(text: str, summary: dict) -> list[str]:
    """Flag dollar figures in the response that do not appear in the summary.

    System-prompt rule 6 says to use the summary's figures verbatim. This checks
    it programmatically rather than trusting the model: a quietly rounded $6.40
    in place of $6.42 is the kind of drift nobody notices in review, and it
    silently misstates a number a stakeholder may act on.
    """
    allowed = _collect_numbers(summary)
    warnings = []
    for raw in re.findall(r"\$\s?(\d[\d,]*\.?\d*)", text):
        val = raw.replace(",", "")
        try:
            f = float(val)
        except ValueError:
            continue
        # Match to the cent. A response figure that differs from every summary
        # value by more than half a cent was rounded, recomputed, or invented.
        if not any(abs(f - v) < 0.005 for v in allowed):
            warnings.append(
                f"Figure ${val} does not match any value in the analysis summary "
                f"(possible hallucinated or silently rounded value)."
            )
    return warnings


def check_answer(text: str, summary: dict) -> list[str]:
    """Run every programmatic guardrail over a candidate response."""
    return validate_response(text) + validate_numeric_fidelity(text, summary)


# ---------------------------------------------------------------------------
# Layer 2: structured output.
#
# The regex validators are a backstop, not a design. They inspect prose after
# the fact and can only catch what a pattern anticipates -- they cannot parse
# negation or hedging, and a fluent wrong answer slips through. Making the
# model fill a schema moves the load-bearing claims out of prose entirely, so
# they can be compared field-by-field against the summary before any text is
# shown. Prose is rendered only if every structured field matches.
# ---------------------------------------------------------------------------
class StakeholderAnswer(BaseModel):
    """The shape the model must produce. Every field is checkable."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, description="Prose for the stakeholder.")
    headline_estimate: float
    ci: list[float] = Field(min_length=2, max_length=2)
    segments_to_target: list[str]
    segments_to_withhold: list[str]
    segments_uncertain: list[str]
    assumptions: list[str] = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)


def validate_structured(payload: dict, summary: dict) -> tuple[StakeholderAnswer | None, list[str]]:
    """Parse and check a structured response against the summary.

    Returns (parsed, errors). `parsed` is None when the payload does not even
    fit the schema. Errors are field-level and specific, because "the model was
    wrong" is not actionable but "segments_to_target claimed high spend" is.
    """
    try:
        parsed = StakeholderAnswer.model_validate(payload)
    except ValidationError as e:
        return None, [f"Schema violation: {err['loc']}: {err['msg']}" for err in e.errors()]

    errors: list[str] = []
    rec = summary.get("recommendation", {})

    # Numbers must match to the cent -- see validate_numeric_fidelity on why
    # this compares floats rather than formatted strings.
    if abs(parsed.headline_estimate - summary["headline_estimate"]) >= 0.005:
        errors.append(
            f"headline_estimate {parsed.headline_estimate} does not match the "
            f"analysis value {summary['headline_estimate']}"
        )
    for i, (got, want) in enumerate(zip(parsed.ci, summary["headline_ci"])):
        if abs(got - want) >= 0.005:
            errors.append(f"ci[{i}] {got} does not match the analysis value {want}")

    # Targeting claims must be exactly the pipeline's verdict groups. This is
    # the field that matters most: it is the instruction a stakeholder acts on,
    # and a model that upgrades an uncertain segment produces a confident,
    # fluent, wrong recommendation that no regex would catch.
    for field, key in (
        ("segments_to_target", "segments_evidence_supports_targeting"),
        ("segments_to_withhold", "segments_evidence_says_destroy_contribution"),
        ("segments_uncertain", "segments_economically_uncertain"),
    ):
        got = sorted(getattr(parsed, field))
        want = sorted(rec.get(key, []))
        if got != want:
            errors.append(f"{field} is {got}, but the analysis says {want}")

    # Assumptions and limitations must be selected from the summary, not
    # invented or paraphrased into something weaker.
    for field, allowed in (
        ("assumptions", summary.get("identifying_assumptions", [])),
        ("limitations", summary.get("limitations", [])),
    ):
        for item in getattr(parsed, field):
            if item not in allowed:
                errors.append(
                    f"{field} contains an entry not present in the analysis "
                    f"summary: {item!r}"
                )

    return parsed, errors


def build_prompt(question: str, summary: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                f"Analysis summary:\n```json\n{json.dumps(summary, indent=2)}\n```\n\n"
                f"Stakeholder question: {question}"
            ),
        }
    ]


def render(payload: dict, summary: dict) -> dict:
    """Validate a structured payload, then render prose only if it holds up.

    The order is the point. Schema validation runs BEFORE any text reaches a
    reader, so a response whose structured claims disagree with the analysis is
    never rendered at all -- it is reported as a failure. The regex validators
    then run as a second pass over the free-text field, catching overclaims
    that live in prose and therefore cannot be schema-checked.
    """
    parsed, errors = validate_structured(payload, summary)
    if errors:
        return {
            "answer": None,
            "structured": parsed.model_dump() if parsed else None,
            "errors": errors,
            "warnings": [],
            "rendered": False,
        }

    # Layer 3: the free-text field still gets the regex backstop.
    warnings = check_answer(parsed.answer, summary)
    return {
        "answer": parsed.answer,
        "structured": parsed.model_dump(),
        "errors": [],
        "warnings": warnings,
        "rendered": True,
    }


def ask(question: str, summary: dict | None = None, model: str = "claude-sonnet-4-5") -> dict:
    """Answer a stakeholder question.

    Returns {'answer', 'structured', 'errors', 'warnings', 'rendered', 'mode'}.
    `answer` is None when structured validation failed -- callers must check
    `rendered` rather than assuming prose is present.

    Requires ANTHROPIC_API_KEY. Falls back to a template response so the
    pipeline is testable without network access; the offline path goes through
    the identical validation, so the guardrails are exercised in CI.
    """
    summary = summary or load_summary()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        out = render(_offline_payload(question, summary), summary)
        out["mode"] = "offline"
        return out

    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=build_prompt(question, summary),
        tools=[{
            "name": "stakeholder_answer",
            "description": "Return the answer as structured fields.",
            "input_schema": StakeholderAnswer.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "stakeholder_answer"},
    )
    blocks = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
    if not blocks:
        return {
            "answer": None, "structured": None,
            "errors": ["Model did not return the structured payload."],
            "warnings": [], "rendered": False, "mode": "api",
        }
    out = render(blocks[0].input, summary)
    out["mode"] = "api"
    return out


def _offline_payload(question: str, s: dict) -> dict:
    """Template answer in the same structured shape the model must return.

    Built from the summary rather than hard-coded, so the offline path
    exercises the real validation instead of trivially satisfying it.
    """
    lo, hi = s["headline_ci"]
    rec = s.get("recommendation", {})
    prose = (
        f"The estimated causal effect of the free-shipping promotion is "
        f"${s['headline_estimate']:.2f} of incremental revenue per customer "
        f"(95% CI ${lo:.2f}-${hi:.2f}), using {s['headline_method']}.\n\n"
        f"A naive comparison of promo users vs non-users suggests "
        f"${s['naive_estimate']:.2f} -- roughly {s['overstatement_factor']}x higher. "
        f"That gap is selection bias: the customers who used the promo were "
        f"already the more engaged, higher-spending ones, and would have bought "
        f"more regardless.\n\n"
        f"This estimate assumes no unmeasured confounding -- that the customer "
        f"attributes we adjusted for capture the reasons people chose to use the "
        f"promo. [offline mode: set ANTHROPIC_API_KEY for a full response]"
    )
    return {
        "answer": prose,
        "headline_estimate": s["headline_estimate"],
        "ci": list(s["headline_ci"]),
        "segments_to_target": rec.get("segments_evidence_supports_targeting", []),
        "segments_to_withhold": rec.get("segments_evidence_says_destroy_contribution", []),
        "segments_uncertain": rec.get("segments_economically_uncertain", []),
        "assumptions": list(s.get("identifying_assumptions", [])),
        "limitations": list(s.get("limitations", [])),
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Did the free shipping promotion actually work?"
    out = ask(q)
    print(f"Q: {q}\n")
    if not out["rendered"]:
        print("REFUSED — structured validation failed before rendering:")
        for e in out["errors"]:
            print(f"  - {e}")
        sys.exit(1)
    print(out["answer"])
    for w in out["warnings"]:
        print(f"\n[guardrail] {w}")
    if out["warnings"]:
        print("\n--- GUARDRAIL WARNINGS ---")
        for w in out["warnings"]:
            print(f"  ! {w}")
