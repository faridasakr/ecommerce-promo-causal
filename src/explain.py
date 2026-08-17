"""LLM explanation layer, with a hard constraint on causal overclaiming.

The design decision that matters here:

    The LLM NEVER sees the raw customer data. It only sees results/summary.json
    -- the numbers the causal pipeline actually computed.

That is deliberate. If you hand an LLM the raw table and ask "did the promo
work?", it will happily run a mental difference-in-means and report $23. The
whole point of the project is that $23 is wrong. So the statistical reasoning
stays in Python, where it is testable, and the LLM is restricted to translating
computed results into stakeholder language.

Three guardrails on top:
  1. A system prompt that forbids causal claims not present in the summary, and
     requires the model to refuse questions the analysis cannot answer.
  2. `validate_response` -- flags causal verbs applied to variables we never
     estimated an effect for.
  3. `validate_numeric_fidelity` -- flags dollar figures that do not appear in
     the summary, catching silent rounding and invented numbers.

None of these is a safety guarantee. (2) is regex-based and cannot parse
negation, hedging, or paraphrase; it catches the common failure, not every
failure. The structured-output upgrade described in the README's future work --
having the model emit a Pydantic schema whose `causal_claims` list is validated
field-by-field before the prose is rendered -- is the principled version.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

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
    path = ROOT / "results" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(
            "results/summary.json not found -- run `python src/run_analysis.py` first."
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


def ask(question: str, summary: dict | None = None, model: str = "claude-sonnet-4-5") -> dict:
    """Answer a stakeholder question. Returns {'answer', 'warnings'}.

    Requires ANTHROPIC_API_KEY. Falls back to a template response so the
    pipeline is testable without network access.
    """
    summary = summary or load_summary()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        answer = _offline_fallback(question, summary)
        return {"answer": answer, "warnings": check_answer(answer, summary), "mode": "offline"}

    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=build_prompt(question, summary),
    )
    answer = resp.content[0].text
    return {"answer": answer, "warnings": check_answer(answer, summary), "mode": "api"}


def _offline_fallback(question: str, s: dict) -> str:
    lo, hi = s["headline_ci"]
    return (
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


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Did the free shipping promotion actually work?"
    out = ask(q)
    print(f"Q: {q}\n")
    print(out["answer"])
    if out["warnings"]:
        print("\n--- GUARDRAIL WARNINGS ---")
        for w in out["warnings"]:
            print(f"  ! {w}")
