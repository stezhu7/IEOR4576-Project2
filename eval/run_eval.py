"""
Usage:
    python eval/run_eval.py [--base-url http://localhost:8080] [--judge]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import requests

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-6"


def check_must_contain_any(answer: str, keywords: list[str]) -> tuple[bool, str]:
    answer_lower = answer.lower()
    for kw in keywords:
        if kw.lower() in answer_lower:
            return True, f"found keyword: '{kw}'"
    return False, f"none of {keywords} found in answer"


def check_must_not_contain_any(answer: str, keywords: list[str]) -> tuple[bool, str]:
    answer_lower = answer.lower()
    for kw in keywords:
        if kw.lower() in answer_lower:
            return False, f"forbidden keyword found: '{kw}'"
    return True, "no forbidden keywords"


def check_min_length(answer: str, min_len: int) -> tuple[bool, str]:
    if len(answer) >= min_len:
        return True, f"length {len(answer)} >= {min_len}"
    return False, f"too short: {len(answer)} < {min_len}"


def check_backstop(backstop: str, expected: str) -> tuple[bool, str]:
    if backstop == expected or backstop.startswith(expected):
        return True, f"backstop '{backstop}' matches '{expected}'"
    return False, f"backstop '{backstop}' != expected '{expected}'"


def run_deterministic(case: dict, answer: str, backstop: str) -> tuple[bool, list[str]]:
    failures = []

    if "must_contain_any" in case:
        ok, msg = check_must_contain_any(answer, case["must_contain_any"])
        if not ok:
            failures.append(f"must_contain_any: {msg}")

    if "must_not_contain_any" in case:
        ok, msg = check_must_not_contain_any(answer, case["must_not_contain_any"])
        if not ok:
            failures.append(f"must_not_contain_any: {msg}")

    if "min_length" in case:
        ok, msg = check_min_length(answer, case["min_length"])
        if not ok:
            failures.append(f"min_length: {msg}")

    if "expected_backstop" in case:
        ok, msg = check_backstop(backstop, case["expected_backstop"])
        if not ok:
            failures.append(f"backstop: {msg}")

    return len(failures) == 0, failures



JUDGE_SYSTEM = """
You are a strict evaluator for a US stock market data analyst AI.
You will be given a user question, the AI's answer, and evaluation criteria.

Evaluate whether the answer meets the rubric criteria.
The answer should be grounded in actual data (specific numbers, tickers, dates) — not generic market knowledge.

Return ONLY a valid JSON object with no markdown fencing:
{
  "pass_": true or false,
  "rationale": "one sentence explaining why",
  "missing": ["list of missing elements if any"]
}
"""


def judge_with_claude(question: str, answer: str, rubric: str) -> dict:
    if not ANTHROPIC_API_KEY:
        return {"pass_": None, "rationale": "ANTHROPIC_API_KEY not set — skipping judge", "missing": []}

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 512,
        "system": JUDGE_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Answer: {answer}\n\n"
                    f"Rubric: {rubric}"
                )
            }
        ],
    }
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip()
        # Strip markdown fences if present
        import re
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
        return json.loads(text)
    except Exception as e:
        return {"pass_": None, "rationale": f"Judge error: {e}", "missing": []}



def run_case(case: dict, base_url: str, use_judge: bool) -> dict:
    start = time.time()
    try:
        resp = requests.post(
            f"{base_url}/chat",
            json={"text": case["input"]},
            timeout=120,
        )
        resp.raise_for_status()
        data     = resp.json()
        answer   = data.get("answer", "")
        backstop = data.get("backstop", "")
    except Exception as e:
        return {
            "id":       case["id"],
            "category": case["category"],
            "input":    case["input"],
            "passed":   False,
            "failures": [f"API error: {e}"],
            "answer":   "",
            "backstop": "",
            "judge":    None,
            "elapsed":  round(time.time() - start, 1),
        }

    det_pass, failures = run_deterministic(case, answer, backstop)

    judge_result = None
    if use_judge and case.get("rubric") and case["category"] == "in_domain":
        judge_result = judge_with_claude(case["input"], answer, case["rubric"])
        if judge_result.get("pass_") is False:
            failures.append(f"judge: {judge_result.get('rationale', '')}")

    overall = det_pass and (judge_result is None or judge_result.get("pass_") is not False)

    return {
        "id":       case["id"],
        "category": case["category"],
        "input":    case["input"],
        "passed":   overall,
        "failures": failures,
        "answer":   answer[:300],
        "backstop": backstop,
        "judge":    judge_result,
        "elapsed":  round(time.time() - start, 1),
    }


def print_results(results: list[dict]) -> int:
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    print("\n" + "=" * 70)
    print("EVAL RESULTS")
    print("=" * 70)

    for cat, cases in sorted(categories.items()):
        cat_pass = sum(1 for c in cases if c["passed"])
        print(f"\n[{cat.upper()}]  {cat_pass}/{len(cases)}")
        for c in cases:
            icon = "✓" if c["passed"] else "✗"
            print(f"  {icon} {c['id']:12s}  ({c['elapsed']}s)  backstop={c['backstop']}")
            if not c["passed"]:
                for f in c["failures"]:
                    print(f"       ↳ {f}")
                print(f"       answer: {c['answer'][:120]!r}")

    print("\n" + "=" * 70)
    pct = passed / total * 100 if total else 0
    print(f"TOTAL: {passed}/{total}  ({pct:.1f}%)")

    # Per-category pass rates
    for cat, cases in sorted(categories.items()):
        cat_pct = sum(1 for c in cases if c["passed"]) / len(cases) * 100
        flag = "✓" if cat_pct == 100 else ("⚠" if cat_pct >= 80 else "✗")
        print(f"  {flag} {cat:30s}  {cat_pct:.0f}%")

    print("=" * 70)

    safety_cases = categories.get("safety", [])
    oos_cases    = categories.get("oos", [])
    safety_pct   = sum(1 for c in safety_cases if c["passed"]) / len(safety_cases) * 100 if safety_cases else 100
    oos_pct      = sum(1 for c in oos_cases    if c["passed"]) / len(oos_cases)    * 100 if oos_cases    else 100

    if safety_pct < 100:
        print(f"⛔ SAFETY FAILURES — must be 100% (current: {safety_pct:.0f}%)")
    if oos_pct < 100:
        print(f"⛔ OOS FAILURES — must be 100% (current: {oos_pct:.0f}%)")
    if pct >= 80:
        print(f"✓ PASS  ({pct:.1f}% >= 80%)")
        return 0
    else:
        print(f"✗ FAIL  ({pct:.1f}% < 80%)")
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--judge", action="store_true", help="Enable Claude MaaJ judge")
    parser.add_argument("--category", help="Filter to a specific category")
    parser.add_argument("--id", help="Run a single test case by id")
    args = parser.parse_args()

    cases = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]

    if not cases:
        print("No cases matched filters.")
        sys.exit(1)

    print(f"Running {len(cases)} cases against {args.base_url}")
    print(f"MaaJ judge: {'enabled (Claude)' if args.judge else 'disabled'}\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i:2d}/{len(cases)}] {case['id']:12s} ", end="", flush=True)
        result = run_case(case, args.base_url, args.judge)
        icon   = "✓" if result["passed"] else "✗"
        print(f"{icon}  ({result['elapsed']}s)")
        results.append(result)

    exit_code = print_results(results)

    # Save results
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()