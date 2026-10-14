"""Evaluation harness for the MVP criterion: >=80% of 20 predefined questions answered correctly.

Each question lists `any_of` groups; an answer passes when it contains, case-insensitively, at
least one term from EVERY group (AND of ORs), and none of `forbid`. Deliberately simple and
deterministic: it checks facts, not style. Run after `puppy-mind seed`.
"""
from __future__ import annotations

import json
from pathlib import Path


def passes(answer: str, q: dict) -> bool:
    a = answer.lower()
    ok = all(any(t.lower() in a for t in group) for group in q["all_of"])
    return ok and not any(t.lower() in a for t in q.get("forbid", []))


def run(brain, path: str, target: float = 0.8) -> int:
    qs = json.loads(Path(path).read_text(encoding="utf-8"))
    good = 0
    for i, q in enumerate(qs, 1):
        brain.agent.history.pop(brain.session, None)  # questions are independent: no carry-over from earlier answers
        ans = brain.ask(q["q"])["answer"]
        ok = passes(ans, q)
        good += ok
        print(f"{'PASS' if ok else 'FAIL'} {i:2d}. {q['q']}\n      -> {ans[:200]!r}")
    rate = good / len(qs)
    print(f"\n{good}/{len(qs)} = {rate:.0%} (target {target:.0%})")
    return 0 if rate >= target else 1
