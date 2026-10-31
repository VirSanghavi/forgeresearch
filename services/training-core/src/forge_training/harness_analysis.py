"""Paired evidence, format sensitivity, and actionable DEVELOPMENT-only diagnoses."""

from __future__ import annotations

import math

from .scoring import paired_bootstrap, wilson_interval


def paired_metric(before: list[bool], after: list[bool], seed: int) -> dict:
    if not before or len(before) != len(after):
        raise ValueError("Paired sample counts differ or are empty")
    differences = [int(b) - int(a) for a, b in zip(before, after, strict=True)]
    wins, losses = differences.count(1), differences.count(-1)
    discordant = wins + losses
    # Exact one-sided McNemar/sign test; zero discordant pairs give p=1, not certainty.
    p = sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / 2**discordant
    n = len(before)
    bci, aci = wilson_interval(sum(before), n), wilson_interval(sum(after), n)
    return {
        "num_examples": n,
        "baseline_correct": sum(before),
        "candidate_correct": sum(after),
        "delta_percentage_points": sum(differences) / n * 100,
        "wins": wins,
        "regressions": losses,
        "both_correct": sum(a and b for a, b in zip(before, after, strict=True)),
        "both_wrong": sum(not a and not b for a, b in zip(before, after, strict=True)),
        "baseline_wilson_95": bci,
        "candidate_wilson_95": aci,
        "paired_bootstrap_95_percentage_points": paired_bootstrap(differences, seed),
        "mcnemar_exact_one_sided_p": p,
    }


def compare_harness(baseline: dict, candidate: dict) -> dict:
    for key in ("protocol", "model", "dataset", "runtime", "task_sha256", "role"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Harness comparison mismatch: {key}")
    if baseline["checkpoint_sha256"] == candidate["checkpoint_sha256"]:
        raise ValueError("Candidate must differ from baseline")
    left, right = baseline["samples"], candidate["samples"]
    if len(left) != len(right) or not left:
        raise ValueError("Paired sample counts differ or are empty")
    if len({s["id"] for s in left}) != len(left):
        raise ValueError("Duplicate sample identities")
    for a, b in zip(left, right, strict=True):
        for key in ("id", "question", "target", "prompt_sha256"):
            if a[key] != b[key]:
                raise ValueError(f"Unpaired sample: {key}")
        for row in (a, b):
            if any(type(row[k]) is not bool for k in ("strict", "flexible", "hit_limit")):
                raise ValueError("Scores must be booleans")
    metrics = {
        key: paired_metric(
            [s[key] for s in left], [s[key] for s in right], baseline["protocol"]["seed"]
        )
        for key in ("strict", "flexible")
    }
    primary = metrics["strict"]
    # A predeclared conservative claim gate. Diagnostics never establish final improvement.
    significant = primary["mcnemar_exact_one_sided_p"] < 0.05
    supported = (
        baseline["role"] == "final"
        and len(left) == 1319
        and primary["delta_percentage_points"] > 0
        and primary["paired_bootstrap_95_percentage_points"][0] > 0
        and significant
        and metrics["flexible"]["delta_percentage_points"] >= 0
    )
    format_only = [
        a["id"]
        for a, b in zip(left, right, strict=True)
        if not a["strict"] and a["flexible"] and b["strict"] and b["flexible"]
    ]
    recommendations = []
    if baseline["role"] == "development":
        if any(s["hit_limit"] for s in left + right):
            recommendations.append(
                {
                    "action": "paired_length_ablation",
                    "reason": (
                        "Generation caps may hide correct answers. Re-evaluate BOTH policies at "
                        "a longer fixed cap on development only."
                    ),
                }
            )
        if format_only or any(s["flexible"] and not s["strict"] for s in left):
            recommendations.append(
                {
                    "action": "separate_format_from_accuracy",
                    "reason": (
                        "Track strict and flexible scores together; format-only gains do not "
                        "demonstrate better reasoning."
                    ),
                }
            )
        if primary["regressions"] or metrics["flexible"]["regressions"]:
            recommendations.append(
                {
                    "action": "retention_control",
                    "reason": (
                        "Test a lower learning rate and training-only replay mixture; never "
                        "copy development answers into training."
                    ),
                }
            )
        if len(left) < 100 or not significant:
            recommendations.append(
                {
                    "action": "expand_disjoint_development",
                    "reason": (
                        "Small or inconclusive gains need a larger predeclared development "
                        "partition, disjoint from all training and regression questions."
                    ),
                }
            )
    return {
        "role": baseline["role"],
        "metrics": metrics,
        "strict_gains_already_flexible_correct": format_only,
        "truncation": {
            "baseline": sum(s["hit_limit"] for s in left),
            "candidate": sum(s["hit_limit"] for s in right),
        },
        "regressed_ids": [
            a["id"] for a, b in zip(left, right, strict=True) if a["strict"] and not b["strict"]
        ],
        "improved_ids": [
            a["id"] for a, b in zip(left, right, strict=True) if not a["strict"] and b["strict"]
        ],
        "final_improvement_supported": supported,
        "development_release_gate": {
            "passed": primary["delta_percentage_points"] > 0
            and metrics["flexible"]["delta_percentage_points"] >= 0,
            "rule": (
                "Strict accuracy must improve without a flexible-extraction accuracy "
                "drop; governance still must approve."
            ),
        },
        "recommendations": recommendations,
        "limitations": [
            "Flexible extraction can select an incidental number; it is a secondary diagnostic.",
            "Paired bootstrap intervals can degenerate on small samples; inspect "
            "counts and the exact test.",
            "Development-selected results are exploratory, not confirmatory.",
            "This measures GSM8K only; it does not establish general or safety superiority.",
        ],
    }


def summarize_ablation(entries: list[dict]) -> dict:
    """Ask whether a development gain survives every predeclared prompt/length variant.

    Each entry holds a variant ``name``, its ``protocol`` and the paired ``comparison``.
    All variants must evaluate the same examples, so per-example stability is comparable.
    """
    if len(entries) < 2:
        raise ValueError("An ablation needs at least two variants")
    ids = None
    variants = []
    for entry in entries:
        comparison = entry["comparison"]
        if comparison["role"] != "development":
            raise ValueError("Ablations run on development data only")
        strict, flexible = comparison["metrics"]["strict"], comparison["metrics"]["flexible"]
        if ids is None:
            ids = strict["num_examples"]
        elif strict["num_examples"] != ids:
            raise ValueError("Ablation variants evaluated different example counts")
        variants.append(
            {
                "name": entry["name"],
                "protocol": {
                    k: entry["protocol"][k]
                    for k in (
                        "num_fewshot",
                        "apply_chat_template",
                        "max_new_tokens",
                        "max_prompt_tokens",
                    )
                },
                "strict_delta_percentage_points": strict["delta_percentage_points"],
                "flexible_delta_percentage_points": flexible["delta_percentage_points"],
                "strict": {k: strict[k] for k in ("baseline_correct", "candidate_correct")},
                "flexible": {k: flexible[k] for k in ("baseline_correct", "candidate_correct")},
                "truncation": comparison["truncation"],
                "release_gate_passed": comparison["development_release_gate"]["passed"],
                "improved_ids": comparison["improved_ids"],
                "regressed_ids": comparison["regressed_ids"],
            }
        )
    passed = [v["release_gate_passed"] for v in variants]
    verdict = "robust" if all(passed) else "fragile" if any(passed) else "no_gain"
    stable = set(variants[0]["improved_ids"])
    regressed: set[str] = set()
    for variant in variants:
        stable &= set(variant["improved_ids"])
        regressed |= set(variant["regressed_ids"])
    # A gain that appears only where generations were cut off is a length artifact.
    uncapped = [
        v for v in variants if v["truncation"]["baseline"] == v["truncation"]["candidate"] == 0
    ]
    return {
        "role": "development",
        "num_examples": ids,
        "variants": variants,
        "verdict": verdict,
        "rule": (
            "robust: every variant improves strict accuracy without a flexible drop; "
            "fragile: only some do; no_gain: none do."
        ),
        "stable_strict_gain_ids": sorted(stable),
        "strict_regression_ids_any_variant": sorted(regressed),
        "uncapped_variants": [v["name"] for v in uncapped],
        "gain_without_truncation": any(v["release_gate_passed"] for v in uncapped),
        "limitations": [
            "Variants are predeclared, but all run on development data: exploratory only.",
            "A robust verdict on a small set still needs a larger disjoint development set.",
            "Flexible extraction can select an incidental number; it is a secondary diagnostic.",
        ],
    }
