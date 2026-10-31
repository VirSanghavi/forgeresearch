"""Decide what a sealed final comparison may claim.

A higher score alone is not a validated improvement: on 24 questions one extra
correct answer is +4.2 points and well inside noise. The claim is validated only
when the paired bootstrap 95% interval of per-question differences (training
core's own estimator) lies entirely above zero.
"""

from forge_training.scoring import paired_bootstrap


def final_verdict(baseline, candidate, seed):
    before, after = baseline["predictions"], candidate["predictions"]
    if len(before) != len(after) or not before:
        raise ValueError("Final evaluations differ in size or are empty")
    differences = []
    for b, a in zip(before, after, strict=True):
        if b["id"] != a["id"]:
            raise ValueError(f"Final evaluations differ at {b['id']} / {a['id']}")
        differences.append(int(a["correct"]) - int(b["correct"]))
    interval = paired_bootstrap(differences, seed)
    delta = (candidate["score"] - baseline["score"]) * 100
    if interval[0] > 0:
        claim = "validated"
    elif delta > 0:
        claim = "uncertain"
    else:
        claim = "none"
    return {
        "improvement_pp": delta,
        "paired_bootstrap_95_pp": interval,
        "improved_examples": differences.count(1),
        "regressed_examples": differences.count(-1),
        "improvement_claim": claim,
        "valid_improvement": claim == "validated",
    }


SUMMARY = {
    "validated": "Validated improvement: the 95% paired interval is above zero.",
    "uncertain": "Higher score observed, but within uncertainty. No validated improvement claimed.",
    "none": "No improvement observed. No validated improvement claimed.",
}
