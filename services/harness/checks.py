"""Predeclared quality gates over JSON evidence, independent of command exit status."""

import json
import math
import operator

OPERATORS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "ge": operator.ge,
    "le": operator.le,
    "gt": operator.gt,
    "lt": operator.lt,
}


def evaluate_checks(checks, workspace):
    results = []
    for check in checks:
        try:
            actual = json.loads((workspace / check.artifact).read_text())
            for part in check.pointer.split("/")[1:]:
                key = part.replace("~1", "/").replace("~0", "~")
                actual = actual[int(key)] if isinstance(actual, list) else actual[key]
            if isinstance(actual, (dict, list)) or (isinstance(actual, float) and not math.isfinite(actual)):
                raise ValueError("Gate requires a finite scalar")
            if check.operator not in {"eq", "ne"}:
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (actual, check.value)):
                    raise ValueError("Ordered gates require numbers")
            # JSON booleans are not interchangeable with numeric scores (True != 1 here).
            compatible = isinstance(actual, bool) == isinstance(check.value, bool)
            passed = compatible and OPERATORS[check.operator](actual, check.value)
            results.append({**check.model_dump(), "passed": bool(passed)})
        except (ValueError, TypeError, KeyError, IndexError):
            results.append({**check.model_dump(), "passed": False, "error": "invalid_or_missing_scalar"})
    return results
