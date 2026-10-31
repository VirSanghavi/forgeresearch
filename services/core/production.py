"""Find failures in production traffic without the answer key, and turn them into data.

Detectors only use what a deployed model leaves behind: its output, the
generation stop reason and user feedback. When reviewer labels exist they are
used twice, never earlier: to audit the detectors (precision and recall), and to
give labeled failures a reference so a campaign can train on half of them and
measure the other half before and after.
"""

import ast
import hashlib
import json
import operator
import re
from decimal import Decimal, InvalidOperation

from forge_training.data import question_key
from forge_training.scoring import extract_answer

MAX_TRACES = 5000
DETECTORS = {
    "arithmetic_slip": "A calculation written in the answer is wrong",
    "missing_final_answer": "No final '#### <number>' line, so the answer cannot be used",
    "truncated": "Generation stopped at the length limit",
    "user_reported": "The user marked the answer wrong or sent a correction",
}
# Detectors for any other task pack: its own checker decides whether an answer is usable.
GENERIC_DETECTORS = {
    "format_failure": "The answer does not have the format the task's checker reads",
    "truncated": DETECTORS["truncated"],
    "user_reported": DETECTORS["user_reported"],
}


def _math(pack):
    return pack is None or pack.id == "gsm8k"


def detectors(pack=None):
    return DETECTORS if _math(pack) else GENERIC_DETECTORS


NUMBER = r"\$?\d[\d,]*(?:\.\d+)?"
TERM = rf"\(?\s*{NUMBER}\s*\)?"
EXPRESSION = rf"{TERM}(?:\s*[-+*/]\s*{TERM})*"
# A chain of two or more arithmetic expressions joined by '=': "4*8 + 3 = 32 + 3 = 35".
CHAIN = re.compile(
    rf"(?<![\w.*/+\-=])({EXPRESSION}(?:\s*=\s*{EXPRESSION})+)(?![\d,]|\.\d)(?![ \t]*[-+*/][ \t]*[\d($])"
)
LATEX = [
    (re.compile(r"\\(?:times|cdot)"), "*"),
    (re.compile(r"\\div"), "/"),
    (re.compile(r"[×·]"), "*"),
    (re.compile(r"÷"), "/"),
    (re.compile(r"(?<=\d)\s*x\s*(?=\$?\d)"), " * "),
    (re.compile(r"\\[\[\]()]|\\[,;!]"), " "),
]
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _evaluate(expression):
    def walk(node):
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return Decimal(str(node.value))
        raise ValueError("unsupported expression")

    return walk(ast.parse(re.sub(r"[$,]", "", expression), mode="eval").body)


def _places(expression):
    return max((len(n.split(".")[1]) for n in re.findall(r"\d+\.\d+", expression)), default=0)


def normalize(output):
    for pattern, replacement in LATEX:
        output = pattern.sub(replacement, output)
    return output


def arithmetic_slips(output):
    """Chained equations whose sides have different values, e.g. '12 * 7 = 74'.

    Each side of 'A = B = C' must be a pure arithmetic expression. A difference is
    a slip only beyond rounding to the written precision and 0.5% (rounded
    intermediate values propagate), so correct but rounded work is not flagged.
    """
    slips = []
    text = normalize(output)
    for match in CHAIN.finditer(text):
        before = text[: match.start()].rstrip()[-1:]
        if before and before in "*/+-=%^" or "%" in text[match.start() : match.end() + 1]:
            continue  # part of a longer or percent expression we cannot read safely
        sides = [side.strip() for side in match.group(1).split("=")]
        for left, right in zip(sides, sides[1:], strict=False):
            if not re.search(r"\d\s*[-+*/]\s*[\d($]", left + " " + right):
                continue  # "1.5 = 2" restates numbers; it is not a calculation
            try:
                a, b = _evaluate(left), _evaluate(right)
            except (ValueError, SyntaxError, ZeroDivisionError, InvalidOperation, TypeError):
                continue
            tolerance = max(
                Decimal(1).scaleb(-max(_places(left), _places(right))) / 2, abs(a) * Decimal("0.005")
            )
            if abs(a - b) > tolerance:
                slips.append({"equation": f"{left} = {right}", "computed": f"{a.normalize():f}"})
                break
    return slips


def detect(trace, pack=None):
    if not _math(pack):
        return detect_generic(trace, pack)
    output = trace["output"]
    signals = []
    for slip in arithmetic_slips(output):
        signals.append({"kind": "arithmetic_slip", "evidence": slip})
    if extract_answer(output) is None:
        signals.append({"kind": "missing_final_answer", "evidence": {"tail": output[-120:]}})
    if trace.get("finish_reason") == "length":
        signals.append({"kind": "truncated", "evidence": {"finish_reason": "length"}})
    feedback = trace.get("feedback") or {}
    if feedback.get("rating", 0) < 0 or feedback.get("correction"):
        signals.append({"kind": "user_reported", "evidence": {k: feedback[k] for k in feedback}})
    return signals


def detect_generic(trace, pack):
    signals = []
    if not pack.task.make_checker().format_ok(trace["output"]):
        signals.append({"kind": "format_failure", "evidence": {"tail": trace["output"][-120:]}})
    if trace.get("finish_reason") == "length":
        signals.append({"kind": "truncated", "evidence": {"finish_reason": "length"}})
    feedback = trace.get("feedback") or {}
    if feedback.get("rating", 0) < 0 or feedback.get("correction"):
        signals.append({"kind": "user_reported", "evidence": {k: feedback[k] for k in feedback}})
    return signals


def _trace(row):
    # Tolerant on read: a malformed row is reported and skipped, never fatal to the scan.
    if not isinstance(row, dict):
        raise ValueError("row is not an object")
    trace = {}
    for key in ("id", "prompt", "output"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing {key}")
        trace[key] = value
    if len(trace["prompt"]) > 32000 or len(trace["output"]) > 32000:
        raise ValueError("prompt or output too long")
    trace["finish_reason"] = row.get("finish_reason") if isinstance(row.get("finish_reason"), str) else None
    feedback = row.get("feedback")
    if isinstance(feedback, dict):
        rating = feedback.get("rating")
        correction = feedback.get("correction")
        trace["feedback"] = {
            **({"rating": int(rating)} if isinstance(rating, int | float) else {}),
            **({"correction": str(correction)[:2000]} if correction else {}),
        }
    trace["timestamp"] = str(row.get("timestamp", ""))[:40]
    return trace


def parse(text):
    traces, malformed, seen = [], [], set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            trace = _trace(json.loads(line))
            if trace["id"] in seen:
                raise ValueError("duplicate id")
            seen.add(trace["id"])
            traces.append(trace)
        except (ValueError, TypeError) as error:
            malformed.append({"line": line_number, "reason": str(error)[:120]})
        if len(traces) > MAX_TRACES:
            raise ValueError(f"At most {MAX_TRACES} traces per scan")
    return traces, malformed


def parse_labels(text, pack=None):
    labels = {}
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if _math(pack):
                if extract_answer(row["answer"]) is None:
                    raise ValueError(f"Label {row['id']} must end with a numeric '#### <answer>' line")
            else:
                try:
                    pack.task.make_checker().check_reference(row["answer"])
                except ValueError as error:
                    raise ValueError(f"Label {row['id']}: {error}") from None
            labels[row["id"]] = row["answer"]
    return labels


def wrong(output, label, pack=None):
    """Whether a deployed answer disagrees with its reviewer label."""
    if _math(pack):
        actual = extract_answer(output)
        return actual is None or actual != extract_answer(label)
    return not pack.task.make_checker().score(output, label)["correct"]


def scan(text, labels_text=None, sealed_questions=(), source="upload", pack=None):
    traces, malformed = parse(text)
    labels = parse_labels(labels_text, pack) if labels_text else {}
    kinds = detectors(pack)
    sealed = {question_key(q) for q in sealed_questions}
    excluded = [t["id"] for t in traces if question_key(t["prompt"]) in sealed]
    findings, counts = [], {kind: 0 for kind in kinds}
    for trace in traces:
        if trace["id"] in excluded:
            continue
        signals = detect(trace, pack)
        for kind in {s["kind"] for s in signals}:
            counts[kind] += 1
        if signals:
            findings.append({"trace": trace, "signals": signals})
    report = {
        "source": source,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "traces": len(traces),
        "malformed": malformed,
        "excluded_sealed_overlap": excluded,
        "flagged": len(findings),
        "categories": [{"kind": kind, "label": kinds[kind], "count": counts[kind]} for kind in kinds],
        "findings": findings,
        "audit": audit(traces, findings, labels, set(excluded), pack) if labels else None,
    }
    if not _math(pack):
        report["task_pack"] = pack.id
    return report


def audit(traces, findings, labels, excluded, pack=None):
    """Precision and recall of 'flagged' against reviewer labels; labels are not used to detect."""
    flagged = {f["trace"]["id"] for f in findings}
    tp = fp = fn = tn = 0
    for trace in traces:
        if trace["id"] in excluded or trace["id"] not in labels:
            continue
        bad = wrong(trace["output"], labels[trace["id"]], pack)
        hit = trace["id"] in flagged
        tp += hit and bad
        fp += hit and not bad
        fn += bad and not hit
        tn += not hit and not bad
    labeled = tp + fp + fn + tn
    return {
        "labeled": labeled,
        "actually_wrong": tp + fn,
        "true_positives": tp,
        "false_positives": fp,
        "missed": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def campaign_rows(report, labels_text, pack=None):
    """Reviewer-labeled, wrongly answered production prompts, split by a stable hash into train and held out."""
    labels = parse_labels(labels_text, pack)
    rows = []
    for finding in report["findings"]:
        trace = finding["trace"]
        # Only reviewer labels become training targets. A user correction is a detection
        # signal, never a label: anyone can send one, so it could poison training.
        answer = labels.get(trace["id"])
        if answer is None:
            continue
        if not wrong(trace["output"], answer, pack):
            continue  # flagged, but the final answer was right; not a training target
        rows.append(
            {
                "id": f"prod-{hashlib.sha256(trace['id'].encode()).hexdigest()[:12]}",
                "trace_id": trace["id"],
                "question": trace["prompt"],
                "answer": answer,
                "kinds": sorted({s["kind"] for s in finding["signals"]}),
            }
        )
    train, held_out = [], []
    for row in rows:
        bucket = int(hashlib.sha256(question_key(row["question"]).encode()).hexdigest(), 16) % 2
        (held_out if bucket else train).append(row)
    return train, held_out


DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def gsm8k(split):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "openai/gsm8k",
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename=f"main/{split}-00000-of-00001.parquet",
    )
    return load_dataset("parquet", data_files={split: path}, split=split)


def sealed_questions():
    """Question text only, to exclude overlap. No model ever answers these here."""
    return [row["question"] for row in gsm8k("test")]


def simulated_traffic(count, seed=2026):
    """Stand-in production prompts: a fixed slice of the official train split.

    Campaigns drop any of their own training rows that share a question with the
    production scan, so held-out production failures are never trained on.
    """
    import random

    rows = gsm8k("train")
    ids = list(range(len(rows)))
    random.Random(seed).shuffle(ids)
    return [dict(rows[i]) | {"id": f"live-{i}"} for i in ids[-count:]]


def _folder():
    from services.core import store

    folder = store.DATA / "production"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


SCAN_ID = re.compile(r"^scan-[0-9a-f]{8}$")


def save_scan(name, traces_text, labels_text=None, source="upload", pack=None):
    import uuid
    from services.core import store

    if _math(pack):
        sealed = sealed_questions()
    else:
        from services.core import packs

        sealed = [row["question"] for row in packs.read(pack.id, "sealed")]
    report = scan(traces_text, labels_text, sealed, source, pack)
    report |= {"id": f"scan-{uuid.uuid4().hex[:8]}", "name": name[:80], "created_at": store.now()}
    folder = _folder()
    (folder / f"{report['id']}.traces.jsonl").write_text(traces_text)
    if labels_text:
        (folder / f"{report['id']}.labels.jsonl").write_text(labels_text)
    (folder / f"{report['id']}.json").write_text(json.dumps(report, indent=2))
    return report


def load_scan(scan_id):
    if not SCAN_ID.match(scan_id or ""):
        return None
    path = _folder() / f"{scan_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_labels(scan_id):
    path = _folder() / f"{scan_id}.labels.jsonl"
    return path.read_text() if SCAN_ID.match(scan_id) and path.exists() else ""


def summary(report):
    return {k: report[k] for k in report if k not in {"findings", "malformed"}} | {
        "malformed": len(report["malformed"])
    }


def scans():
    return sorted(
        (summary(json.loads(p.read_text())) for p in _folder().glob("scan-*.json")),
        key=lambda r: r["created_at"],
        reverse=True,
    )
