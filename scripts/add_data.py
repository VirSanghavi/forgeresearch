"""Turn your own CSV or JSONL file into a task pack: `npm run add-data -- path/to/file.csv`.

It guesses the columns and the answer checker the same way the lab's "Add your data" wizard does
(components/setup-wizard.tsx), shows what it found on real rows, asks only what it cannot guess,
then validates and registers the pack with the same code as `POST /packs`. Every guess can be
overridden with a flag; `--yes` accepts the guesses without asking (and then needs `--terms`).
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.core import packs  # noqa: E402
from services.core.packs import register  # noqa: E402

QUESTION = ("question", "prompt", "input", "text", "instruction")
ANSWER = ("answer", "output", "reference", "label", "target", "response")
CHOSEN = ("chosen", "preferred", "good")
REJECTED = ("rejected", "dispreferred", "bad")
IDS = ("id", "uid", "ticket_id")
INSTRUCTIONS = {
    "number": "Answer the question. End with a final line in the form #### <number>.",
    "marker_text": "Answer the question. End with a final line in the form Answer: <answer>.",
    "exact": "Reply with the answer only.",
    "json": "Reply with a JSON object containing the requested fields only.",
}
CHECKER_HELP = {
    "number": "a number on a final '#### 42' line, compared as numbers",
    "marker_text": "the text after a final 'Answer:' line, ignoring case and spacing",
    "exact": "the whole reply, ignoring case and spacing",
    "json": "a JSON object whose fields must match the reference",
}


def checker_spec(choice, reference):
    """Mirror of checkerSpec() in components/setup-wizard.tsx."""
    if choice == "number":
        return {
            "extract": {"kind": "final_marker", "marker": "####", "grammar": "number"},
            "compare": {"kind": "numeric"},
        }
    if choice == "marker_text":
        return {
            "extract": {"kind": "final_marker", "marker": "Answer:"},
            "compare": {"kind": "exact", "normalize": ["strip", "casefold", "whitespace"]},
        }
    if choice == "exact":
        return {
            "extract": {"kind": "whole"},
            "compare": {"kind": "exact", "normalize": ["strip", "casefold", "whitespace"]},
            "reference": "whole",
        }
    try:
        keys = list(json.loads(reference[0] if reference else "{}"))
    except json.JSONDecodeError:
        keys = []
    fields = [{"path": "/" + k.replace("~", "~0").replace("/", "~1")} for k in keys]
    return {
        "extract": {"kind": "json"},
        "compare": {
            "kind": "json_fields",
            "fields": fields,
            "json_schema": {"type": "object", "required": keys},
        },
        "reference": "whole",
    }


def guess_checker(values):
    """Mirror of guess() in components/setup-wizard.tsx."""
    v = [x for x in values if x]
    if v and all(x.strip().startswith("{") for x in v):
        return "json"
    if v and all(re.search(r"####\s*[-\d.,]+\s*$", x) for x in v):
        return "number"
    if v and all(re.search(r"^answer:", x, re.I | re.M) for x in v):
        return "marker_text"
    return "exact"


def pick(columns, names):
    return next((c for c in columns if c.lower() in names), None)


def ask(question, default=None):
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer or default


def short(text, n=90):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", help="CSV with a header row, or JSONL with one object per line")
    parser.add_argument("--name", help="task name shown in the lab (default: from the file name)")
    parser.add_argument("--objective", help="what the model should get better at")
    parser.add_argument("--question", help="column with the input the model sees")
    parser.add_argument("--answer", help="column with the reference answer")
    parser.add_argument("--chosen", help="preference data: column with the better reply")
    parser.add_argument("--rejected", help="preference data: column with the worse reply")
    parser.add_argument("--id", dest="id_column", help="column with a stable row id")
    parser.add_argument("--checker", choices=sorted(CHECKER_HELP), help="how an answer is scored")
    parser.add_argument("--instruction", help="instruction shown to the model before each input")
    parser.add_argument("--license", default=None, help="the data's license (default: Proprietary)")
    parser.add_argument("--terms", help="your statement that this data may be used to train a model")
    parser.add_argument("--yes", action="store_true", help="accept every guess without asking")
    a = parser.parse_args()

    path = Path(a.file).expanduser()
    if not path.is_file():
        sys.exit(f"No such file: {path}")
    text = path.read_text(encoding="utf-8-sig")
    fmt = "csv" if path.suffix.lower() == ".csv" else "jsonl"
    rows = [r for _, r in register.parse(text, fmt) if r]
    if not rows:
        sys.exit(
            f"No rows found in {path.name}. Use a CSV with a header row, or JSONL with one object per line."
        )
    columns = list(rows[0])
    interactive = sys.stdin.isatty() and not a.yes

    question = a.question or pick(columns, QUESTION) or columns[0]
    answer = a.answer or pick(columns, ANSWER)
    chosen = a.chosen or pick(columns, CHOSEN)
    rejected = a.rejected or pick(columns, REJECTED)
    id_column = a.id_column or pick(columns, IDS)
    for flag, value in [
        ("question", question),
        ("answer", answer),
        ("chosen", chosen),
        ("rejected", rejected),
    ]:
        if value and value not in columns:
            sys.exit(f"--{flag} {value!r} is not a column. Columns: {', '.join(columns)}")
    kind = "preference" if chosen and rejected else "qa"
    values = [r.get(answer, "") for r in rows[:20]] if answer else []
    checker = a.checker or guess_checker(values)
    if kind == "qa" and checker == "json":
        kind = "extraction"

    print(f"\n{path.name}: {len(rows)} rows, columns: {', '.join(columns)}")
    if kind == "preference":
        print(f"  Preference data: prompt '{question}', better reply '{chosen}', worse reply '{rejected}'")
    else:
        if not answer:
            sys.exit(
                "Could not find a reference-answer column. Name it with --answer (columns: "
                + ", ".join(columns)
                + ")."
            )
        print(f"  The model sees:        '{question}'")
        print(f"  The correct answer is: '{answer}'")
        print(f"  Scored by:             {CHECKER_HELP[checker]}")
    if id_column:
        print(f"  Row ids:               '{id_column}'")
    print("\n  First rows:")
    for r in rows[:2]:
        print(f"    {short(r.get(question, ''))}")
        print(f"      -> {short(r.get(chosen if kind == 'preference' else answer, ''))}")

    name = a.name or path.stem.replace("-", " ").replace("_", " ")[:80]
    objective = a.objective
    terms = a.terms
    if interactive:
        print("\nPress Enter to accept a suggestion. Flags override any guess; see --help.")
        if input("Is that mapping right? [Y/n]: ").strip().lower() in ("n", "no"):
            sys.exit(
                "Rerun with --question, --answer (or --chosen/--rejected) and --checker set. See --help."
            )
        name = ask("Task name", name)
        objective = objective or ask("What should the model get better at?", f"Answer {name} correctly")
        while not terms or len(terms) < 20:
            terms = ask(
                "Why may this data be used to train a model? (recorded with every report)",
                "I own this data and permit training models on it.",
            )
    if not objective:
        objective = f"Answer {name} correctly"
    if not terms:
        sys.exit("State why this data may be used for training with --terms (at least 20 characters).")
    if len(name) < 3:
        name = f"{name} task"

    mapping = {"input_template": f"{{{question}}}"}
    if id_column:
        mapping["id"] = id_column
    if kind == "preference":
        mapping |= {"chosen": f"{{{chosen}}}", "rejected": f"{{{rejected}}}", "target_template": None}
    else:
        mapping["target_template"] = f"{{{answer}}}"
    draft = {
        "name": name,
        "objective": objective if len(objective) >= 10 else f"{objective} correctly and reliably",
        "kind": kind,
        "format": fmt,
        "mapping": mapping,
        "instruction": a.instruction
        or ("Reply to the user." if kind == "preference" else INSTRUCTIONS[checker]),
        "license": a.license or "Proprietary",
        "data_terms": terms,
    }
    if kind != "preference":
        draft["checker"] = checker_spec(checker, values)

    try:
        pack, pools, report = register.build(draft, text)
    except ValueError as error:
        sys.exit(f"\nThe data could not be registered: {error}")
    split = report["split"]
    print(
        f"\nChecked {report['rows']} rows: {split['train']} for training, {split['dev']} for development, "
        f"{split['sealed']} sealed for the final test."
    )
    if report["duplicates"]:
        print(f"  {report['duplicates']} duplicate questions were merged.")
    if report["invalid"]:
        # A reference that fails its own checker usually means the wrong column or checker.
        print(f"  {report['invalid']} rows were skipped, for example:")
        for bad in report["invalid_examples"][:5]:
            print(f"    line {bad['line']}: {bad['reason']}")
    for problem in report["problems"]:
        print(f"  Problem: {problem}")
    if report["problems"]:
        if report["invalid"] > report["valid"]:
            hint = "Most rows were skipped, so the answer column or --checker is probably wrong."
        else:
            # 20% sealed and 15% development by default, with at least 24 and 12 rows.
            need = max(register.MIN_SEALED / 0.2, register.MIN_DEV / 0.15)
            hint = f"The file is too small: it needs about {int(need)} usable rows ({report['valid']} now)."
        sys.exit(f"\nNot registered. {hint}")
    saved = packs.save(pack, pools, report)
    print(f"\nRegistered task pack '{saved.id}'.")
    print("Next: `npm run lab`, then New campaign and pick this task. It is already in the list.")


if __name__ == "__main__":
    main()
