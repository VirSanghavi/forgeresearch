"""`npm run add-data`: a company's file becomes a task pack from the command line."""

import csv
import json
import sys

import pytest

from services.core import packs, store

sys.path.insert(0, str(store.ROOT / "scripts"))
import add_data  # noqa: E402

TERMS = "Synthetic rows written for this test only."


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(store, "campaigns", lambda: [])
    return tmp_path


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["add_data.py", *map(str, argv)])
    add_data.main()


def orders(path, n=200):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticket_id", "question", "answer"])
        for i in range(n):
            w.writerow([f"T{i:04d}", f"What is {i} + {i * 3}?", f"{i} + {i * 3} = {i * 4}\n#### {i * 4}"])
    return path


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["work\n#### 42", "#### 7"], "number"),
        (["Because...\nAnswer: Paris", "answer: Lima"], "marker_text"),
        (['{"city": "Paris"}', '{"city": "Lima"}'], "json"),
        (["Paris", "#### 7"], "exact"),
        ([], "exact"),
    ],
)
def test_checker_guesses_match_the_lab_wizard(values, expected):
    assert add_data.guess_checker(values) == expected


def test_a_csv_is_registered_with_guessed_columns_and_checker(tmp_path, monkeypatch, capsys):
    run(monkeypatch, orders(tmp_path / "orders.csv"), "--yes", "--terms", TERMS)
    out = capsys.readouterr().out
    assert "Registered task pack 'orders-" in out
    [pack] = [p for p in packs.available() if p.id.startswith("orders-")]
    assert pack.kind == "qa"
    assert pack.task.input_template == "{question}" and pack.task.target_template == "{answer}"
    checker = pack.task.make_checker()
    assert checker.score("so 8\n#### 8", "#### 8")["correct"]
    assert not checker.score("#### 9", "#### 8")["correct"]
    report = json.loads((packs.folder(pack.id) / "report.json").read_text())
    assert report["split"] == {"sealed": 40, "dev": 30, "train": 130}
    # Registering the same file and answers again returns the same immutable pack.
    run(monkeypatch, tmp_path / "orders.csv", "--yes", "--terms", TERMS)
    assert len([p for p in packs.available() if p.id.startswith("orders-")]) == 1


def test_jsonl_preference_and_extraction_data_are_recognized(tmp_path, monkeypatch):
    pref = tmp_path / "replies.jsonl"
    pref.write_text(
        "".join(
            json.dumps({"prompt": f"Say hi to user {i}", "chosen": f"Hi user {i}!", "rejected": "no"}) + "\n"
            for i in range(160)
        )
    )
    run(monkeypatch, pref, "--yes", "--terms", TERMS)
    extraction = tmp_path / "invoices.jsonl"
    extraction.write_text(
        "".join(
            json.dumps({"input": f"Invoice {i} for ${i}.00", "output": json.dumps({"number": str(i)})}) + "\n"
            for i in range(160)
        )
    )
    run(monkeypatch, extraction, "--yes", "--terms", TERMS)
    kinds = {p.id.split("-")[0]: p.kind for p in packs.available() if p.id != "gsm8k"}
    assert kinds == {"replies": "preference", "invoices": "extraction"}


def test_too_little_data_is_refused_with_how_much_is_needed(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="needs about 120 usable rows \\(30 now\\)"):
        run(monkeypatch, orders(tmp_path / "small.csv", n=30), "--yes", "--terms", TERMS)
    assert not (store.DATA / "packs").exists() or not list((store.DATA / "packs").iterdir())


def test_a_wrong_answer_column_is_refused_as_a_mapping_problem(tmp_path, monkeypatch):
    path = orders(tmp_path / "orders.csv")
    with pytest.raises(SystemExit, match="answer column or --checker is probably wrong"):
        run(monkeypatch, path, "--yes", "--answer", "ticket_id", "--checker", "number", "--terms", TERMS)
    with pytest.raises(SystemExit, match="is not a column"):
        run(monkeypatch, path, "--yes", "--answer", "missing", "--terms", TERMS)


def test_without_a_terminal_the_training_terms_must_be_stated(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="--terms"):
        run(monkeypatch, orders(tmp_path / "orders.csv"), "--yes")
