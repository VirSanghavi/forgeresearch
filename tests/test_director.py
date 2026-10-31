import json

import pytest

from services.core import director


@pytest.fixture
def no_director_key(monkeypatch):
    for name in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)


def evidence():
    config = {"objective": "Fix production answers", "strategies": ["targeted_sft", "broad_sft"]}
    development = {
        "score": 0.0,
        "n": 2,
        "predictions": [{"question": "q", "output": "o", "expected": "1", "correct": False}],
    }
    production = {"flagged": 2, "traces": 4, "categories": [{"kind": "truncated", "count": 2}]}
    rows = [{"question": "How many?", "kinds": ["truncated"]}]
    return config, development, director.build_input(config, development, [], production, rows)


def test_local_director_targets_labeled_production_failures_first(no_director_key):
    config, development, context = evidence()
    proposal = director.propose(config, development, [], context)
    assert proposal["strategy"] == "targeted_sft"
    assert proposal["focus"] == "production_failures"
    assert proposal["director"] == "Evidence-driven local director"


def test_model_director_schema_allows_every_proposal_focus(monkeypatch):
    config, development, context = evidence()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sent = {}

    class Reply:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            proposal = {
                "strategy": "targeted_sft",
                "learning_rate": 0.0001,
                "lora_rank": 8,
                "hypothesis": "Train the labeled production failures first.",
                "focus": "production_failures",
            }
            return {
                "id": "resp-1",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(proposal)}]}],
            }

    def post(url, headers, json, timeout):
        sent.update(json)
        return Reply()

    monkeypatch.setattr(director.httpx, "post", post)
    proposal = director.propose(config, development, [], context)
    schema = sent["text"]["format"]["schema"]["properties"]["focus"]["enum"]
    assert set(schema) == set(director.FOCUS)
    assert proposal["focus"] == "production_failures"
