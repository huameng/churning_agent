"""
Post-classifier tests for the pure prompt construction and the classify()
contract (response JSON -> Classification), with the LLM client mocked. Live
classification quality is covered by eval/eval_classifier.py.
"""

from churning_agent.tools import classifier
from churning_agent.tools.classifier import Classification, _build_prompt, classify
from churning_agent.tools.profile import UserProfile


def _profile() -> UserProfile:
    return UserProfile(state="FL", zip_code="33613", existing_accounts=["chase"])


def test_build_prompt_includes_title_content_and_profile():
    prompt = _build_prompt("Chase $300 Bonus", "Open a checking account...", _profile())
    assert "Chase $300 Bonus" in prompt
    assert "Open a checking account" in prompt
    assert "Minimum profit threshold" in prompt   # profile injected
    assert "chase" in prompt                       # existing account injected


class _FakeModels:
    def __init__(self, text):
        self._text = text

    def generate_content(self, **_kwargs):
        return type("R", (), {"text": self._text})()


class _FakeClient:
    def __init__(self, text):
        self.models = _FakeModels(text)


def test_classify_parses_model_json(monkeypatch):
    payload = Classification(
        label="MONEYMAKER", reasoning="clear cash bonus", estimated_value=300.0
    ).model_dump_json()
    monkeypatch.setattr(classifier, "load_profile", _profile)
    monkeypatch.setattr(classifier, "_get_client", lambda: _FakeClient(payload))

    result = classify("Chase $300 Bonus", "Open a checking account...")
    assert isinstance(result, Classification)
    assert result.label == "MONEYMAKER"
    assert result.estimated_value == 300.0
