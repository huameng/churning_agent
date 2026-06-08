from churning_agent.tools.escalation import StuckDetector, ask_human


def test_detector_trips_after_threshold_identical_signatures():
    d = StuckDetector(threshold=3)
    assert d.record("a") is False
    assert d.record("a") is False
    assert d.record("a") is True          # third identical -> stuck


def test_detector_resets_on_change():
    d = StuckDetector(threshold=3)
    d.record("a")
    d.record("a")
    assert d.record("b") is False         # page changed; counter restarts
    assert d.record("b") is False
    assert d.record("b") is True


def test_explicit_reset():
    d = StuckDetector(threshold=2)
    d.record("a")
    d.reset()
    assert d.record("a") is False         # counted fresh after reset


def test_ask_human_returns_signal():
    sig = ask_human("Do you shop at Dell?")
    assert sig["status"] == "needs_human"
    assert sig["question"] == "Do you shop at Dell?"
