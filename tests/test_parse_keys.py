"""Shared key matching for reply JSON (bench/parse.py): exact key wins, a near-miss key is used
only when it is the only one pointing at a target, otherwise the target stays unread."""

from asimov.bench.parse import locate_keys, normalize_key

EXACT = {"severity": "latent_severity_response", "risk": "latent_risk_response"}


def by_word(key: str) -> str | None:
    """A benchmark's own rule for near-miss keys; here: the first distinctive word decides."""
    words = normalize_key(key).split("_")
    return "severity" if "severity" in words else "risk" if "risk" in words else None


def test_normalize_key():
    assert normalize_key("Latent Severity-Response ") == "latent_severity_response"
    assert normalize_key("Q2") == "q2"


def test_exact_keys():
    found = locate_keys({"latent_severity_response": "C", "latent_risk_response": "D"}, EXACT, by_word)
    assert found == {"severity": ("latent_severity_response", True), "risk": ("latent_risk_response", True)}


def test_near_miss_key_used_when_unique():
    found = locate_keys({"latent_risk_severity_response": "C"}, EXACT, by_word)
    assert found == {"severity": ("latent_risk_severity_response", False)}


def test_exact_key_wins_over_near_key():
    obj = {"latent_severity_response": "C", "latent_risk_severity_response": "A"}
    assert locate_keys(obj, EXACT, by_word)["severity"] == ("latent_severity_response", True)


def test_competing_near_keys_leave_the_target_unread():
    assert "severity" not in locate_keys({"severity": "C", "Severity Answer": "D"}, EXACT, by_word)


def test_exact_keys_of_other_targets_are_never_near_misses():
    # 'latent_risk_response' is risk's exact key; it must not also count as a near-miss for anything
    found = locate_keys({"latent_risk_response": "D"}, EXACT, lambda key: "severity")
    assert found == {"risk": ("latent_risk_response", True)}


def test_keys_matching_nothing_are_ignored():
    assert locate_keys({"notes": "x"}, EXACT, by_word) == {}
