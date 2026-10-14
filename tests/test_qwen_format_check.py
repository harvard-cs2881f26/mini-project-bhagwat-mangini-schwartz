"""scripts/qwen_format_check.py: how a failing reply's causes are named (offline, no results needed)."""

import importlib.util
import json

from asimov.runner import ROOT

spec = importlib.util.spec_from_file_location("qwen_format_check", ROOT / "scripts" / "qwen_format_check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

OK = dict.fromkeys(("latent_risk", "latent_severity", "effect", "activated_risk"), "ok")


def test_e_on_a_severity_question_is_its_own_cause():
    text = json.dumps({"latent_risk_response": "E", "latent_severity_response": "E", "effect_response": "C"})
    item = {"parse": {**OK, "latent_severity": "invalid_letter", "activated_risk": "missing_key"}, "finish": "complete"}
    assert check.causes(item, {"text": text}) == {"invalid E", "missing key"}


def test_a_cut_off_reply_is_counted_as_cut_off_not_as_missing_keys():
    item = {"parse": {**OK, "activated_risk": "missing_key"}, "finish": "truncated"}
    assert check.causes(item, {"text": '{"latent_risk_response": "A"'}) == {"cut off"}


def test_other_invalid_values_are_kept_apart():
    text = json.dumps({"latent_severity_response": "F"})
    item = {"parse": {**OK, "latent_severity": "invalid_letter"}, "finish": "complete"}
    assert check.causes(item, {"text": text}) == {"invalid other"}
