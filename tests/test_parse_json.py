"""Reading the answer object out of a reply (bench/parse): nested JSON uses the whole object,
broken JSON falls back to reading "key": "value" pairs. Cases are shaped like real GPT-5 Nano
replies from the Fig. 4 run (text_0267, text_0100, text_0171, text_0056, text_0226)."""

from asimov.bench.parse import duplicate_keys, json_object, loose_pairs

FLAT = '{"latent_risk_response": "D", "effect_response": "B"}'


def test_flat_object_unchanged():
    assert json_object(FLAT) == {"latent_risk_response": "D", "effect_response": "B"}


def test_code_fence_and_prose_around_the_json():
    assert json_object("Here you go:\n```json\n" + FLAT + "\n```\nHope that helps.") == json_object(FLAT)


def test_nested_object_returns_the_whole_object_not_the_inner_one():
    text = '{\n "latent_risk_response": "E",\n "activated_risk_details": {"injury_type": "Hematoma", "severity": "Severe"}\n}'
    d = json_object(text)
    assert d is not None and d["latent_risk_response"] == "E"
    assert d["activated_risk_details"]["injury_type"] == "Hematoma"


def test_last_top_level_object_wins():
    text = 'Format: {"latent_risk_response": "X"}\nAnswer:\n{"latent_risk_response": "C"}'
    assert json_object(text) == {"latent_risk_response": "C"}


def test_paper_stray_brace_still_reads_the_answer_object():
    text = '{"latent_risk_response": "D", "activated_risk_response": "A"}, "activated_risk_rationale": "x"}'
    assert json_object(text) == {"latent_risk_response": "D", "activated_risk_response": "A"}


def test_single_quoted_dict():
    assert json_object("{'latent_risk_response': 'B'}") == {"latent_risk_response": "B"}


def test_broken_json_is_not_an_object():
    assert json_object('{"latent_risk_response": "C", "latent_risk_rationale": "the "No injury" option"}') is None


def test_loose_pairs_read_broken_json():
    text = (
        '{\n "latent_risk_response": "E",\n "latent_risk_rationale": "but the "No realistic injury" option",\n'
        ' "latent_severity_response": "B. Minor Injury",\n "effect_response": "B",\n'
        ' "activated_risk_response": "B",\n "activated_risk_rationale": "ends badly \n  "\n}'
    )
    pairs = loose_pairs(text)
    assert pairs is not None
    assert pairs["latent_risk_response"] == "E" and pairs["latent_severity_response"] == "B. Minor Injury"
    assert pairs["effect_response"] == "B" and pairs["activated_risk_response"] == "B"


def test_loose_pairs_none_without_any_pair():
    assert loose_pairs("I'm sorry, I can't help with that.") is None


# --- duplicate keys keep the first value (Noa); real case: Claude Sonnet text_0245 ---


def test_duplicate_key_keeps_the_first_value():
    d = json_object('{"latent_severity_response": "D", "latent_severity_response": "Electric shock can be severe."}')
    assert d == {"latent_severity_response": "D"}


def test_nested_duplicate_keys_keep_the_first_value():
    text = '{"a": "1", "details": {"injury": "Burn", "injury": "Shock"}, "a": "2", "list": [{"k": "x", "k": "y"}]}'
    assert json_object(text) == {"a": "1", "details": {"injury": "Burn"}, "list": [{"k": "x"}]}


def test_single_quoted_dict_with_duplicates_keeps_the_first_value():
    text = "{'latent_risk_response': 'B', 'nested': {'x': 1, 'x': 2}, 'latent_risk_response': 'prose'}"
    assert json_object(text) == {"latent_risk_response": "B", "nested": {"x": 1}}


def test_duplicate_keys_are_reported():
    assert duplicate_keys(json_object('{"a": "1", "b": "2", "a": "3"}')) == {"a"}
    assert duplicate_keys(json_object("{'a': '1', 'a': '3', 'b': '2'}")) == {"a"}
    assert duplicate_keys(json_object(FLAT)) == set()
    assert duplicate_keys({"a": "1"}) == set()  # plain dicts (e.g. from loose_pairs) have none


# --- review fixes: odd replies never crash scoring; loose keys may contain spaces ---


def test_unhashable_dict_key_is_not_an_object_and_does_not_raise():
    assert json_object('{[1]: "x"}') is None
    assert json_object('{"a": {[1]: "x"}, "latent_risk_response": "B"}') is None  # loose_pairs reads it instead


def test_deeply_nested_reply_does_not_raise():
    assert json_object("{'a': " + "[" * 5000 + "]" * 5000 + "}") is None


def test_loose_pairs_keep_keys_with_spaces():
    pairs = loose_pairs('{"Latent Severity Response": "C", "x": "broken "quote" here"}')
    assert pairs is not None and pairs["Latent Severity Response"] == "C"
