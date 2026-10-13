"""Facts about the full download (data/), checked against dataset_info.json counts and
data/README.md. Skipped on the 5-item sample (see conftest.py)."""
import pytest

from asimov.data.image import load_image_items
from asimov.data.rows import read_rows
from asimov.data.video import load_video_answers

pytestmark = pytest.mark.full_data


@pytest.mark.parametrize("folder,n", [("text", 319), ("video", 287), ("image", 164), ("image_rationale", 164)])
def test_item_counts(folder, n):
    assert len(read_rows(folder)) == n


def test_video_answer_key_split():
    answers = load_video_answers().values()
    assert sum(a.risk == "Yes" for a in answers) == 94
    assert sum(a.last_intervention_s == -1 for a in answers) == 98


def test_image_variants_load():
    assert len(load_image_items("image")) == len(load_image_items("image_rationale")) == 164


def test_text_majority_baselines():
    # docs/PLAN.md §1 / paper App. D: always answering the most common letter
    from collections import Counter

    from asimov.bench.text_score import QUESTIONS, score
    from asimov.data.text import load_text_answers

    answers = load_text_answers()
    majority = {q: Counter(a[q] for a in answers.values()).most_common(1)[0][0] for q in QUESTIONS}
    reply = "{" + ", ".join(f'"{q}_response": "{v}"' for q, v in majority.items()) + "}"
    s = score(answers, {i: reply for i in answers})
    assert s["n"] == 319
    assert round(s["latent_severity_accuracy"] * 319) == 142
    assert round(s["latent_severity_severe_binary_accuracy"] * 319) == 180
    assert round(s["effect_accuracy"] * 319) == 121
    assert round(s["activated_risk_accuracy"] * 319) == 108


def test_every_text_prompt_has_four_option_lists():
    from asimov.bench.text_score import QUESTIONS
    from asimov.data.text import load_text_options

    options = load_text_options()
    assert len(options) == 319
    assert all([len(o[q]) for q in QUESTIONS] == [5, 4, 4, 4] for o in options.values())
