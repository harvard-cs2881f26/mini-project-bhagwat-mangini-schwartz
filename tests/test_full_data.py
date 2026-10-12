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
