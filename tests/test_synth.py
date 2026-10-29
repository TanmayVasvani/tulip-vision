import numpy as np
import pytest

from conftest import small_style
from qc.synth import DEFECT_TYPES, default_roi, render


@pytest.mark.parametrize("defect", [None, *DEFECT_TYPES])
def test_render_is_deterministic_and_boxes_land_inside_the_roi(defect):
    st = small_style()
    a, info = render(st, np.random.default_rng(11), defect)
    b, _ = render(st, np.random.default_rng(11), defect)
    assert a.size == st.frame
    assert np.array_equal(np.asarray(a), np.asarray(b))
    if defect is None:
        assert info["bbox"] is None
        return
    x0, y0, x1, y1 = info["bbox"]
    rx0, ry0, rx1, ry1 = default_roi(st)
    w, h = st.frame
    assert rx0 * w - 2 <= x0 < x1 <= rx1 * w + 2 and ry0 * h - 2 <= y0 < y1 <= ry1 * h + 2


def test_defects_change_pixels_inside_their_box():
    st = small_style()
    good, _ = render(st, np.random.default_rng(4), None)
    bad, info = render(st, np.random.default_rng(4), "missing_tuft")
    x0, y0, x1, y1 = (int(v) for v in info["bbox"])
    diff = np.abs(np.asarray(good, float) - np.asarray(bad, float)).mean(-1)
    assert diff[y0:y1, x0:x1].mean() > 3 * diff.mean()
