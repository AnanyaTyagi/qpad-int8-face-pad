import numpy as np
import pandas as pd
import pytest

from qpad.annotations import (annotation_summary, infer_frame_base, parse_eye_annotation, resolve_dataset_frame_base,
                              valid_eye_mask)


def test_parse_comma_and_space_formats():
    p = parse_eye_annotation("1, 10, 20, 50, 21\n2 11 20 51 21\n\n3,12,20,52,21\n")
    assert p.n_malformed == 0 and len(p.table) == 3
    assert p.table["frame"].tolist() == [1, 2, 3]
    assert p.table.iloc[0].tolist() == [1, 10, 20, 50, 21]


def test_malformed_lines_counted_not_dropped_silently():
    p = parse_eye_annotation("1, 10, 20, 50, 21\nbad line\n2, 1, 2, 3\n2.5, 1, 2, 3, 4\n")
    assert len(p.table) == 1 and p.n_malformed == 3 and p.n_lines == 4
    assert "bad line" in p.malformed_examples


def test_valid_mask():
    t = pd.DataFrame({"frame": [1, 2, 3, 4], "x1": [10, 0, 10, np.nan], "y1": [20, 0, 20, 1],
                      "x2": [50, 0, 12, 5], "y2": [20, 0, 20, 5]})
    assert valid_eye_mask(t, 10.0).tolist() == [True, False, False, False]


@pytest.mark.parametrize("frames,n,base", [([0, 1, 2], 3, "0"), ([1, 2, 3], 3, "1"), ([1, 2], 3, "ambiguous"),
                                           ([0, 1, 2, 3], 3, "ambiguous"), ([], 3, "ambiguous")])
def test_infer_frame_base(frames, n, base):
    assert infer_frame_base(np.array(frames), n) == base


def test_resolve_dataset_frame_base():
    assert resolve_dataset_frame_base("auto", pd.Series(["1", "1", "ambiguous"])) == 1
    assert resolve_dataset_frame_base(0, pd.Series(["1"])) == 0
    with pytest.raises(ValueError):
        resolve_dataset_frame_base("auto", pd.Series(["0", "1"]))
    with pytest.raises(ValueError):
        resolve_dataset_frame_base("auto", pd.Series(["ambiguous"]))


def test_annotation_summary_coverage():
    text = "\n".join(f"{i}, 10, 20, 60, 20" for i in range(1, 31))
    s = annotation_summary(parse_eye_annotation(text), 30, 10.0)
    assert s["ann_frame_base"] == "1" and s["ann_valid_in_range"] == 30 and s["ann_malformed_rows"] == 0
