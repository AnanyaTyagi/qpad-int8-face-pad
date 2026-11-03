"""Pilot diagnostics on synthetic results (no TensorRT, no biometric data)."""

import numpy as np
import pandas as pd
import pytest

from qpad import diagnostics as D
from qpad.experiments import pad_to_batch
from qpad.reproducibility import write_json


def videos(scores, classes=("bona_fide", "print", "replay")):
    n = len(scores) // len(classes)
    return pd.DataFrame({"video_id": [f"v{i}" for i in range(len(scores))],
                         "class": [c for c in classes for _ in range(n)], "score": scores})


def test_margin_table_counts_videos_near_threshold():
    v = videos([0.99, 0.72, 0.60, 0.01, 0.74, 0.20, 0.05, 0.10, 0.80])
    t = D.margin_table(v, 0.73).set_index("class")
    assert t.loc["bona_fide", "within_0.02"] == 1 and t.loc["bona_fide", "within_0.1"] == 1
    assert t.loc["print", "within_0.02"] == 1 and t.loc["print", "within_0.05"] == 1
    assert t.loc["replay", "within_0.1"] == 1 and t.loc["replay", "within_0.05"] == 0
    assert (t["videos"] == 3).all()


def test_paired_score_shift_and_mismatch():
    a = videos([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1])
    b = a.assign(score=a["score"] + np.r_[0, 0, 0, 0.02, 0, 0, -0.05, 0, 0])
    s = D.paired_score_shift(a, b).set_index("class")
    assert s.loc["bona_fide", "max_abs_delta"] == 0
    assert s.loc["print", "moved_over_0.01"] == 1 and s.loc["replay", "mean_delta"] == pytest.approx(-0.05 / 3)
    with pytest.raises(ValueError):
        D.paired_score_shift(a, b.iloc[:-1])


def test_compare_pairs_reads_int8_and_fp32_results(tmp_path):
    base = videos([0.9, 0.8, 0.7, 0.1, 0.75, 0.3, 0.1, 0.1, 0.1])
    fp32, int8 = tmp_path / "fp32", tmp_path / "int8"
    for d in (fp32, int8):
        d.mkdir()
    base.to_csv(fp32 / "test_video_scores.csv", index=False)
    write_json(fp32 / "metrics.json", {})
    write_json(fp32 / "threshold.json", {"threshold": 0.73})
    base.assign(score=base["score"] - np.r_[0, 0, 0, 0, 0.04, 0, 0, 0, 0]).to_csv(int8 / "test_video_scores.csv",
                                                                                 index=False)
    write_json(int8 / "metrics.json", {"threshold_A_fp32": 0.73, "threshold_B_engine": {"threshold": 0.70}})
    shift, flips = D.compare_pairs([("int8 vs fp32", fp32, int8)], "A")
    assert flips.iloc[0]["attack_accepted_to_rejected"] == 1        # 0.75 -> 0.71 crosses 0.73
    _, flips_b = D.compare_pairs([("int8 vs fp32", fp32, int8)], "B")
    assert flips_b.iloc[0]["attack_accepted_to_rejected"] == 0      # 0.71 is still above 0.70
    assert set(shift["comparison"]) == {"int8 vs fp32"}


def test_engine_metadata_table_flags_incomplete_calibration(tmp_path):
    dirs = {}
    for tag, served in (("a", 30), ("b", 29)):
        d = tmp_path / tag
        d.mkdir()
        write_json(d / "engine_meta.json", {"sha256": "f" * 64, "method": "entropy", "n_calibration_images": 600,
                                            "calibration_batches_served": served, "n_scaled_tensors": 132,
                                            "build_seconds": 12.3, "hardware": {"gpu_name": "GPU", "tensorrt": "10"}})
        dirs[tag] = d
    t = D.engine_metadata_table(dirs, 30).set_index("engine")
    assert t.loc["a", "ok"] and not t.loc["b", "ok"]


def test_block_of_maps_exported_node_names():
    assert D.block_of("/enc/enc.4/denselayer3/conv2/Conv") == "denseblock1"
    assert D.block_of("/enc/enc.7/conv/Conv") == "transition2"
    assert D.block_of("/dec/Conv") == "pixel-map head"
    assert D.block_of("/linear/Gemm") == "binary head"
    assert D.block_of("Sigmoid_3") == "other"


def test_scale_changes_located_in_onnx_graph(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    nodes = [helper.make_node("Relu", ["input"], ["a"], name="/enc/enc.2/Relu"),
             helper.make_node("Relu", ["a"], ["b"], name="/enc/enc.4/denselayer1/relu1/Relu"),
             helper.make_node("Sigmoid", ["b"], ["pixel_map"], name="/dec/Sigmoid")]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info("pixel_map", TensorProto.FLOAT, [1, 4])])
    path = tmp_path / "m.onnx"
    onnx.save(helper.make_model(graph), str(path))
    order = D.onnx_tensor_order(path)
    assert order.set_index("tensor").loc["b", "block"] == "denseblock1"
    scales = {"nat": {"input": 1.0, "a": 1.0, "b": 2.0, "x": 1.0},
              "odd": {"input": 1.0, "a": 1.1, "b": 3.0, "x": 1.0}}
    changes = D.scale_changes_by_layer(scales, "nat", order)
    assert changes.set_index("tensor").loc["b", "abs_rel"] == pytest.approx(0.5)
    assert changes.set_index("tensor").loc["x", "block"] == "not in ONNX graph"
    by_block = D.scale_change_by_block(changes).set_index("block")
    assert by_block.loc["denseblock1", "max_change_pct"] == pytest.approx(50.0)
    assert list(D.scale_change_by_block(changes)["block"])[:3] == ["input", "relu0", "denseblock1"]


def test_layer_precision_from_inspector_json():
    layers = [
        {"Name": "conv1", "LayerType": "CaskConvolution",
         "Inputs": [{"Name": "x", "Format/Datatype": "Thirty-two wide channel vectorized row major Int8 format"}],
         "Outputs": [{"Name": "y", "Format/Datatype": "Thirty-two wide channel vectorized row major Int8 format"}]},
        {"Name": "head", "LayerType": "CaskConvolution",
         "Inputs": [{"Name": "y", "Format/Datatype": "Row major linear FP32"}],
         "Outputs": [{"Name": "z", "Format/Datatype": "Row major linear FP32"}]},
        {"Name": "reformat", "LayerType": "Reformat",
         "Inputs": [{"Name": "z", "Format/Datatype": "Int8"}], "Outputs": [{"Name": "w", "Format/Datatype": "Float"}]},
    ]
    t = D.layer_precision_table(layers)
    assert list(t["precision"]) == ["INT8", "FP32", "INT8"]
    s = D.precision_summary(t).set_index("precision")
    assert s.loc["INT8", "layers"] == 1 and s.loc["FP32", "layers"] == 1   # Reformat excluded
    assert s["share_pct"].sum() == pytest.approx(100)


def test_pad_to_batch_repeats_images_cyclically():
    x = np.arange(15)[:, None]
    p = pad_to_batch(x, 20)
    assert len(p) == 20 and list(p[15:, 0]) == [0, 1, 2, 3, 4]
    assert pad_to_batch(np.arange(40)[:, None], 20).shape == (40, 1)
