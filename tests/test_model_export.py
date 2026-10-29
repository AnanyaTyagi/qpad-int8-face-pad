import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.fixture(scope="module")
def model():
    from qpad.deeppixbis import DeepPixBiS

    torch.manual_seed(0)
    return DeepPixBiS(pretrained=False).eval()


def test_output_shapes(model):
    with torch.no_grad():
        m, b = model(torch.zeros(2, 3, 224, 224))
    assert m.shape == (2, 1, 14, 14) and b.shape == (2,)
    assert float(m.min()) >= 0 and float(m.max()) <= 1


def test_encoder_is_densenet161_first_eight_children(model):
    names = [type(c).__name__ for c in model.enc]
    assert names == ["Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", "_DenseBlock", "_Transition", "_DenseBlock",
                     "_Transition"]
    assert model.dec.in_channels == 384 and model.linear.in_features == 196


def test_loss_targets():
    from qpad.deeppixbis import DeepPixBiSLoss

    loss = DeepPixBiSLoss(0.5)
    perfect_map = torch.ones(2, 1, 14, 14) * torch.tensor([1.0, 0.0]).view(2, 1, 1, 1)
    total, lp, lb = loss(perfect_map.clamp(1e-6, 1 - 1e-6), torch.tensor([1 - 1e-6, 1e-6]), torch.tensor([1.0, 0.0]))
    assert float(total) < 1e-4


def test_frame_score_is_map_mean(model):
    from qpad.deeppixbis import ScoreHead, map_to_frame_score
    from qpad.scoring import grid_to_frame_scores, torch_backend

    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        m, _ = model(x)
        s = ScoreHead(model)(x)
    np.testing.assert_allclose(s.numpy(), map_to_frame_score(m).numpy())
    np.testing.assert_allclose(grid_to_frame_scores(torch_backend(model)(x.numpy())), s.numpy(), atol=1e-6)


def test_onnx_export_equivalence(model, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from qpad.export import export_onnx, onnxruntime_backend
    from qpad.scoring import grid_to_frame_scores, torch_backend

    info = export_onnx(model, tmp_path / "m.onnx")
    assert set(info["outputs"]) == {"pixel_map", "binary"} and info["inputs"]["input"][0] == "batch"
    x = np.random.default_rng(0).standard_normal((4, 3, 224, 224)).astype(np.float32)
    a = grid_to_frame_scores(torch_backend(model)(x))
    b = grid_to_frame_scores(onnxruntime_backend(tmp_path / "m.onnx")(x))
    assert np.abs(a - b).max() <= 1e-5


def test_compare_scores_reports_failures():
    import pandas as pd

    from qpad.export import compare_scores

    a = pd.DataFrame({"video_id": ["a", "b"], "frame_idx": [0, 0], "frame_score": [0.5, 0.6]})
    b = a.assign(frame_score=[0.5, 0.61])
    r = compare_scores(a, b, 1e-3)
    assert not r["passed"] and r["n_failing"] == 1 and r["worst_key"]["video_id"] == "b"
    with pytest.raises(ValueError):
        compare_scores(a, b.iloc[:1], 1e-3)
