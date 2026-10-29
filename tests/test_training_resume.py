import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset

from qpad.datasets import seeded_loader
from qpad.training import fit

CFG = {"runtime": {"amp": False}, "schedule": {"epochs": 3, "early_stopping_patience": 99},
       "loss": {"beta": 0.5}, "optimizer": {"name": "adam", "lr": 1e-2, "weight_decay": 0.0}}


class TinyPixBiS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(14)
        self.linear = nn.Linear(196, 1)

    def forward(self, x):
        m = torch.sigmoid(self.pool(self.conv(x)))
        return m, torch.sigmoid(self.linear(m.flatten(1))).flatten()


def _data():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 3, 16, 16, generator=g)
    y = (torch.arange(40) % 2).long()
    ds = TensorDataset(x, y, torch.arange(40))
    index = pd.DataFrame({"video_id": [f"v{i // 4}" for i in range(40)], "label": y.numpy()})
    return ds, index


def _run(tmp, epochs):
    ds, index = _data()
    torch.manual_seed(1)
    model = TinyPixBiS()
    fit(model, seeded_loader(ds, 8, True, 42, 0), seeded_loader(ds, 8, False, 42, 0), index, CFG,
        tmp / "local", tmp / "drive", "cpu", epochs_override=epochs)
    return model


def test_resumed_training_matches_uninterrupted(tmp_path):
    full = _run(tmp_path / "a", 3)
    _run(tmp_path / "b", 1)                # "disconnect" after epoch 1
    resumed = _run(tmp_path / "b", 3)      # new loader from the same seed, resumes from the last checkpoint
    for (k, a), (_, b) in zip(full.state_dict().items(), resumed.state_dict().items()):
        np.testing.assert_array_equal(a.numpy(), b.numpy(), err_msg=k)
