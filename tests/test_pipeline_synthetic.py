"""End-to-end check on a synthetic OULU-like dataset: inventory -> audit -> preprocessing
-> EDA -> (tiny) training -> threshold -> test evaluation, including resume behaviour."""

import json

import pandas as pd
import pytest

from conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def env(synthetic, tmp_path_factory, repo_root, monkeypatch_module):
    from qpad.config import ProjectPaths, load_config

    out = tmp_path_factory.mktemp("drive") / "QPAD_Project"
    work = tmp_path_factory.mktemp("content") / "qpad_work"
    monkeypatch_module.setenv("QPAD_DATASET_CONFIG", str(synthetic["config_path"]))
    monkeypatch_module.setenv("QPAD_PROJECT_DRIVE_ROOT", str(out))
    monkeypatch_module.setenv("QPAD_LOCAL_WORK_ROOT", str(work))
    cfgs = load_config(repo_root / "configs")
    cfgs["training"]["model"]["pretrained_imagenet"] = False   # no weight download in tests
    cfgs["training"]["data"]["num_workers"] = 0
    paths = ProjectPaths.from_config(cfgs["dataset"])
    paths.ensure_output_dirs()
    return {"paths": paths, "cfgs": cfgs}


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_inventory_discovers_layout(env):
    from qpad.pipeline import archive_inventory

    inv = archive_inventory(env["paths"], env["cfgs"]["dataset"])
    assert inv["archives"]["train"]["kinds"]["video"] == 20
    assert inv["archives"]["train"]["stem_pairing"]["videos_with_same_stem_annotation"] == 20
    assert sorted(inv["protocol_files"]) == ["dev", "test", "train"]
    assert (env["paths"].manifests / "archive_inventory.json").is_file()


def test_full_audit_passes_and_resumes(env):
    from qpad.pipeline import run_dataset_audit

    r = run_dataset_audit(env["paths"], env["cfgs"]["dataset"], dry_run=False, n_per_class=2)
    failed = [c.name for c in r["checks"] if c.status != "pass" and c.required]
    assert r["status"] == "PASS", failed
    m = r["manifest"]
    assert m["in_protocol"].sum() == 36 and (~m["in_protocol"]).sum() == 24
    assert set(m.loc[m.in_protocol, "ann_frame_base"]) == {"1"}
    names = {c.name: c.status for c in r["checks"]}
    assert names["official_protocol_files_agree"] == "pass"
    # Resume: a second run re-uses every stored record.
    store = pd.read_csv(env["paths"].manifests / "audit_progress.csv")
    assert len(store) == 36
    r2 = run_dataset_audit(env["paths"], env["cfgs"]["dataset"], dry_run=False, n_per_class=2)
    assert r2["status"] == "PASS"


def test_dry_run_audit_is_isolated(env):
    from qpad.pipeline import run_dataset_audit

    r = run_dataset_audit(env["paths"], env["cfgs"]["dataset"], dry_run=True, n_per_class=1)
    assert r["status"] == "INCOMPLETE"
    assert r["output_root"].endswith("_dry_run")
    assert json.loads((env["paths"].reports / "dataset_audit.json").read_text())["overall_status"] == "PASS"


def test_preprocessing_all_splits(env, repo_root):
    from qpad.pipeline import run_preprocessing

    for split in ("train", "dev", "test"):
        r = run_preprocessing(env["paths"], env["cfgs"], split, dry_run=False, n_per_class=2, repo_root=repo_root)
        assert r["verified"], r["problems"]
        assert r["n_failed"] == 0
    f = pd.read_csv(env["paths"].processed / "processed_manifest_train.csv")
    assert f.groupby("video_id")["is_calib"].sum().eq(5).all()
    assert f.groupby("video_id")["is_eval"].sum().eq(20).all()
    assert (f["outside_fraction"] < 0.5).all()
    # Resume: rerun skips all complete shards.
    r = run_preprocessing(env["paths"], env["cfgs"], "dev", dry_run=False, n_per_class=2, repo_root=repo_root)
    assert r["written"] == 0 and r["skipped_complete"] == r["n_shards"]


def test_saved_calibration_manifest_must_match(env):
    import copy

    from qpad.experiments import build_calibration_manifest

    cfgs = copy.deepcopy(env["cfgs"])
    cfgs["int8"]["calibration_budget"] = {"videos": 3, "frames_per_video": 5, "images": 15}
    cfgs["int8"]["conditions"] = {"natural": {"bona_fide": 1, "print": 1, "replay": 1}}
    man, _, path = build_calibration_manifest(env["paths"], cfgs, False, [1], ["natural"])
    assert len(man) == 15
    build_calibration_manifest(env["paths"], cfgs, False, [1], ["natural"])   # identical rerun is accepted
    edited = pd.read_csv(path, dtype={"video_id": str})
    edited.loc[0, "rank_in_class"] += 7                                         # same keys, altered selection
    edited.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="different selection"):
        build_calibration_manifest(env["paths"], cfgs, False, [1], ["natural"])
    path.unlink()


def test_dry_run_calibration_shrinks_mixtures(env):
    """Dry-run crops hold ~2 videos per class: the 40/40/40 mixtures must shrink instead of failing."""
    from qpad.experiments import build_calibration_manifest

    man, _, path = build_calibration_manifest(env["paths"], env["cfgs"], True, [1, 2], ["natural", "no_print"])
    counts = man.groupby(["seed", "condition", "class"])["video_id"].nunique().unstack(fill_value=0)
    assert counts.loc[(1, "natural")].tolist() == [1, 1, 1]
    assert counts.loc[(1, "no_print")].tolist() == [1, 0, 2]
    assert len(man) % int(env["cfgs"]["int8"]["calibration_budget"]["frames_per_video"]) == 0
    assert "_dry_run" in str(path)
    path.unlink()


def test_eda_outputs(env):
    from qpad.pipeline import run_eda

    r = run_eda(env["paths"], dry_run=False)
    assert not r["skipped"], r["skipped"]
    names = {pathlib_name(f["png"]) for f in r["figures"]}
    for n in ("class_balance_by_split", "brightness_vs_sharpness", "sharpness_by_class", "phone_distribution"):
        assert n in names
    caps = json.loads((env["paths"].results / "eda" / "captions.json").read_text())
    assert all(caps.values())


def pathlib_name(p):
    from pathlib import Path

    return Path(p).stem


def test_training_threshold_and_test_evaluation(env, repo_root):
    from qpad.pipeline import evaluate_fp32_test, run_training, select_fp32_threshold
    from qpad.training import BEST_NAME, freeze_checkpoint

    cfgs = env["cfgs"]
    cfgs["training"]["data"]["batch_size"] = 16
    r = run_training(env["paths"], cfgs, dry_run=False, device="cpu", repo_root=repo_root, max_batches=2,
                     epochs_override=2)
    assert len(r["history"]) == 2
    # Resume: asking for 3 epochs continues from epoch 2.
    r = run_training(env["paths"], cfgs, dry_run=False, device="cpu", repo_root=repo_root, max_batches=2,
                     epochs_override=3)
    assert r["history"]["epoch"].tolist() == [1, 2, 3]
    best = env["paths"].models / BEST_NAME
    freeze_checkpoint(best)
    with pytest.raises(RuntimeError):
        run_training(env["paths"], cfgs, dry_run=False, device="cpu", repo_root=repo_root, max_batches=1,
                     epochs_override=4)
    t = select_fp32_threshold(env["paths"], cfgs, dry_run=False, device="cpu", repo_root=repo_root)
    assert t["threshold"]["n_attacks"] == 8 and t["threshold"]["n_bona_fide"] == 4
    e = evaluate_fp32_test(env["paths"], cfgs, dry_run=False, device="cpu", repo_root=repo_root)
    m = e["metrics"]["test"]
    assert m["print_apcer"]["total"] == 4 and m["replay_apcer"]["total"] == 4 and m["bpcer"]["total"] == 4
    assert m["threshold"] == t["threshold"]["threshold"]
    # Changing the threshold after the test evaluation must be refused.
    thr = env["paths"].results / "fp32" / "threshold.json"
    d = json.loads(thr.read_text())
    d["threshold"] = 0.123
    thr.write_text(json.dumps(d))
    with pytest.raises(RuntimeError):
        evaluate_fp32_test(env["paths"], cfgs, dry_run=False, device="cpu", repo_root=repo_root)


def test_corrupt_dataset_fails_audit(tmp_path, repo_root, monkeypatch):
    from qpad.config import ProjectPaths, load_config
    from qpad.pipeline import run_dataset_audit
    from synthetic_oulu import build_synthetic_dataset

    syn = build_synthetic_dataset(tmp_path / "bad", corrupt="missing_annotation")
    monkeypatch.setenv("QPAD_DATASET_CONFIG", str(syn["config_path"]))
    monkeypatch.setenv("QPAD_PROJECT_DRIVE_ROOT", str(tmp_path / "drive"))
    monkeypatch.setenv("QPAD_LOCAL_WORK_ROOT", str(tmp_path / "work"))
    cfgs = load_config(repo_root / "configs")
    paths = ProjectPaths.from_config(cfgs["dataset"])
    r = run_dataset_audit(paths, cfgs["dataset"], dry_run=False, n_per_class=2)
    status = {c.name: c.status for c in r["checks"]}
    assert r["status"] == "FAIL"
    assert status["annotation_member_exists"] == "fail" and status["audit_errors"] == "fail"
