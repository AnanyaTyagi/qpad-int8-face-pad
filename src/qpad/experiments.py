"""Priority 1/2 stages: export equivalence, INT8 calibration engines, rebuild noise,
statistics and latency. Every function reads/writes files under the project
root so stages can be resumed and audited.

Weights are never modified here: every engine is built from the ONNX export of
the frozen FP32 checkpoint (hash-checked).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .datasets import FrameStore, materialize, video_index
from .metrics import auroc, decision_flips, pad_metrics
from .pipeline import find_input, output_paths, processed_split_dir
from .reproducibility import file_sha256, write_json
from .scoring import aggregate_video_scores, score_store
from .thresholds import predict_bona_fide, select_eer_threshold

ONNX_NAME = "deeppixbis_fp32.onnx"


class EquivalenceError(RuntimeError):
    """Raised when export tolerances fail; INT8 work must not continue."""


def subset_store(store: FrameStore, n_videos: int | None) -> FrameStore:
    """First ``n_videos`` videos (sorted IDs) of a store, all their frames; ``None`` = all."""
    if n_videos is None:
        return store
    keep = sorted(store.index["video_id"].unique())[:n_videos]
    mask = store.index["video_id"].isin(keep).to_numpy()
    return FrameStore(np.asarray(store.images[mask]), store.index[mask].reset_index(drop=True))


def balanced_subset(store: FrameStore, n_per_class: int | None) -> FrameStore:
    """First ``n_per_class`` videos of each class (deterministic) — used only in DRY_RUN."""
    if n_per_class is None:
        return store
    vids = video_index(store).sort_values("video_id").groupby("class").head(n_per_class)["video_id"]
    mask = store.index["video_id"].isin(set(vids)).to_numpy()
    return FrameStore(np.asarray(store.images[mask]), store.index[mask].reset_index(drop=True))


def disable_torch_tf32() -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ---------------------------------------------------------------------------------------
# Notebook 06: export validation
# ---------------------------------------------------------------------------------------
def export_validation(paths: ProjectPaths, cfgs: Mapping[str, Any], dry_run: bool, device: str,
                      n_videos: int | None, run_tensorrt: bool, progress: Callable | None = None) -> dict[str, Any]:
    """PyTorch -> ONNX -> TensorRT FP32 (TF32 off) on the same saved development frames."""
    from .export import compare_scores, export_onnx, onnxruntime_backend
    from .scoring import torch_backend
    from .training import BEST_NAME, load_frozen_model

    out = output_paths(paths, dry_run)
    res = out.results / "export"
    res.mkdir(parents=True, exist_ok=True)
    disable_torch_tf32()
    ck = find_input(paths, dry_run, "models", BEST_NAME)
    model, ck_hash = load_frozen_model(ck, device)
    onnx_path = out.models / ONNX_NAME
    info = export_onnx(model, onnx_path, int(cfgs["int8"]["tensorrt"]["onnx_opset"]),
                       int(cfgs["preprocessing"]["output_size"]))
    info.update(sha256=file_sha256(onnx_path), checkpoint_sha256=ck_hash)
    norm = cfgs["preprocessing"]["normalization"]
    store = subset_store(materialize(processed_split_dir(paths, dry_run, "dev"), "dev", "eval", out.local_cache), n_videos)
    tol = cfgs["int8"]["export_tolerances"]
    s_pt = score_store(store, torch_backend(model, device), norm, 32, progress)
    s_onnx = score_store(store, onnxruntime_backend(onnx_path), norm, 32, progress)
    report: dict[str, Any] = {"onnx": info, "n_frames": len(store), "n_videos": int(store.index["video_id"].nunique()),
                              "pytorch_vs_onnx": compare_scores(s_pt, s_onnx, float(tol["pytorch_vs_onnx_max_abs"]),
                                                                label_a="pt", label_b="onnx")}
    s_pt.to_csv(res / "pytorch_frame_scores.csv", index=False)
    s_onnx.to_csv(res / "onnx_frame_scores.csv", index=False)
    if run_tensorrt:
        from .tensorrt_utils import TrtRunner, build_engine

        eng = out.engines / "fp32_tf32off.engine"
        meta = build_engine(onnx_path, eng, "fp32", tf32=False, trt_cfg=cfgs["int8"]["tensorrt"])
        s_trt = score_store(store, TrtRunner(eng), norm, int(cfgs["int8"]["tensorrt"]["opt_batch"]), progress)
        s_trt.to_csv(res / "trt_fp32_frame_scores.csv", index=False)
        report["trt_fp32_engine"] = meta
        report["onnx_vs_trt_fp32"] = compare_scores(s_onnx, s_trt, float(tol["onnx_vs_trt_fp32_max_abs"]),
                                                    label_a="onnx", label_b="trt")
        v = [aggregate_video_scores(s) for s in (s_pt, s_trt)]
        report["video_max_abs_pt_vs_trt"] = float(np.abs(v[0]["score"] - v[1]["score"]).max())
    else:
        report["onnx_vs_trt_fp32"] = {"passed": None, "note": "TensorRT not run"}
    report["passed"] = bool(report["pytorch_vs_onnx"]["passed"] and report["onnx_vs_trt_fp32"].get("passed"))
    write_json(res / "equivalence.json", report)
    failing = pd.DataFrame(report["pytorch_vs_onnx"]["failing_examples"] +
                           report["onnx_vs_trt_fp32"].get("failing_examples", []))
    failing.to_csv(res / "failing_samples.csv", index=False)
    return report


# ---------------------------------------------------------------------------------------
# Notebooks 07/08: calibration manifests and INT8 engines
# ---------------------------------------------------------------------------------------
def engine_tag(method: str, condition: str, seed: int, repeat: int | None = None) -> str:
    return f"int8_{method}_{condition}_s{seed}" + (f"_rep{repeat}" if repeat is not None else "")


def build_calibration_manifest(paths: ProjectPaths, cfgs: Mapping[str, Any], dry_run: bool, seeds: Sequence[int],
                               conditions: Sequence[str]) -> tuple[pd.DataFrame, FrameStore, Path]:
    """Nested calibration selection with saved frame indices (written BEFORE any engine is built)."""
    from .calibration import attach_calibration_frames, nested_calibration_sets

    icfg = cfgs["int8"]
    out = output_paths(paths, dry_run)
    store = materialize(processed_split_dir(paths, dry_run, "train"), "train", "calib", out.local_cache)
    if set(store.index["split"]) != {"train"}:
        raise RuntimeError("calibration frames must come from the training split only")
    vids = video_index(store)
    conds = {c: icfg["conditions"][c] for c in conditions}
    fpv = int(icfg["calibration_budget"]["frames_per_video"])
    if dry_run:
        # The dry-run crops hold only a few videos per class: shrink every mixture (each class count / 40,
        # rounded up) so the plumbing runs end to end. These engines are meaningless by construction.
        conds = {name: {c: -(-int(k) // 40) for c, k in comp.items()} for name, comp in conds.items()}
        sel = pd.concat([nested_calibration_sets(vids, {name: comp}, seeds, sum(comp.values()))
                         for name, comp in conds.items()], ignore_index=True)
        man = attach_calibration_frames(sel, store.index, fpv)
    else:
        sel = nested_calibration_sets(vids, conds, seeds, int(icfg["calibration_budget"]["videos"]))
        man = attach_calibration_frames(sel, store.index, fpv)
        if (man.groupby(["seed", "condition"]).size() != int(icfg["calibration_budget"]["images"])).any():
            raise RuntimeError("calibration image budget violated")
    path = out.results / "int8" / "calibration_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        old = pd.read_csv(path, dtype={"video_id": str})
        order = ["seed", "condition", "class", "rank_in_class", "video_id", "frame_idx"]
        if not set(order) <= set(old.columns):
            raise RuntimeError(f"{path} lacks columns {sorted(set(order) - set(old.columns))}")
        key = ["seed", "condition"]
        saved = old.merge(man[key].drop_duplicates(), on=key)
        fresh = man.merge(saved[key].drop_duplicates(), on=key)

        def canon(df: pd.DataFrame) -> pd.DataFrame:
            return df[order].astype(str).sort_values(order).reset_index(drop=True)

        # Every (seed, condition) already saved must match the canonical selection row for row.
        if not canon(saved).equals(canon(fresh)):
            raise RuntimeError(f"{path} exists with a different selection; refusing to overwrite a saved manifest")
        # keep saved (seed, condition) sets from earlier runs (e.g. the pilot) and add new ones
        merged = pd.concat([old, man]).drop_duplicates(["seed", "condition", "video_id", "frame_idx"])
        man = merged.sort_values(["seed", "condition", "class", "rank_in_class", "frame_idx"]).reset_index(drop=True)
    man.to_csv(path, index=False)
    return man, store, path


def calibration_array(store: FrameStore, manifest: pd.DataFrame, seed: int, condition: str,
                      norm_cfg: Mapping[str, Any]) -> np.ndarray:
    """Normalized float32 NCHW calibration images in manifest order."""
    from .preprocessing import to_model_input

    rows = manifest[(manifest["seed"] == seed) & (manifest["condition"] == condition)]
    pos = {(v, int(f)): i for i, (v, f) in enumerate(zip(store.index["video_id"], store.index["frame_idx"]))}
    idx = [pos[(v, int(f))] for v, f in zip(rows["video_id"], rows["frame_idx"])]
    return to_model_input(np.asarray(store.images[idx]), norm_cfg)


def build_int8_engine(paths: ProjectPaths, cfgs: Mapping[str, Any], dry_run: bool, images: np.ndarray, tag: str,
                      method: str) -> dict[str, Any]:
    """Build one INT8 engine from the frozen ONNX graph and record its per-tensor scales."""
    from .calibration import make_calibrator, parse_calibration_cache
    from .tensorrt_utils import build_engine

    out = output_paths(paths, dry_run)
    onnx_path = out.models / ONNX_NAME
    eq = json.loads((out.results / "export" / "equivalence.json").read_text())
    if not eq.get("passed"):
        raise EquivalenceError("export equivalence has not passed; INT8 engines must not be built")
    if file_sha256(onnx_path) != eq["onnx"]["sha256"]:
        raise RuntimeError("ONNX file changed since equivalence validation")
    edir = out.engines / tag
    edir.mkdir(parents=True, exist_ok=True)
    # Remove completion markers first, so a failed (re)build can never leave an older engine that
    # engine_is_complete() would accept.
    for stale in ("engine_meta.json", "engine.plan", "scales.csv"):
        (edir / stale).unlink(missing_ok=True)
    (out.results / "int8" / tag / "metrics.json").unlink(missing_ok=True)
    cache = edir / "calibration.cache"
    cache.unlink(missing_ok=True)
    batch = int(cfgs["int8"]["calibration_method"]["calibration_batch_size"])
    if dry_run:   # the shrunken dry-run sets are a multiple of frames_per_video, not of the real batch
        batch = int(cfgs["int8"]["calibration_budget"]["frames_per_video"])
    calib = make_calibrator(images, batch, cache, method)
    meta = build_engine(onnx_path, edir / "engine.plan", "int8", tf32=False, trt_cfg=cfgs["int8"]["tensorrt"],
                        calibrator=calib, log_path=edir / "build_log.json")
    meta.update(tag=tag, method=method, n_calibration_images=int(len(images)),
                calibration_batches_served=int(calib.batches_served), onnx_sha256=eq["onnx"]["sha256"],
                checkpoint_sha256=eq["onnx"]["checkpoint_sha256"])
    scales = parse_calibration_cache(cache.read_bytes()) if cache.is_file() else {}
    pd.DataFrame({"tensor": list(scales), "scale": list(scales.values())}).to_csv(edir / "scales.csv", index=False)
    meta["n_scaled_tensors"] = len(scales)
    write_json(edir / "engine_meta.json", meta)
    return meta


def evaluate_backend(backend, dev: FrameStore, test: FrameStore, norm_cfg: Mapping[str, Any], fp32_threshold: float,
                     out_dir: Path, n_frames: "int | Mapping[str, int]", batch: int,
                     progress: Callable | None = None) -> dict[str, Any]:
    """Score dev+test, select the engine-specific dev threshold, evaluate both threshold policies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").unlink(missing_ok=True)   # completion marker: written last
    res: dict[str, Any] = {}
    vids = {}
    for name, store in (("dev", dev), ("test", test)):
        f = score_store(store, backend, norm_cfg, batch, progress)
        v = aggregate_video_scores(f, n_frames)
        f.to_csv(out_dir / f"{name}_frame_scores.csv", index=False)
        v.to_csv(out_dir / f"{name}_video_scores.csv", index=False)
        vids[name] = v
    thr_b, table = select_eer_threshold(vids["dev"]["score"].to_numpy(), (vids["dev"]["class"] == "bona_fide").to_numpy())
    table.to_csv(out_dir / "dev_threshold_candidates.csv", index=False)
    res["threshold_A_fp32"] = float(fp32_threshold)
    res["threshold_B_engine"] = thr_b.as_dict()
    res["policy_A"] = pad_metrics(vids["test"], fp32_threshold)
    res["policy_B"] = pad_metrics(vids["test"], thr_b.threshold)
    write_json(out_dir / "metrics.json", res)
    return res


def engine_is_complete(engine_dir: Path, result_dir: Path) -> bool:
    """Done only if the engine hash matches its metadata and metrics exist."""
    meta, plan = engine_dir / "engine_meta.json", engine_dir / "engine.plan"
    if not (meta.is_file() and plan.is_file() and (result_dir / "metrics.json").is_file()):
        return False
    return json.loads(meta.read_text())["sha256"] == file_sha256(plan)


# ---------------------------------------------------------------------------------------
# Rebuild noise
# ---------------------------------------------------------------------------------------
def rebuild_noise(result_dirs: Sequence[Path], engine_dirs: Sequence[Path], max_pp: float) -> dict[str, Any]:
    """Score/decision/metric movement across identical rebuilds of one engine."""
    vids = [pd.read_csv(d / "test_video_scores.csv", dtype={"video_id": str}).set_index("video_id").sort_index()
            for d in result_dirs]
    mets = [json.loads((d / "metrics.json").read_text()) for d in result_dirs]
    score_mat = np.stack([v["score"].to_numpy() for v in vids])
    thr = [m["threshold_B_engine"]["threshold"] for m in mets]
    dec = np.stack([predict_bona_fide(v["score"].to_numpy(), t) for v, t in zip(vids, thr)])
    keys = ("print_apcer", "replay_apcer", "apcer_pooled", "bpcer")
    movement = {k: 100 * (max(m["policy_B"][k]["rate"] for m in mets) - min(m["policy_B"][k]["rate"] for m in mets))
                for k in keys}
    movement["acer"] = 100 * (max(m["policy_B"]["acer"] for m in mets) - min(m["policy_B"]["acer"] for m in mets))
    hashes = [json.loads((e / "engine_meta.json").read_text())["sha256"] for e in engine_dirs]
    return {"n_builds": len(result_dirs), "max_score_movement": float((score_mat.max(0) - score_mat.min(0)).max()),
            "videos_with_decision_flip": int((dec.max(0) != dec.min(0)).sum()), "metric_movement_pp": movement,
            "exceeds_limit": bool(max(movement.values()) > max_pp), "limit_pp": max_pp, "engine_sha256": hashes,
            "identical_engines": len(set(hashes)) == 1, "thresholds": thr}


# ---------------------------------------------------------------------------------------
# Notebook 09: statistics
# ---------------------------------------------------------------------------------------
def load_decisions(result_root: Path, tags: Mapping[tuple[str, int], str], policy: str) -> pd.DataFrame:
    """Long-form test decisions for every (condition, seed) engine under policy ``A`` or ``B``."""
    rows = []
    for (cond, seed), tag in tags.items():
        d = result_root / tag
        v = pd.read_csv(d / "test_video_scores.csv", dtype={"video_id": str})
        m = json.loads((d / "metrics.json").read_text())
        t = m["threshold_A_fp32"] if policy == "A" else m["threshold_B_engine"]["threshold"]
        rows.append(v.assign(condition=cond, seed=seed, accepted=predict_bona_fide(v["score"].to_numpy(), t)))
    return pd.concat(rows, ignore_index=True)


def analyze_conditions(result_root: Path, tags: Mapping[tuple[str, int], str], reference: str,
                       stats_cfg: Mapping[str, Any], rebuild_noise_pp: float) -> dict[str, Any]:
    """Seed effects, clustered CIs, permutation p-values, Holm and rule-based labels."""
    from .bootstrap import cluster_bootstrap, paired_counts, seed_effects
    from .decision_rules import classify_effect
    from .permutation import holm_correction, permutation_test

    eff = float(stats_cfg["meaningful_effect_pp"]) / 100
    dec = {p: load_decisions(result_root, tags, p) for p in ("A", "B")}
    conds = sorted({c for c, _ in tags if c != reference})
    rows = []
    for cond in conds:
        for attack in ("print", "replay"):
            out = {"condition": cond, "attack": attack}
            for p in ("A", "B"):
                d = dec[p]
                pair = pd.concat([d[d.condition == cond].assign(model="unusual"),
                                  d[d.condition == reference].assign(model="natural")])
                pair = pair[pair["seed"].isin(set(d.loc[d.condition == cond, "seed"]))]
                pc = paired_counts(pair, attack, error_when_accepted=True)
                se = seed_effects(pc, eff)
                ci = cluster_bootstrap(pc, int(stats_cfg["bootstrap_reps"]), int(stats_cfg["bootstrap_seed"]),
                                       float(stats_cfg["ci_level"]))
                pt = permutation_test(pc, stats_cfg["permutation_alternative"],
                                      int(stats_cfg["permutation_max_exact_subjects"]),
                                      int(stats_cfg["permutation_monte_carlo"]), int(stats_cfg["bootstrap_seed"]))
                out[p] = {**se, **ci, **pt}
            aucs = []
            for (c, s), tag in tags.items():
                if c == cond:
                    u = pd.read_csv(result_root / tag / "test_video_scores.csv")
                    n = pd.read_csv(result_root / tags[(reference, s)] / "test_video_scores.csv")
                    keep_u = u["class"].isin(["bona_fide", attack])
                    keep_n = n["class"].isin(["bona_fide", attack])
                    aucs.append(auroc(u.loc[keep_u, "score"], u.loc[keep_u, "class"] == "bona_fide")
                                - auroc(n.loc[keep_n, "score"], n.loc[keep_n, "class"] == "bona_fide"))
            out["auroc_diff_mean"] = float(np.mean(aucs))
            rows.append(out)
    holm = holm_correction([r["B"]["p_value"] for r in rows])
    for r, h in zip(rows, holm):
        r["holm_p_B"] = float(h)
        r["verdict"] = classify_effect(r["B"], r["A"], float(h), float(stats_cfg["alpha"]), rebuild_noise_pp / 100,
                                       r["auroc_diff_mean"], stats_cfg)
    return {"comparisons": rows, "n_primary_comparisons": len(rows)}


def comparisons_table(analysis: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for r in analysis["comparisons"]:
        b = r["B"]
        rows.append({"condition": r["condition"], "attack": r["attack"],
                     "mean_diff_pp_B": 100 * b["mean"], "median_pp_B": 100 * b["median"],
                     "min_pp_B": 100 * b["min"], "max_pp_B": 100 * b["max"],
                     "seeds_ge_effect_B": b["n_exceeding"], "ci_low_pp_B": 100 * b["ci_low"],
                     "ci_high_pp_B": 100 * b["ci_high"], "p_B": b["p_value"], "holm_p_B": r["holm_p_B"],
                     "mean_diff_pp_A": 100 * r["A"]["mean"], "auroc_diff": r["auroc_diff_mean"],
                     "per_seed_pp_B": ";".join(f"{100 * x:.2f}" for x in b["per_seed_diff"]),
                     "verdict": r["verdict"]["label"]})
    return pd.DataFrame(rows)


def flips_vs(ref_dir: Path, other_dir: Path, policy: str = "B") -> dict[str, int]:
    """Decision flips of ``other`` relative to ``ref`` on identical test videos."""
    def load(d: Path):
        m = json.loads((d / "metrics.json").read_text())
        if "threshold_B_engine" in m:          # an INT8 engine result directory
            thr = m["threshold_A_fp32"] if policy == "A" else m["threshold_B_engine"]["threshold"]
        else:                                  # the FP32 results directory
            thr = json.loads((d / "threshold.json").read_text())["threshold"]
        return pd.read_csv(d / "test_video_scores.csv", dtype={"video_id": str}), thr
    (a, ta), (b, tb) = load(ref_dir), load(other_dir)
    return decision_flips(a, b, ta, tb)
