"""Hydra entry point: preprocess, train, evaluate, aggregate, and resume."""
from __future__ import annotations
import itertools
import json
import traceback
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from .provenance import read_cache_provenance
from .training import Coordinates, build_model, fit_head
from .utils import (atomic_json, complete, digest_file, environment, fingerprint,
                    mark_complete, seed_everything, source_fingerprint)


def load_cache(path: Path):
    manifest_path=path/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in ["features","covariates","metadata"]:
            expected=manifest.get(name+"_sha256")
            suffix=".csv" if name=="metadata" else ".npy"
            if expected and digest_file(path/(name+suffix))!=expected:
                raise ValueError(f"Corrupted {name} cache; resume the feature stage to repair it")
    meta = pd.read_csv(path / "metadata.csv", keep_default_na=False)
    feat = np.load(path / "features.npy", mmap_mode="r")
    cov = np.load(path / "covariates.npy", mmap_mode="r")
    if feat.ndim != 2 or cov.ndim != 2 or feat.shape[1] < 1 or cov.shape[1] != 4:
        raise ValueError("Features must be [rows,dimensions] and covariates [rows,4]")
    if not (len(meta) == len(feat) == len(cov)):
        raise ValueError("Feature/cache metadata dimensions disagree")
    if meta.duplicated(["dataset", "video_id", "frame_idx", "condition"]).any():
        raise ValueError("Duplicate cached frame-condition keys")
    if "row" in meta and not np.array_equal(meta.row, np.arange(len(meta))):
        raise ValueError("metadata.row does not match feature array ordering")
    if ((meta.dataset != "ffpp") & (meta.split != "test")).any():
        raise ValueError("Target dataset must be TEST ONLY")
    for split in ["train", "val", "test"]:
        ids = set(meta.loc[(meta.dataset == "ffpp") & (meta.split == split), "group_id"])
        for other in ["train", "val", "test"]:
            if other != split and ids & set(meta.loc[(meta.dataset == "ffpp") & (meta.split == other), "group_id"]):
                raise ValueError(f"Source group leakage between {split} and {other}")
    for array in [feat,cov]:
        if any(not np.isfinite(array[i:i+16384]).all() for i in range(0,len(array),16384)):
            raise ValueError("Nonfinite cached features/covariates")
    return feat, cov, meta


def train_experiment(cfg, cache: Path) -> Path:
    from .evaluation import evaluate_run
    cache_provenance = read_cache_provenance(cache, synthetic=bool(cfg.experiment.synthetic))
    features, covariates, meta = load_cache(cache)
    if cfg.experiment.get("conditions"):
        keep=meta.condition.isin(list(cfg.experiment.conditions)).to_numpy()
        features,covariates,meta=features[keep],covariates[keep],meta.loc[keep].reset_index(drop=True)
        meta["row"]=np.arange(len(meta))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    identity = {"config": resolved, "source": source_fingerprint(), "runtime":environment(),
                "dependency_files":{n:digest_file(Path.cwd()/n) for n in ["uv.lock","pyproject.toml"]
                                    if (Path.cwd()/n).is_file()},
                "cache": {n: digest_file(cache/n) for n in ["features.npy", "covariates.npy", "metadata.csv"]}}
    key = fingerprint(identity)
    out = Path(cfg.paths.output_dir) / cfg.experiment.name / key
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    atomic_json(out / "manifest.json", {**identity, "environment": environment(),
                "synthetic": bool(cfg.experiment.synthetic), "cache_dir": str(cache.resolve()),
                "cache_provenance": cache_provenance,
                "video_counts": meta.drop_duplicates("video_id").groupby(["dataset", "split", "method"]).size().reset_index(name="videos").to_dict("records")})
    atomic_json(out.parent / "LATEST.json", {"path": str(out.resolve()), "key": key})
    combinations = list(itertools.product(cfg.experiment.ks, cfg.experiment.covariates,
                                          cfg.experiment.held_out_methods))
    for k, cov, held_out in tqdm(combinations, desc="Feature/fold configurations"):
        coords = Coordinates().fit(features, covariates, meta, int(k), bool(cov), held_out,
            covariate_indices=cfg.train.get("covariate_indices"),
            covariate_scaling=str(cfg.train.get("covariate_scaling", "standard")),
            scaling_conditions=cfg.train.get("scaling_conditions"),
            range_quantile=float(cfg.train.get("range_quantile", .995)),
            range_limit=float(cfg.train.get("range_limit", 2.5)))
        x = coords.transform(features, covariates)
        common = out / f"k{k}_cov{int(cov)}" / f"heldout_{held_out or 'none'}"
        common.mkdir(parents=True, exist_ok=True)
        coords.save(common / "coordinates.npz")
        atomic_json(common / "fit_provenance.json", {"dataset": "ffpp", "split": "train", "condition": "clean",
                     "held_out_method": held_out, "video_ids": coords.train_videos,
                     "head_training_conditions": list(cfg.train.get("conditions") or ["clean"]),
                     "coordinate_fit_conditions": ["clean"],
                     "covariate_indices": coords.covariate_indices.tolist(),
                     "covariate_scaling": coords.covariate_scaling,
                     "covariate_scaling_fit_conditions": coords.scaling_conditions.tolist(),
                     "coordinates": ("native CLS coordinates, train-standardized; no compression" if int(k)==features.shape[1]
                                     else "frozen train-only PCA on L2-normalized CLS; shared across seeds and heads")})
        for head, budget in tqdm(list(itertools.product(cfg.experiment.heads, cfg.experiment.budgets)),
                                 desc="Head/budget", leave=False):
            hpo_dir = common / head / budget
            hpo_dir.mkdir(parents=True, exist_ok=True)
            hpo_key = fingerprint({"key": key, "head": head, "budget": budget, "k": k, "cov": cov,
                                   "heldout": held_out, "stage": "hpo"})
            try:
                if not cfg.force and complete(hpo_dir, hpo_key):
                    hpo = json.loads((hpo_dir / "hpo.json").read_text(encoding="utf-8"))
                else:
                    trials = []
                    for lr in cfg.train.lrs:
                        seed_everything(int(cfg.experiment.seeds[0]))
                        model, matching = build_model(head, x.shape[1], budget, int(cfg.train.target_hidden),
                                                      cfg.get("model"), target_params=cfg.train.get("target_params"))
                        model, _, log = fit_head(model, x, meta, cfg.train, float(lr), int(cfg.experiment.seeds[0]),
                                                budget, held_out, progress=False)
                        trials.append({"lr": float(lr), "validation_auc": log["validation_auc"],
                                       "wall_s": log["wall_s"], "matching": matching})
                    best = max(trials, key=lambda t: t["validation_auc"])
                    hpo = {"best_lr": best["lr"], "selection": "FF++ seen-method validation video AUC only",
                           "selection_seed": int(cfg.experiment.seeds[0]), "trials": trials}
                    atomic_json(hpo_dir / "hpo.json", hpo)
                    mark_complete(hpo_dir, hpo_key, ["hpo.json"])
                for seed in cfg.experiment.seeds:
                    run_dir = hpo_dir / f"seed{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    run_key = fingerprint({"hpo_key": hpo_key, "lr": hpo["best_lr"], "seed": seed})
                    if not cfg.force and complete(run_dir, run_key):
                        print(f"SKIP {run_dir}")
                        continue
                    seed_everything(int(seed))
                    model, matching = build_model(head, x.shape[1], budget, int(cfg.train.target_hidden),
                                                  cfg.get("model"), target_params=cfg.train.get("target_params"))
                    model, initial_state, training = fit_head(model, x, meta, cfg.train, hpo["best_lr"],
                                                             int(seed), budget, held_out)
                    torch.save({"state": model.state_dict(), "initial_state": initial_state,
                                "head": head, "in_features": x.shape[1], "matching": matching,
                                "coordinates_sha256": digest_file(common / "coordinates.npz")},
                               run_dir / "checkpoint.pt")
                    pd.DataFrame(training.pop("log")).to_csv(run_dir / "train_log.csv", index=False)
                    result = evaluate_run(model, initial_state, x, meta, coords, cfg, run_dir, held_out)
                    result.update({"run": {"head": head, "budget": budget, "k": int(k), "covariates": bool(cov),
                                           "held_out": held_out, "seed": int(seed), "lr": hpo["best_lr"],
                                           "synthetic": bool(cfg.experiment.synthetic)},
                                   "matching": matching, "training": training, "environment": environment()})
                    atomic_json(run_dir / "metrics.json", result)
                    names = [p.name for p in run_dir.iterdir() if p.is_file() and p.name != "DONE.json"]
                    mark_complete(run_dir, run_key, names)
            except Exception as exc:
                atomic_json(hpo_dir / "FAILED.json", {"error": str(exc), "traceback": traceback.format_exc()})
                raise
    return out


def execute(cfg: DictConfig):
    if cfg.experiment.name == "functions":
        from .controlled import run_controlled
        return run_controlled(cfg)
    from .data import index_datasets
    from .preprocess import extract_faces
    from .features import extract_features
    data = Path(cfg.paths.data_dir)
    if cfg.experiment.synthetic:
        from .synthetic import make_cache
        # Integration fixture runs no downloads and never reads face videos.
        cfg.train.device, cfg.train.epochs, cfg.train.patience = "cpu", 2, 2
        cfg.train.wall_seconds, cfg.train.target_hidden = .12, 16
        cfg.train.lrs = [.001]
        cfg.eval.bootstrap = min(int(cfg.eval.bootstrap), 100)
        cfg.eval.interpretation_samples, cfg.eval.grid_points = 32, 11
        cache = make_cache(data / "synthetic_cache")
    else:
        index = data / "index" / "index.csv"
        faces = data / "faces" / "faces.csv"
        cache = Path(cfg.paths.feature_cache)
        if "index" in cfg.stages:
            index = index_datasets(Path(cfg.paths.ffpp_root), Path(cfg.paths.celeb_root),
                                   data / "splits", data / "index")
        if "faces" in cfg.stages:
            faces = extract_faces(index, data / "faces", frames=int(cfg.preprocess.frames),
                    workers=int(cfg.preprocess.workers), detector=str(cfg.preprocess.detector),
                    limit=cfg.preprocess.limit)
        if "features" in cfg.stages:
            result = extract_features(faces, cache, device=str(cfg.features.device),
                      batch_size=int(cfg.features.batch_size), conditions=list(cfg.features.conditions),
                      encoder_name=str(cfg.features.encoder_name), pretrained=str(cfg.features.pretrained),
                      recover_provenance=bool(cfg.features.get("recover_provenance", False)),
                      source_conditions=cfg.features.get("source_conditions"))
            cache = Path(result)
    if "train" in cfg.stages:
        out = train_experiment(cfg, cache)
    else:
        latest = Path(cfg.paths.output_dir) / cfg.experiment.name / "LATEST.json"
        out = Path(json.loads(latest.read_text(encoding="utf-8"))["path"]) if latest.exists() else None
    if "report" in cfg.stages:
        if out is None:
            raise FileNotFoundError("No completed training experiment found; run stages=[train,report]")
        from .reporting import build_report
        build_report(out, cfg)
        print(f"REPORT: {(out/'REPORT.md').resolve()}")
    return out


_config_root=Path(__file__).resolve().parent/"configs"
if not _config_root.is_dir():
    _config_root=Path(__file__).resolve().parents[2]/"configs"


@hydra.main(version_base="1.3", config_path=str(_config_root), config_name="config")
def main(cfg: DictConfig):
    execute(cfg)


if __name__ == "__main__":
    main()

