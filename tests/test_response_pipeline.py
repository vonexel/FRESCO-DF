import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
import pandas as pd

from fresco.reporting import build_report
from fresco.run import train_experiment
from fresco.synthetic import make_cache
from fresco.utils import atomic_json, digest_file


def test_augmented_recipe_trains_and_reports_as_exploratory(tmp_path):
    cache = make_cache(tmp_path / "cache")
    meta = pd.read_csv(cache / "metadata.csv", keep_default_na=False)
    features, quality = np.load(cache / "features.npy"), np.load(cache / "covariates.npy")
    tables, arrays, covariates = [meta], [features], [quality]
    source = (meta.dataset == "ffpp") & meta.split.isin(["train", "val"])
    for strength, condition in enumerate(["jpeg70", "blur1", "resize0.5"], start=1):
        rows = meta.loc[source].copy()
        rows["condition"] = condition
        tables.append(rows)
        arrays.append(features[source]+strength*.01)
        covariates.append(quality[source]+strength*.02)
    meta = pd.concat(tables, ignore_index=True)
    meta["row"] = np.arange(len(meta))
    meta.to_csv(cache / "metadata.csv", index=False)
    np.save(cache / "features.npy", np.concatenate(arrays))
    np.save(cache / "covariates.npy", np.concatenate(covariates))
    atomic_json(cache / "manifest.json", {"synthetic": True,
        **{name+"_sha256": digest_file(cache / filename) for name, filename in
           [("features", "features.npy"), ("covariates", "covariates.npy"), ("metadata", "metadata.csv")]}})
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1]/"configs")):
        cfg = compose(config_name="config", overrides=["experiment=response_augmented",
            "experiment.synthetic=true", "experiment.seeds=[0]", "experiment.budgets=[params]",
            "train.device=cpu", "train.epochs=1", "train.threads=1", "train.lrs=[0.001]",
            "train.max_steps=2", "train.validation_every_steps=1",
            "eval.bootstrap=10", "eval.interpretation_samples=8", "eval.grid_points=5",
            "eval.spline_support_samples=2", "+eval.benchmark_repeats=1", "+eval.benchmark_warmup=1"])
    cfg.paths.output_dir = str(tmp_path / "outputs")
    out = train_experiment(cfg, cache)
    build_report(out, cfg)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["primary_paired_response"]["status"] == "EXPLORATORY"
    assert stats["analysis_role"].startswith("exploratory")
    assert all(row["status"] == "EXPLORATORY" for row in stats["H2a"])
    assert "G_curve" in (out / "REPORT.md").read_text()
    summaries = pd.read_csv(out / "response_audit_summary.csv")
    assert set(summaries["head"]) == {"kan_spline", "mlp_gelu"}
    for path in out.rglob("checkpoint.pt"):
        import torch
        assert "coordinates_sha256" in torch.load(path, weights_only=True)
    for path in out.rglob("metrics.json"):
        detail = json.loads(path.read_text())
        assert detail["matching"]["target_params"] == 1585
        assert detail["training"]["training_conditions"] == ["clean", "jpeg70", "blur1", "resize0.5"]
        assert "response_audit" in detail["interp"]
