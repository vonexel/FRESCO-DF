"""Report-only regressions for the clean, uncompressed feature reference.

These fixtures contain no CLIP features or trained models. They reproduce the
completed report-consumed artifact schema and label every fixture synthetic.
"""
import itertools
import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from fresco.evaluation import COVARIATE_NAMES, VIDEO_KEYS
from fresco.reporting import pareto_table
from fresco.run import execute
from fresco.statistics import metrics, reliability
from fresco.utils import atomic_json, complete, digest_file, mark_complete


INTERVENTION_COLUMNS = [
    "dataset", "condition", "video_id", "label", "method", "group_id", "frames", "frame_count", "total",
    "covariate", "embedding", "interaction", "covariate_residual_rmse", "covariate_relative_l2",
    "embedding_residual_rmse", "embedding_relative_l2", "decomposition_residual_max", "total_rms",
    "covariate_train_support_fraction", "covariate_outside_support_fraction", "error_sse", "total_sse",
    "embedding_error_sse", "paired_response_agreement_gap",
]
PD_COLUMNS = ["feature", "feature_index", "grid_type", "grid", "response", "raw_response", "q05", "q95",
              "background_rows", "within_train_support"]


def run_record(head="linear", budget="params", covariates=False, seed=0):
    dimension = 1024 + 4 * covariates
    return {"head": head, "budget": budget, "k": 1024, "covariates": covariates, "held_out": None,
            "seed": seed, "synthetic": True, "lr": .001,
            "run_dir": f"k1024_cov{int(covariates)}/heldout_none/{head}/{budget}/seed{seed}",
            "params": dimension + 1 if head == "linear" else (dimension + 1) * 2 + 3,
            "head_latency_ms": .1 if head == "linear" else .2, "train_wall_s": 1.}


def pareto_row(run, condition="clean", auc=.8):
    return {"run_dir": run["run_dir"], "dataset": "celebdfpp", "level": "video", "method": "all",
            "calibration": "raw", "condition": condition, "auc": auc}


def write_intervention(directory, gap=None):
    directory.mkdir(parents=True, exist_ok=True)
    data = [] if gap is None else [{"dataset": "celebdfpp", "video_id": "fixture_0",
                                   "error_sse": gap**2, "total_sse": 1., "frame_count": 1}]
    pd.DataFrame(data, columns=INTERVENTION_COLUMNS).to_csv(directory / "intervention.csv", index=False)


def test_clean_only_pareto_retains_unavailable_objectives(tmp_path):
    runs = [run_record(head=head, budget=budget, covariates=covariates)
            for head, budget, covariates in itertools.product(
                ["linear", "mlp_gelu"], ["params", "time"], [False, True])]
    for run in runs:
        write_intervention(tmp_path / run["run_dir"])
    result = pareto_table(tmp_path, pd.DataFrame([pareto_row(run) for run in runs]), runs)
    assert len(result) == 8
    assert result[["A", "P", "C"]].notna().all().all()
    assert result[["R", "I_gap"]].isna().all().all()
    assert result.pareto_nondominated.isna().all()
    assert result.pareto_status.eq("UNAVAILABLE").all()
    assert all(set(json.loads(value)) == {"R", "I_gap"} for value in result.pareto_missing_objectives)


def test_pareto_complete_objectives_can_still_be_ranked(tmp_path):
    runs = [run_record(head=head) for head in ["linear", "mlp_gelu"]]
    rows = []
    for index, run in enumerate(runs):
        write_intervention(tmp_path / run["run_dir"], gap=.2 + index * .1)
        rows += [pareto_row(run, auc=.9 - index * .1), pareto_row(run, "jpeg70", .8 - index * .1)]
    result = pareto_table(tmp_path, pd.DataFrame(rows), runs).set_index("head")
    assert result.pareto_status.eq("DESCRIPTIVE_COMPLETE_CASES").all()
    assert bool(result.loc["linear", "pareto_nondominated"])
    assert not bool(result.loc["mlp_gelu", "pareto_nondominated"])
    assert all(json.loads(value) == [] for value in result.pareto_missing_objectives)


def test_pareto_does_not_hide_one_seed_missing_objectives_in_an_average(tmp_path):
    runs = [run_record(seed=seed) for seed in [0, 1]]
    rows = [pareto_row(run) for run in runs] + [pareto_row(runs[0], "jpeg70", .7)]
    write_intervention(tmp_path / runs[0]["run_dir"], gap=.2)
    write_intervention(tmp_path / runs[1]["run_dir"])
    result = pareto_table(tmp_path, pd.DataFrame(rows), runs)
    assert len(result) == 1
    assert result.run_count.iloc[0] == 2
    assert result.pareto_status.iloc[0] == "UNAVAILABLE"
    assert pd.isna(result.pareto_nondominated.iloc[0])
    assert set(json.loads(result.pareto_missing_objectives.iloc[0])) == {"R", "I_gap"}


def write_full_features_fixture(out):
    cfg = OmegaConf.create({
        "experiment": {"name": "full_features", "synthetic": True, "seeds": [0], "ks": [1024],
                       "heads": ["linear", "mlp_gelu"], "budgets": ["params", "time"],
                       "covariates": [True, False], "held_out_methods": [None], "conditions": ["clean"]},
        "eval": {"grid_points": 3, "bootstrap": 20, "margin": .01, "min_seeds_confirmatory": 10},
        "paths": {"output_dir": str(out.parent / "outputs"), "data_dir": str(out.parent / "data")},
        "train": {}, "stages": ["train", "report"],
        "force": False,
    })
    atomic_json(out / "manifest.json", {
        "synthetic": True, "config": OmegaConf.to_container(cfg), "source": "legacy-training-source",
        "cache_provenance": {"synthetic": True, "warning": "Report-only synthetic artifact fixture"},
    })
    OmegaConf.save(cfg, out / "config.yaml")
    runs = []
    for head, budget, covariates in itertools.product(["linear", "mlp_gelu"], ["params", "time"], [True, False]):
        run = run_record(head, budget, covariates)
        runs.append(run)
        directory = out / run["run_dir"]
        directory.mkdir(parents=True)
        score_rows = []
        for dataset, splits in [("ffpp", ["train", "val", "test"]), ("celebdfpp", ["test"])]:
            for split in splits:
                for index, (label, probability) in enumerate(zip([0, 0, 1, 1], [.15, .35, .65, .85])):
                    logit = float(np.log(probability / (1-probability)))
                    score_rows.append({"dataset": dataset, "condition": "clean", "video_id": f"{dataset}_{split}_{index}",
                                       "label": label, "split": split, "method": "Deepfakes" if label else "real",
                                       "scenario": "synthetic", "group_id": f"{dataset}_{split}_group{index}",
                                       "logit": logit, "probability": probability,
                                       "calibrated": float(1 / (1 + np.exp(-logit / 1.25))), "frames": 1})
        scores = pd.DataFrame(score_rows)
        scores.to_csv(directory / "scores.csv", index=False)
        test = scores[scores.split == "test"].copy()
        test["frame_idx"] = 0
        test.to_csv(directory / "frame_scores.csv", index=False)
        records, calibration = [], []
        for dataset, population in test.groupby("dataset"):
            for level, name, method in itertools.product(
                    ["frame", "video"], ["raw", "temperature"], ["all", "Deepfakes"]):
                column = "probability" if name == "raw" else "calibrated"
                value = metrics(population.label, population[column], .5)
                if level == "frame":
                    value.pop("auc_ci_delong_video", None)
                records.append({"dataset": dataset, "condition": "clean", "level": level,
                                "method": method, "calibration": name, **value})
                if method == "all":
                    calibration.extend({"dataset": dataset, "condition": "clean", "level": level,
                                        "calibration": name, **row}
                                       for row in reliability(population.label, population[column]))
        pd.DataFrame(records).to_csv(directory / "metrics.csv", index=False)
        pd.DataFrame(calibration).to_csv(directory / "calibration.csv", index=False)
        pd_rows = []
        if covariates:
            pd_rows = [{"feature": feature, "feature_index": 1024 + index, "grid_type": kind,
                        "grid": grid, "response": grid * .1 * (index + 1), "raw_response": .2 + grid * .1,
                        "q05": grid * .1 - .1, "q95": grid * .1 + .1, "background_rows": 4,
                        "within_train_support": True}
                       for index, feature in enumerate(COVARIATE_NAMES)
                       for kind, grid in itertools.product(["train_quantile", "shared_standardized"], [-2., 0., 2.])]
        pd.DataFrame(pd_rows, columns=PD_COLUMNS).to_csv(directory / "pd_curves.csv", index=False)
        write_intervention(directory)
        pd.DataFrame(columns=VIDEO_KEYS).to_csv(directory / "openset_scores.csv", index=False)
        symbolic = {"status": "not_applicable", "reason": "Full-feature reference is not a low-dimensional interpretation model"}
        atomic_json(directory / "symbolic_surrogate.json", symbolic)
        atomic_json(directory / "temperature.json", {"temperature": 1.25})
        pd.DataFrame({"epoch": [0], "validation_auc": [1.]}).to_csv(directory / "train_log.csv", index=False)
        atomic_json(directory / "metrics.json", {
            "run": {key: value for key, value in run.items() if key not in {
                "run_dir", "params", "head_latency_ms", "train_wall_s"}},
            "matching": {"params": run["params"], "feasible": head != "linear" or budget == "time"},
            "training": {"wall_s": 1., "validation_auc": 1., "time_budget_relative_error": 0.},
            "compute": {"params": run["params"], "input_features": 1024 + 4 * covariates,
                        "latency_batch256": {"p50_ms": run["head_latency_ms"]}},
            "eval": records, "openset": {"metrics": [], "reason": "Full-feature reference; novelty omitted"},
            "interp": {"symbolic": symbolic, "pd": {"status": "computed" if covariates else "not_applicable"},
                       "intervention": {"status": "unavailable" if covariates else "not_applicable"}},
        })
        mark_complete(directory, f"synthetic-full-features-{head}-{budget}-{covariates}",
                      sorted(path.name for path in directory.iterdir() if path.is_file()))
    return cfg, runs


def test_full_features_clean_only_report_stage_finishes_without_training(tmp_path, monkeypatch):
    out = tmp_path / "completed_full_features"
    cfg, runs = write_full_features_fixture(out)
    latest = tmp_path / "outputs" / "full_features" / "LATEST.json"
    atomic_json(latest, {"path": str(out), "key": "legacy-trained-experiment"})
    def no_training(*args, **kwargs):
        raise AssertionError("Report-only recovery must not train or load a feature cache")
    monkeypatch.setattr("fresco.run.train_experiment", no_training)
    monkeypatch.setattr("fresco.run.load_cache", no_training)
    cfg.stages = ["report"]
    inputs = {path: digest_file(path) for run in runs for path in (out / run["run_dir"]).iterdir()}
    assert execute(cfg) == out
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "SYNTHETIC INTEGRATION FIXTURE" in report
    assert "Completed runs: 8." in report
    assert "Measured results from the supplied face videos" not in report
    for name in ["results.csv", "results_summary.csv", "pareto.csv", "stats.json", "stability.csv",
                 "openset_results.csv", "symbolic_fidelity.csv"]:
        assert (out / name).is_file()
    pareto = pd.read_csv(out / "pareto.csv")
    assert len(pareto) == 8
    assert pareto[["R", "I_gap"]].isna().all().all()
    assert pareto.pareto_nondominated.isna().all()
    assert pareto.pareto_status.eq("UNAVAILABLE").all()
    assert (out / "openset_results.csv").read_text(encoding="utf-8").strip() == ""
    assert (out / "symbolic_fidelity.csv").read_text(encoding="utf-8").strip() == ""
    statistics = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert statistics["primary_paired_response"]["status"] == "INCONCLUSIVE"
    assert statistics["H2a"] == []
    stability = pd.read_csv(out / "stability.csv")
    assert set(stability.feature) == set(COVARIATE_NAMES)
    assert not stability.coverage_complete.any()  # One seed has no defined seed-pair correlation.
    for stem in ["auc_celebdfpp_by_head", "pareto_auc_latency", "robustness_curves",
                 "response_curves_covariates", "stability", "calibration_reliability", "covariate_ablation"]:
        assert (out / "figures" / f"{stem}.svg").is_file()
        assert (out / "figures" / f"{stem}.csv").is_file()
        assert (out / "qa" / f"{stem}.png").is_file()
    recorded = json.loads((out / "report_stage" / "artifacts.json").read_text(encoding="utf-8"))
    assert all(digest_file(out / name) == checksum for name, checksum in recorded.items())
    marker = json.loads((out / "report_stage" / "DONE.json").read_text(encoding="utf-8"))
    assert complete(out / "report_stage", marker["key"])
    assert all(digest_file(path) == checksum for path, checksum in inputs.items())
    # A report-only retry uses the completed report and leaves report outputs untouched.
    timestamp = (out / "REPORT.md").stat().st_mtime_ns
    assert execute(cfg) == out
    assert (out / "REPORT.md").stat().st_mtime_ns == timestamp
