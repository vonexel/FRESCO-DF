from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from fresco.factorial_audit import factorial_auc_uncertainty
from fresco.run import train_experiment
from fresco.utils import atomic_json
from .historical_reference import discover_from_main
from .inventory import sha256
from .preflight import HEADS, SEEDS
from .reuse import inspect_candidate
from .runtime import FrozenInputs, create_run, digest_id
from .source_pairs import load_source_pairs


RECIPES = ("clean", "augmented")


def _label_factorial_report(report, k, dataset):
    """Keep this extension's predeclared primary separate from the old audit's."""
    result = dict(report)
    result["historical_engine_primary_diagnostic"] = report.get("primary_diagnostic")
    result["primary_diagnostic"] = ("clean: architecture x covariates interaction"
                                    if k == 16 and dataset == "celebdfpp" else None)
    result["analysis_role"] = ("predeclared primary diagnostic; exploratory after target inspection"
                               if result["primary_diagnostic"] else "predeclared secondary or sensitivity analysis")
    return result


def _assert_matching_coordinates(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[name], b[name]) for name in a.files):
            raise ValueError("Clean/augmented source-fitted coordinates differ")


def validate_reuse_candidates(clean, augmented, source_cache, historical_code):
    if any(x is None for x in (clean, augmented, source_cache, historical_code)):
        raise FileNotFoundError("BLOCKED_INPUT: exact historical clean/augmented runs, their common transformed "
                                "cache and historical source directory must be supplied; old heads will not be retrained")
    manifest = json.loads((Path(source_cache) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_only") is True:
        raise ValueError("BLOCKED_INPUT: this new source-only delta cache cannot prove identity with the historical "
                         "combined cache used by the existing k16 heads")
    reports = {}
    for recipe, path in (("clean", clean), ("augmented", augmented)):
        report = inspect_candidate(path, source_cache, historical_code, recipe)
        reports[recipe] = report
        if report["status"] != "POSSIBLE_MATCH_REVIEW_REQUIRED":
            raise ValueError(f"BLOCKED_REUSE: {recipe} recipe is not exactly verified: {report['reasons']}")
        for cov in (0, 1):
            common = Path(path) / f"k16_cov{cov}" / "heldout_none"
            coords_hash = sha256(common / "coordinates.npz")
            for head in HEADS:
                for seed in SEEDS:
                    checkpoint = torch.load(common/head/"params"/f"seed{seed}"/"checkpoint.pt",
                                            map_location="cpu", weights_only=True)
                    if checkpoint.get("coordinates_sha256") != coords_hash:
                        raise ValueError("Historical k16 checkpoint is not bound to its saved coordinates")
                    matching = checkpoint.get("matching", {})
                    if (matching.get("target_params") != 1585 or not matching.get("feasible")
                            or matching.get("relative_error", 1) > .05):
                        raise ValueError("Historical k16 parameter match is outside the fixed tolerance")
    a, b = (json.loads((Path(p)/"manifest.json").read_text(encoding="utf-8"))["cache"]
            for p in (clean, augmented))
    if a != b:
        raise ValueError("Historical clean and augmented runs use different feature-cache hashes")
    for cov in (0, 1):
        _assert_matching_coordinates(Path(clean)/f"k16_cov{cov}"/"heldout_none"/"coordinates.npz",
                                     Path(augmented)/f"k16_cov{cov}"/"heldout_none"/"coordinates.npz")
    return reports


def _config_for_k128(recipe, cache, out, device):
    root = Path(__file__).resolve().parents[3] / "configs"
    if not root.is_dir():
        root = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(root)):
        config = compose(config_name="config", overrides=[f"experiment=response_{recipe}"])
    config.experiment.name = f"representation_k128_{recipe}"
    config.experiment.ks = [128]
    config.experiment.budgets = ["params"]
    config.experiment.target_params = 9585
    config.paths.feature_cache = str(cache)
    config.paths.output_dir = str(out)
    config.train.device = device
    return config


def _measurement(labels, probabilities, threshold):
    labels, p = np.asarray(labels, int), np.asarray(probabilities, float)
    if labels.shape != p.shape or not np.isfinite(p).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("AUC requires aligned finite scores for both video classes")
    fpr, tpr, _ = roc_curve(labels, p)
    nreal = int((labels == 0).sum())
    values = {"auc": float(roc_auc_score(labels, p)), "balanced_accuracy": float(
        balanced_accuracy_score(labels, p >= threshold)), "source_validation_threshold": float(threshold),
        "n_videos": len(labels), "n_real": nreal, "n_fake": int((labels == 1).sum())}
    for level in (.01, .05):
        key = f"tpr_at_fpr_{int(level*100)}pct"
        values[key] = float(tpr[fpr <= level].max()) if nreal*level >= 5 else None
        values[key+"_status"] = "computed" if nreal*level >= 5 else "unstable_insufficient_negatives"
    return values


def _saved_seed_valid(path):
    try:
        path = Path(path)
        marker = json.loads((path/"DONE.json").read_text(encoding="utf-8"))
        return all(marker.get("files", {}).get(name) == sha256(path/name)
                   for name in ("checkpoint.pt", "metrics.json", "scores.csv", "temperature.json"))
    except (OSError, KeyError, ValueError, TypeError):
        return False


def validate_saved_factorial_artifacts(clean, augmented, original_main):
    columns = ["dataset", "condition", "video_id", "label", "split", "method", "scenario", "group_id", "frames"]

    def population(run):
        if not _saved_seed_valid(run):
            raise ValueError(f"Saved factorial score/checkpoint verification failed: {run}")
        scores = pd.read_csv(run/"scores.csv", usecols=columns, keep_default_na=False)
        held_out = scores.loc[scores.split == "test", columns].reset_index(drop=True)
        if held_out.empty or held_out.duplicated(["dataset", "condition", "video_id"]).any():
            raise ValueError(f"Saved factorial held-out video population is incomplete or duplicated: {run}")
        return held_out

    reference = population(Path(original_main)/"k16_cov1"/"heldout_none"/"kan_spline"/"params"/"seed0")
    count = 0
    for recipe, experiment in (("clean", clean), ("augmented", augmented)):
        for cov in (0, 1):
            for head in HEADS:
                for seed in SEEDS:
                    run = Path(experiment)/f"k16_cov{cov}"/"heldout_none"/head/"params"/f"seed{seed}"
                    if not reference.equals(population(run)):
                        raise ValueError(f"Historical factorial test video/condition population differs: {recipe}/{head}/cov{cov}/seed{seed}")
                    count += 1
    return count


def _read_run(experiment, k, recipe, head, cov, seed):
    run = Path(experiment)/f"k{k}_cov{int(cov)}"/"heldout_none"/head/"params"/f"seed{seed}"
    marker = json.loads((run/"DONE.json").read_text(encoding="utf-8"))
    for name in ("checkpoint.pt", "metrics.json", "scores.csv", "temperature.json"):
        if marker.get("files", {}).get(name) != sha256(run/name):
            raise ValueError(f"Missing or corrupted saved representation artifact: {run/name}")
    result = json.loads((run/"metrics.json").read_text(encoding="utf-8"))
    identity = result["run"]
    wanted = {"head": head, "seed": seed, "covariates": cov, "k": k, "budget": "params", "held_out": None}
    if any(identity.get(key) != value for key, value in wanted.items()):
        raise ValueError(f"Saved run identity mismatch: {run}")
    temperature = json.loads((run/"temperature.json").read_text(encoding="utf-8"))
    threshold = temperature["thresholds"]["video"]["raw"]
    if temperature.get("target_validation_fpr") != .05:
        raise ValueError("Expected source-validation-selected 5% FPR threshold")
    scores = pd.read_csv(run/"scores.csv", keep_default_na=False)
    return result, float(threshold), scores


def _analyze(experiments, cfg, out):
    metrics, cells, identity_ref, seed_rows = [], {}, {}, []
    scores_tmp = out/"per_video_scores.csv.gz.tmp"
    first_chunk = True
    for k in (16, 128):
        for rindex, recipe in enumerate(RECIPES):
            for hindex, head in enumerate(HEADS):
                for cov in (False, True):
                    for seed in SEEDS:
                        result, threshold, scores = _read_run(experiments[(k, recipe)], k, recipe, head, cov, seed)
                        matching, train = result["matching"], result["training"]
                        for (dataset, condition), part in scores.loc[scores.split == "test"].groupby(
                                ["dataset", "condition"], sort=True):
                            measure = _measurement(part.label, part.probability, threshold)
                            row = {"k": k, "recipe": recipe, "head": head, "covariates": cov, "seed": seed,
                                   "dataset": dataset, "condition": condition, **measure,
                                   "parameters": matching["params"], "budget_relative_error": matching.get("relative_error"),
                                   "budget_feasible": matching.get("feasible"), "train_wall_seconds": train.get("wall_s"),
                                   "peak_vram_mb": train.get("peak_allocated_mb"),
                                   "input_features": k+4*int(cov), "source_validation_auc": train.get("validation_auc"),
                                   "selected_lr": result["run"]["lr"],
                                   "head_latency_batch1_ms": result.get("compute", {}).get("latency_batch1", {}).get("p50_ms")}
                            metrics.append(row)
                            out_rows = part[["video_id", "group_id", "label", "probability", "frames"]].copy()
                            out_rows["video_id"] = out_rows.video_id.map(lambda v: digest_id("video", v))
                            out_rows["group_id"] = out_rows.group_id.map(lambda v: digest_id("group", v))
                            out_rows = out_rows.assign(k=k, recipe=recipe, head=head, covariates=cov,
                                                       seed=seed, dataset=dataset, condition=condition)
                            # Stream the large 160-run table: retaining every
                            # hashed identifier in memory can exhaust a server.
                            # Concatenated gzip members are valid gzip streams.
                            out_rows.to_csv(scores_tmp, index=False, header=first_chunk,
                                            mode="w" if first_chunk else "a",
                                            compression={"method": "gzip", "mtime": 0})
                            first_chunk = False
                            if condition == "clean":
                                part = part.sort_values("video_id").reset_index(drop=True)
                                identity = part[["video_id", "label", "method", "group_id", "frames"]]
                                key = (k, dataset)
                                if key not in identity_ref:
                                    identity_ref[key] = identity
                                elif not identity_ref[key].equals(identity):
                                    raise ValueError(f"Factorial cells have different test videos or provenance: {key}")
                                cells[(k, dataset, rindex, hindex, int(cov), seed)] = part.probability.to_numpy(float)
                        val_aug = scores.loc[(scores.dataset == "ffpp") & (scores.split == "val")
                                             & (scores.condition != "clean")]
                        for condition, part in val_aug.groupby("condition", sort=True):
                            if part.label.nunique() == 2:
                                seed_rows.append({"k": k, "recipe": recipe, "head": head, "covariates": cov,
                                                  "seed": seed, "condition": condition,
                                                  "validation_video_auc": float(roc_auc_score(part.label, part.probability)),
                                                  "role": "source-only diagnostic; not used for HPO"})
    for dataset in ("ffpp", "celebdfpp"):
        if not identity_ref[(16, dataset)].equals(identity_ref[(128, dataset)]):
            raise ValueError(f"k16 and k128 evaluation populations differ for {dataset}")
    table = pd.DataFrame(metrics)
    # This tests the pair itself rather than relying only on distance to a common target.
    match = table.loc[(table.dataset == "ffpp") & (table.condition == "clean")].pivot_table(
        index=["k", "recipe", "covariates", "seed"], columns="head", values="parameters")
    match["pair_relative_error"] = abs(match.kan_spline-match.mlp_gelu)/match.mlp_gelu
    match["pair_matched_within_5pct"] = match.pair_relative_error <= .05
    table = table.merge(match[["pair_relative_error", "pair_matched_within_5pct"]].reset_index(),
                        on=["k", "recipe", "covariates", "seed"], validate="many_to_one")
    table.to_csv(out/"metrics.csv", index=False)
    scores_tmp.replace(out/"per_video_scores.csv.gz")
    pd.DataFrame(seed_rows).to_csv(out/"source_augmented_validation.csv", index=False)
    reports = {}
    for k in (16, 128):
        for dataset in ("ffpp", "celebdfpp"):
            reference = identity_ref[(k, dataset)]
            arrays = np.empty((2, 2, 2, 10, len(reference)), dtype=float)
            for r in range(2):
                for h in range(2):
                    for cov in range(2):
                        for seed in SEEDS:
                            arrays[r, h, cov, seed] = cells[(k, dataset, r, h, cov, seed)]
            report = factorial_auc_uncertainty(reference.label.to_numpy(), arrays, reference.group_id.to_numpy(),
                                               resamples=int(cfg.bootstrap_resamples), seed=int(cfg.bootstrap_seed))
            reports[f"k{k}/{dataset}"] = _label_factorial_report(report, k, dataset)
    return table, reports


def run_representation(cfg, cache, main, replay, source_cache, clean_run, augmented_run,
                       historical_code, run_id, device="cuda", *, historical_factorial_cache=False):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; choose cpu")
    reuse = validate_reuse_candidates(clean_run, augmented_run, source_cache, historical_code)
    if historical_factorial_cache:
        verified_saved_runs = validate_saved_factorial_artifacts(clean_run, augmented_run, main)
        if verified_saved_runs != 80:
            raise ValueError("Historical factorial saved-run coverage is incomplete")
    inputs = FrozenInputs(cache, main, replay)
    if historical_factorial_cache:
        from .factorial_cache import audit_factorial_cache
        source = audit_factorial_cache(inputs, source_cache, reuse, device=device)
        source["saved_factorial_runs_verified"] = verified_saved_runs
    else:
        source = load_source_pairs(inputs, source_cache)
    reference, reference_audit = discover_from_main(main)
    extra = {
        "source_cache_manifest_sha256": source["manifest_sha256"],
        "clean_manifest_sha256": sha256(Path(clean_run)/"manifest.json"),
        "augmented_manifest_sha256": sha256(Path(augmented_run)/"manifest.json"), "device": device,
        "historical_full_feature_reference": reference_audit}
    if historical_factorial_cache:
        extra["historical_factorial_cache_audit"] = source
    out, _ = create_run("representation", run_id, cfg, inputs, extra=extra)
    atomic_json(out/"reuse_audit.json", reuse)
    atomic_json(out/"historical_reference_audit.json", reference_audit)
    if historical_factorial_cache:
        atomic_json(out/"factorial_cache_audit.json", source)
    if reference is not None:
        reference.to_csv(out/"historical_full_feature_reference.csv", index=False)
    status = {f"k{k}/{recipe}/{head}/cov{int(cov)}/seed{seed}": {"status": "REUSED" if k == 16 else "PENDING"}
              for k in (16, 128) for recipe in RECIPES for head in HEADS for cov in (False, True) for seed in SEEDS}
    atomic_json(out/"seed_status.json", status)
    experiments = {(16, "clean"): Path(clean_run), (16, "augmented"): Path(augmented_run)}
    for recipe in RECIPES:
        run_cfg = _config_for_k128(recipe, source_cache, out, device)
        try:
            experiments[(128, recipe)] = train_experiment(run_cfg, Path(source_cache))
        except Exception as exc:
            latest = out / f"representation_k128_{recipe}" / "LATEST.json"
            partial = (Path(json.loads(latest.read_text(encoding="utf-8"))["path"]) if latest.is_file() else None)
            for head in HEADS:
                for cov in (False, True):
                    for seed in SEEDS:
                        candidate = (partial/f"k128_cov{int(cov)}"/"heldout_none"/head/"params"/f"seed{seed}"
                                     if partial is not None else None)
                        status[f"k128/{recipe}/{head}/cov{int(cov)}/seed{seed}"] = {
                            "status": "COMPLETED" if candidate is not None and _saved_seed_valid(candidate)
                                      else "BLOCKED",
                            "reason": f"training interrupted or failed: {type(exc).__name__}: {exc}"}
            atomic_json(out/"seed_status.json", status)
            raise
        for head in HEADS:
            for cov in (False, True):
                for seed in SEEDS:
                    path = experiments[(128, recipe)]/f"k128_cov{int(cov)}"/"heldout_none"/head/"params"/f"seed{seed}"
                    status[f"k128/{recipe}/{head}/cov{int(cov)}/seed{seed}"] = {
                        "status": "COMPLETED" if _saved_seed_valid(path) else "FAILED"}
        atomic_json(out/"seed_status.json", status)
    for cov in (0, 1):
        _assert_matching_coordinates(experiments[(128, "clean")]/f"k128_cov{cov}"/"heldout_none"/"coordinates.npz",
                                     experiments[(128, "augmented")]/f"k128_cov{cov}"/"heldout_none"/"coordinates.npz")
    table, reports = _analyze(experiments, cfg, out)
    from .representation_visuals import representation_figures, representation_tables
    (out/"figures").mkdir(exist_ok=True)
    representation_tables(table, reports, out/"figures")
    representation_figures(out/"figures")
    from .figure_manifest import write_figure_manifest
    write_figure_manifest(out/"figures", "representation")
    atomic_json(out/"metrics.json", {"status": "RUN", "seeds": SEEDS, "contrasts": reports,
                                     "historical_k16_reused": True, "new_k128_runs": 80,
                                     "original_verdict": "NOT_SUPPORTED; unchanged"})
    cache_note = ("The factorial uses the same verified historical combined feature cache for every cell. "
                  "Its clean source embeddings differ numerically from the original teacher cache; the original "
                  "baseline is a separate reference, not an identical factorial cell. "
                  "See factorial_cache_audit.json for the recorded discrepancy and population checks.\n\n"
                  if historical_factorial_cache else "")
    (out/"REPORT.md").write_text(
        "# Representation factorial\n\n"
        "Sixteen predeclared cells; ten paired seeds per cell. Historical k16 models were reused only after "
        "cache, source, runtime, HPO, coordinates and checkpoint verification; k128 is a separate extension. "
        "The primary diagnostic is KAN minus GELU in the covariate effect at k16/clean.\n\n" + cache_note +
        "See metrics.csv, metrics.json and figures/ for outcomes, parameter matching, seed trajectories and "
        "approximate fixed-seed cluster and crossed seed × cluster confidence intervals.\n\n"
        "Historical full-feature linear/GELU probes, when verified and present, are contextual references "
        "in a separate table; they are never included in the matched factorial contrasts.\n\n"
        "## Limitations\n\nExploratory after target inspection; multiple comparisons are unadjusted. "
        "Source and target generators and source training data are fixed. Head latency excludes encoder and preprocessing. "
        "A k16→k128 difference is not a pure causal effect of PCA information loss.\n", encoding="utf-8")
    return out