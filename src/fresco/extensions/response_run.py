from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.covariance import LedoitWolf
from tqdm.auto import tqdm
from fresco.interpret import intervention_decomposition
from fresco.research_audit import paired_gap_uncertainty
from fresco.response_audit import curve_delta
from fresco.training import COVARIATE_NAMES
from fresco.utils import atomic_json
from .inventory import sha256
from .paired import response_metrics
from .preflight import HEADS, SEEDS
from .runtime import FrozenInputs, create_run, digest_id


def read_historical_curves(directory, expected=None):
    directory = Path(directory)
    provenance_path = directory / "provenance.json"
    if expected is not None:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (provenance.get("original_manifest_sha256") != expected["main_manifest_sha256"]
                or provenance.get("coordinates_sha256") != expected["coordinates_sha256"]
                or provenance.get("selected_seeds") != SEEDS
                or provenance.get("declared_seeds") != SEEDS
                or provenance.get("complete_declared_seed_coverage") is not True):
            raise ValueError("Historical curves do not belong to the verified original experiment")
        proofs = {(r["run"]["head"], r["run"]["seed"]): r for r in provenance.get("runs", [])}
        original = {(r["head"], r["seed"]): r["checkpoint_sha256"] for r in expected["teachers"]}
    else:
        proofs, original = {}, {}
    curves, reference, hashes = {}, None, {}
    for head in HEADS:
        for seed in SEEDS:
            path = directory / head / f"seed{seed}" / "curve_predictor.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing historical v2 curve: {path}")
            if expected is not None:
                proof = proofs.get((head, seed), {})
                if (proof.get("checkpoint_sha256") != original.get((head, seed))
                        or proof.get("frozen_predictor_sha256") != sha256(path)):
                    raise ValueError(f"Frozen historical curve or checkpoint checksum mismatch: {path}")
            table = pd.read_csv(path)
            grids, values = [], []
            for name in COVARIATE_NAMES:
                part = table.loc[table.feature == name].sort_values("grid")
                g, v = part.grid.to_numpy(float), part.response.to_numpy(float)
                if len(g) < 2 or not np.isfinite(g).all() or not np.isfinite(v).all() or not (np.diff(g) > 0).all():
                    raise ValueError(f"Invalid historical curve {name}: {path}")
                grids.append(g)
                values.append(v)
            if reference is None:
                reference = grids
            curves[(head, seed)] = (grids, values)
            hashes[f"{head}/seed{seed}"] = sha256(path)
    return curves, hashes


def common_coverage(c0, ca, curves):
    mask = np.ones(len(c0), dtype=bool)
    seen = set()
    for grids, _ in curves.values():
        key = tuple(tuple(g) for g in grids)
        if key not in seen:
            _, covered = curve_delta(c0, ca, grids, [np.zeros(len(g)) for g in grids])
            mask &= covered
            seen.add(key)
    return mask


def _quality_support(inputs, c0, ca):
    selected = ((inputs.meta.dataset == "ffpp") & (inputs.meta.split == "train")
                & (inputs.meta.condition == "clean")).to_numpy()
    indices = np.flatnonzero(selected)
    if len(indices) > 50000:
        indices = np.sort(np.random.default_rng(2026).choice(indices, 50000, replace=False))
    source = inputs.standardized_quality(indices)
    support = LedoitWolf().fit(source)
    source_dist = np.sqrt(np.maximum(support.mahalanobis(source), 0))
    bins = np.unique(np.quantile(source_dist, [0, .2, .4, .6, .8, 1]))
    if len(bins) < 2:
        bins = np.array([0., max(float(source_dist.max()), 1e-8)])
    d0 = np.sqrt(np.maximum(support.mahalanobis(c0), 0))
    da = np.sqrt(np.maximum(support.mahalanobis(ca), 0))
    distance = np.maximum(d0, da)
    return distance, np.clip(np.searchsorted(bins, distance, side="right")-1, 0, len(bins)-1), {
        "model": "Ledoit-Wolf Mahalanobis on source-train clean quality only; diagnostic, not manifold support",
        "source_rows": len(source), "distance_quantile_edges": bins.tolist(),
        "above_source_max_bin": len(bins)-1}


def _project(inputs, indices, batch=4096):
    indices = np.asarray(indices, dtype=np.int64)
    result = np.empty((len(indices), 20), dtype=np.float32)
    for start in tqdm(range(0, len(indices), batch), desc="Project supported features", leave=False):
        end = min(start+batch, len(indices))
        result[start:end] = inputs.coordinates(indices[start:end])
    return result


def _video_statistics(frame):
    table = frame.copy()
    table["total_sse"] = table.total**2
    table["curve_error_sse"] = (table.curve-table.total)**2
    table["block_error_sse"] = (table["cov"]-table.total)**2
    for name in ("cov", "embed", "interaction"):
        table[name+"_sse"] = table[name]**2
    keys = ["dataset", "video_hash", "group_hash", "label", "condition"]
    return table.groupby(keys, sort=True, observed=True, as_index=False).agg(
        pairs=("total", "size"), total_sse=("total_sse", "sum"),
        curve_error_sse=("curve_error_sse", "sum"), block_error_sse=("block_error_sse", "sum"),
        cov_sse=("cov_sse", "sum"), embed_sse=("embed_sse", "sum"),
        interaction_sse=("interaction_sse", "sum"),
        total_mean=("total", "mean"), cov_mean=("cov", "mean"), embed_mean=("embed", "mean"),
        interaction_mean=("interaction", "mean"))


def _strata(frame, head, seed, sign_tolerance):
    rows = []
    populations = []
    for dataset, part in frame.groupby("dataset", observed=True, sort=True):
        populations.append((dataset, "all", "all", part))
        populations.extend((dataset, condition, "all", sub) for condition, sub in
                           part.groupby("condition", observed=True, sort=True))
        populations.extend((dataset, "all", int(label), sub) for label, sub in
                           part.groupby("label", observed=True, sort=True))
    for dataset, condition, label, part in populations:
        for method in ("curve", "cov", "zero"):
            predicted = np.zeros(len(part)) if method == "zero" else part[method].to_numpy()
            values = response_metrics(part.total, predicted, part.video_hash,
                                      sign_tolerance=sign_tolerance)
            y = part.total.to_numpy()
            if len(y) > 1 and np.std(y) > 0 and np.std(predicted) > 0:
                values["pearson_frame_weighted"] = float(pearsonr(y, predicted).statistic)
                values["spearman_frame_weighted"] = float(spearmanr(y, predicted).statistic)
            else:
                values["pearson_frame_weighted"] = values["spearman_frame_weighted"] = None
            rows.append({"head": head, "seed": seed, "dataset": dataset, "condition": condition, "label": label,
                         "predictor": method, **values})
    return pd.DataFrame(rows)


def _bootstrap(seed_tables, *, resamples, seed):
    reports = {}
    for dataset in ("ffpp", "celebdfpp"):
        matrices, reference = {}, None
        for head in HEADS:
            errors_curve, errors_block, totals = [], [], []
            for s in SEEDS:
                table = seed_tables[(head, s)]
                table = table.loc[table.dataset == dataset].groupby(
                    ["video_hash", "group_hash"], observed=True, sort=True)[
                        ["pairs", "total_sse", "curve_error_sse", "block_error_sse"]].sum()
                if reference is None:
                    reference = table.index
                if not table.index.equals(reference):
                    raise ValueError("Common-support video/group populations differ across heads/seeds")
                n = table.pairs.to_numpy(float)
                errors_curve.append(table.curve_error_sse.to_numpy(float)/n)
                errors_block.append(table.block_error_sse.to_numpy(float)/n)
                totals.append(table.total_sse.to_numpy(float)/n)
            matrices[head] = (np.stack(errors_curve), np.stack(errors_block), np.stack(totals))
        group = reference.get_level_values("group_hash").to_numpy()
        kan, mlp = matrices["kan_spline"], matrices["mlp_gelu"]
        summary = {"head_difference_curve": paired_gap_uncertainty(
            kan[0], kan[2], mlp[0], mlp[2], group, resamples=resamples, seed=seed, min_total_rms=1e-8)}
        for head, vals in matrices.items():
            summary[f"{head}_curve_minus_zero"] = paired_gap_uncertainty(
                vals[0], vals[2], vals[2], vals[2], group, resamples=resamples, seed=seed, min_total_rms=1e-8)
            summary[f"{head}_block_minus_zero"] = paired_gap_uncertainty(
                vals[1], vals[2], vals[2], vals[2], group, resamples=resamples, seed=seed, min_total_rms=1e-8)
        reports[dataset] = summary
    return reports


def _historical_full_block(inputs):
    records = []
    for head in HEADS:
        for seed in SEEDS:
            run = inputs.common / head / "params" / f"seed{seed}"
            marker = json.loads((run / "DONE.json").read_text(encoding="utf-8"))
            path = run / "intervention.csv"
            if sha256(path) != marker.get("files", {}).get(path.name):
                raise ValueError(f"Historical full-test block artifact failed checksum: {path}")
            table = pd.read_csv(path, usecols=["dataset", "video_id", "frame_count", "error_sse", "total_sse"])
            for dataset, part in table.groupby("dataset", sort=True):
                video = part.groupby("video_id")[["frame_count", "error_sse", "total_sse"]].sum()
                numerator = (video.error_sse/video.frame_count).mean()
                denominator = (video.total_sse/video.frame_count).mean()
                records.append({"head": head, "seed": seed, "dataset": dataset,
                                "videos": len(video), "frames": int(video.frame_count.sum()),
                                "G_block_full_test": np.sqrt(numerator/denominator) if denominator > 1e-16 else None})
    return pd.DataFrame(records)


def run_response(cfg, cache, main, replay, curves_dir, run_id, device="cpu"):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; explicitly choose cpu")
    inputs = FrozenInputs(cache, main, replay)
    curves, curve_hashes = read_historical_curves(curves_dir, inputs.preflight)
    out, signature = create_run("response", run_id, cfg, inputs,
                                extra={"historical_curve_hashes": curve_hashes, "device": device})
    status_path = out / "seed_status.json"
    if status_path.exists():
        statuses = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        statuses = {f"{head}/seed{seed}": {"status": "PENDING"} for head in HEADS for seed in SEEDS}
        atomic_json(status_path, statuses)
    pairs = inputs.pairs()
    c0 = inputs.standardized_quality(pairs.clean_row.to_numpy())
    ca = inputs.standardized_quality(pairs.changed_row.to_numpy())
    supported = common_coverage(c0, ca, curves)
    distance, distance_bin, support_info = _quality_support(inputs, c0, ca)
    pairs["supported"] = supported
    coverage = pairs.groupby(["dataset", "condition", "label"], observed=True, as_index=False).agg(
        paired=("supported", "size"), supported=("supported", "sum"),
        videos=("video_id", "nunique"))
    covered_videos = pairs.loc[pairs.supported].groupby(
        ["dataset", "condition", "label"], observed=True).video_id.nunique().rename("supported_videos")
    coverage = coverage.merge(covered_videos.reset_index(), how="left", on=["dataset", "condition", "label"])
    coverage["supported_videos"] = coverage.supported_videos.fillna(0).astype(int)
    coverage["unsupported"] = coverage.paired-coverage.supported
    coverage["coverage"] = coverage.supported/coverage.paired
    (out / "figures").mkdir(exist_ok=True)
    coverage.to_csv(out / "figures" / "coverage.csv", index=False)
    feature_coverage = []
    for j, name in enumerate(COVARIATE_NAMES):
        mask = np.ones(len(c0), dtype=bool)
        for grids, _ in curves.values():
            unchanged = np.isfinite(c0[:, j]) & np.isfinite(ca[:, j]) & (c0[:, j] == ca[:, j])
            mask &= unchanged | ((c0[:, j] >= grids[j][0]) & (c0[:, j] <= grids[j][-1])
                                 & (ca[:, j] >= grids[j][0]) & (ca[:, j] <= grids[j][-1]))
        temp = pairs[["dataset", "condition", "label"]].copy()
        temp["eligible"] = mask
        f = temp.groupby(["dataset", "condition", "label"], observed=True, as_index=False).agg(
            paired=("eligible", "size"), supported=("eligible", "sum"))
        f["feature"] = name
        feature_coverage.append(f)
    pd.concat(feature_coverage, ignore_index=True).to_csv(out / "coverage_by_covariate.csv", index=False)
    atomic_json(out / "support_model.json", support_info)
    pairs = pairs.loc[supported].reset_index(drop=True)
    if pairs.empty:
        raise ValueError("No common-support pairs; report coverage as undefined")
    pairs["support_distance"] = distance[supported]
    pairs["support_bin"] = distance_bin[supported]
    pairs["video_hash"] = pairs.video_id.astype(str).map(lambda x: digest_id("video", x))
    pairs["group_hash"] = pairs.group_id.astype(str).map(lambda x: digest_id("group", x))
    pairs["pair_hash"] = [digest_id(d, v, int(f), c) for d, v, f, c in zip(
        pairs.dataset.astype(str), pairs.video_id.astype(str), pairs.frame_idx, pairs.condition.astype(str))]
    pairs[["dataset", "condition", "label", "video_hash", "group_hash", "pair_hash", "support_distance",
           "support_bin"]].to_csv(out / "common_support_ids.csv.gz", index=False, compression="gzip")
    x0 = _project(inputs, pairs.clean_row.to_numpy())
    xa = _project(inputs, pairs.changed_row.to_numpy())
    seed_tables, strata, components, agreement, support_results = {}, [], [], [], []
    epsilon = float(cfg.zero_denominator_epsilon)
    for head in HEADS:
        for seed in SEEDS:
            key = f"{head}/seed{seed}"
            statuses[key] = {"status": "RUNNING"}
            atomic_json(status_path, statuses)
            dest = out / head / f"seed{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            done = dest / "DONE.json"
            names = ["video.csv", "strata.csv", "components.csv", "agreement.csv", "support.csv"]
            if done.is_file():
                saved = json.loads(done.read_text(encoding="utf-8"))
                if saved.get("signature") != signature or any(sha256(dest/n) != saved["files"].get(n) for n in names):
                    raise ValueError(f"Corrupt or incompatible resumed seed: {dest}")
                video, metrics, component, agree, dist = [pd.read_csv(dest/n) for n in names]
            else:
                model = inputs.teacher(head, seed, device)
                values = {k: [] for k in ("total", "covariate", "embedding", "interaction")}
                for start in tqdm(range(0, len(x0), 4096), desc=f"{head} seed{seed}", leave=False):
                    a = torch.as_tensor(x0[start:start+4096], device=device)
                    b = torch.as_tensor(xa[start:start+4096], device=device)
                    result = intervention_decomposition(model, a, b, 16)
                    for component_name, pieces in values.items():
                        pieces.append(result[component_name])
                values = {key: np.concatenate(value) for key, value in values.items()}
                pred, available = curve_delta(x0[:, 16:], xa[:, 16:], *curves[(head, seed)])
                if not available.all():
                    raise ValueError("Common coverage changed across heads or seeds")
                frame = pairs[["dataset", "condition", "label", "video_hash", "group_hash", "support_bin"]].copy()
                frame["total"], frame["cov"], frame["embed"], frame["interaction"], frame["curve"] = (
                    values["total"], values["covariate"], values["embedding"], values["interaction"], pred)
                video = _video_statistics(frame)
                metrics = _strata(frame, head, seed, epsilon)
                grouped = frame.groupby(["dataset", "condition", "label"], observed=True, sort=True)
                component = grouped[["total", "cov", "embed", "interaction"]].agg(["mean", "median", "std"])
                component.columns = ["_".join(c) for c in component.columns]
                component["pairs"] = grouped.size()
                component = component.reset_index().assign(head=head, seed=seed)
                # Fixed-width binned means avoid displaying millions of opaque points
                frame["observed_bin"] = np.floor(np.clip(frame.total, -10, 10)*10)/10
                agree = frame.groupby(["dataset", "observed_bin"], observed=True, as_index=False).agg(
                    n=("total", "size"), observed=("total", "mean"), predicted=("curve", "mean"))
                agree["head"], agree["seed"] = head, seed
                dist_rows = []
                for (dataset, bin_id), part in frame.groupby(["dataset", "support_bin"], observed=True):
                    val = response_metrics(part.total, part.curve, part.video_hash, sign_tolerance=epsilon)
                    dist_rows.append({"head": head, "seed": seed, "dataset": dataset,
                                      "support_bin": bin_id, **val})
                dist = pd.DataFrame(dist_rows)
                for table, name in zip((video, metrics, component, agree, dist), names):
                    table.to_csv(dest/name, index=False)
                atomic_json(done, {"signature": signature, "files": {n: sha256(dest/n) for n in names}})
            video["head"], video["seed"] = head, seed
            seed_tables[(head, seed)] = video
            strata.append(metrics)
            components.append(component)
            agreement.append(agree)
            support_results.append(dist)
            statuses[key] = {"status": "COMPLETED"}
            atomic_json(status_path, statuses)
    pd.concat(seed_tables.values(), ignore_index=True).to_csv(out / "per_video.csv.gz", index=False, compression="gzip")
    pd.concat(strata, ignore_index=True).to_csv(out / "metrics.csv", index=False)
    pd.concat(components, ignore_index=True).to_csv(out / "figures" / "response_components.csv", index=False)
    pd.concat(agreement, ignore_index=True).to_csv(out / "figures" / "response_agreement.csv", index=False)
    pd.concat(support_results, ignore_index=True).to_csv(out / "figures" / "fidelity_vs_support.csv", index=False)
    confidence = _bootstrap(seed_tables, resamples=int(cfg.bootstrap_resamples), seed=int(cfg.bootstrap_seed))
    _historical_full_block(inputs).to_csv(out / "historical_full_block.csv", index=False)
    energy_rows = []
    for (head, seed), table in seed_tables.items():
        for dataset, part in table.groupby("dataset", observed=True):
            video = part.groupby("video_hash")[["pairs", "total_sse", "cov_sse", "embed_sse", "interaction_sse"]].sum()
            denominator = (video.total_sse/video.pairs).mean()
            row = {"head": head, "seed": seed, "dataset": dataset, "videos": len(video)}
            for name in ("cov", "embed", "interaction"):
                row[name+"_energy_fraction"] = ((video[name+"_sse"]/video.pairs).mean()/denominator
                                                if denominator > 1e-16 else None)
            energy_rows.append(row)
    pd.DataFrame(energy_rows).to_csv(out / "response_energy.csv", index=False)
    atomic_json(out / "metrics.json", {"status": "RUN", "seeds": SEEDS, "bootstrap": confidence,
                                       "covered_pairs": len(pairs), "historical_coverage_rule": "v2",
                                       "weighting": "equal video", "support": support_info,
                                       "original_verdict": "NOT_SUPPORTED; unchanged"})
    from .visuals import response_figures
    response_figures(out / "figures")
    from .figure_manifest import write_figure_manifest
    write_figure_manifest(out / "figures", "response")
    (out / "REPORT.md").write_text(
        "# Response: common supported pairs\n\n"
        f"All ten original seeds per head; {len(pairs)} common-support frame/condition pairs. "
        "G_curve, G_block and zero use identical pairs and equal-video weights. "
        "The historical full-test G_block belongs to a different population and is not subtracted here.\n\n"
        "See metrics.json for approximate fixed-seed cluster and crossed seed × cluster percentile CIs, "
        "metrics.csv for condition-level response metrics, and figures/ for SVG and source CSV.\n\n"
        "## Limitations\n\nExploratory after prior target inspection; intervals are unadjusted and conditional "
        "on the source training data and test generators. Source marginal support and Mahalanobis distance "
        "do not establish joint manifold support. Hybrid responses are not causal effects. "
        "Component energies need not sum to one because of covariance. Near-zero responses are counted.\n",
        encoding="utf-8")
    return out