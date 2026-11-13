from __future__ import annotations

import argparse
import json
import pickle
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from .surrogates import fit_surrogates


def run_worker(shared_path, targets_path, destination, seed):
    destination = Path(destination)
    shared = np.load(shared_path, allow_pickle=False)
    targets = np.load(targets_path, allow_pickle=False)
    train_x, val_x, test_x = shared["train_x"], shared["val_x"], shared["test_x"]
    train_meta = pd.DataFrame({"dataset": "ffpp", "split": "train", "video_id": shared["train_video"],
                               "group_id": shared["train_group"]})
    val_meta = pd.DataFrame({"dataset": "ffpp", "split": "val", "video_id": shared["val_video"],
                             "group_id": shared["val_group"]})
    models, selection = fit_surrogates(train_x, targets["train_y"], train_meta, val_x, targets["val_y"],
                                       val_meta, seed=seed, include_ebm=True, ebm_interactions=True)
    if "ebm" not in models or "ebm_interactions" not in models:
        raise RuntimeError(f"BLOCKED_DEPENDENCY: InterpretML EBM unavailable: {selection}")
    if any(len(term) != 1 for term in models["ebm"].term_features_):
        raise ValueError("Primary EBM contains an interaction")
    methods = ["zero", "ridge", "gam", "ebm", "ebm_interactions"]
    predictions = np.empty((len(methods), len(test_x)), dtype=np.float32)
    for j, name in enumerate(methods):
        start = time.perf_counter()
        predictions[j] = np.asarray(models[name].predict(test_x), dtype=np.float32)
        elapsed = time.perf_counter()-start
        if not np.isfinite(predictions[j]).all():
            raise ValueError(f"Nonfinite {name} predictions")
        selection[name]["test_prediction_seconds"] = elapsed
        selection[name]["test_seconds_per_pair"] = elapsed/max(len(test_x), 1)
        selection[name]["model_bytes_pickle"] = len(pickle.dumps(models[name], protocol=5))
        if name.startswith("ebm"):
            selection[name]["term_count"] = len(models[name].term_features_)
            selection[name]["term_features"] = [list(term) for term in models[name].term_features_]
        elif name in ("gam", "ridge"):
            selection[name]["term_count"] = 8
            selection[name]["coefficient_count"] = int(models[name][-1].coef_.size)
    np.save(destination / "predictions.npy", predictions, allow_pickle=False)

    center = np.median(train_x, axis=0)
    function_rows = []
    for j in range(8):
        low, high = np.quantile(train_x[:, j], [.01, .99])
        grid = np.linspace(low, high, 31)
        if low < high:
            counts, bins = np.histogram(train_x[:, j], bins=30, range=(low, high))
            bin_index = np.clip(np.searchsorted(bins, grid, side="right")-1, 0, len(counts)-1)
            occupancy = counts[bin_index]
        else:
            occupancy = np.full(len(grid), len(train_x), dtype=int)
        x = np.tile(center, (len(grid), 1))
        x[:, j] = grid
        for name in ("gam", "ebm"):
            reference = float(models[name].predict(center.reshape(1, -1))[0])
            response = models[name].predict(x)-reference
            for point, value, count in zip(grid, response, occupancy):
                function_rows.append({"feature": j, "method": name, "grid": point, "response": value,
                                      "source_q01": low, "source_q99": high, "source_bin_count": int(count),
                                      "source_training_pairs": len(train_x)})
    pd.DataFrame(function_rows).to_csv(destination / "functions.csv", index=False)
    metadata = {"seed": seed, "methods": methods, "selection": selection,
                "versions": {name: version(name) for name in ("interpret-core", "numpy", "pandas", "scikit-learn")},
                "fit_access": "FF++ train responses; source validation used only for hyperparameter selection",
                "test_teacher_responses_supplied_to_worker": False,
                "primary_ebm_interactions": 0, "exploratory_ebm_interactions": 2}
    (destination / "worker.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_worker(args.shared, args.targets, args.destination, args.seed)


if __name__ == "__main__":
    main()