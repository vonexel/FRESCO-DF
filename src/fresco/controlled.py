from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from .heads import count_params
from .training import build_model
from .utils import atomic_json, complete, fingerprint, mark_complete, seed_everything, source_fingerprint, environment, digest_file


TASKS = ("smooth_additive", "composition", "oscillatory", "piecewise_smooth", "noisy_additive")


def target_function(x: torch.Tensor, task: str) -> torch.Tensor:
    # analytic scalar ground truth on [-1,1]^8; noisy case returns it's clean mean
    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError("Controlled functions require [N,8] inputs")
    if task in {"smooth_additive", "noisy_additive"}:
        return (torch.sin(math.pi*x) + .3*x.square()).sum(1)/math.sqrt(8)
    if task == "composition":
        return (torch.sin(math.pi*(x[:, 0]*x[:, 1] + .5*x[:, 2].square()))
                + .5*torch.cos(math.pi*(x[:, 3]+x[:, 4])*x[:, 5]) + .25*x[:, 6]*x[:, 7])
    if task == "oscillatory":
        return (torch.sin(6*math.pi*x[:, 0])*torch.cos(4*math.pi*x[:, 1])
                + .5*torch.sin(3*math.pi*x[:, 2:]).sum(1)/math.sqrt(6))
    if task == "piecewise_smooth":
        return (x[:, 0].abs() + torch.relu(x[:, 1]-.2) - .5*torch.relu(-x[:, 2]-.3)
                + .25*x[:, 3:].square().sum(1))
    raise ValueError(f"Unknown task {task!r}; expected {TASKS}")


def generate_data(task, sizes=(2048, 512, 2048), seed=2026, noise_std=.1):
    if len(sizes) != 3 or min(sizes) < 1 or noise_std < 0:
        raise ValueError("Require three positive split sizes and nonnegative noise")
    generator = torch.Generator().manual_seed(int(seed))
    noise_generator = torch.Generator().manual_seed(int(seed)+1)
    result = {}
    for split, n in zip(("train", "val", "test"), sizes):
        x = 2*torch.rand(int(n), 8, generator=generator)-1
        clean = target_function(x, task)
        observed = clean + noise_std*torch.randn(int(n), generator=noise_generator) if task == "noisy_additive" else clean.clone()
        result[split] = (x, observed, clean)
    return result


def _fit(model, data, cfg, lr, seed, budget):
    seed_everything(seed)
    device = str(cfg.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use train.device=cpu explicitly")
    torch.set_num_threads(int(cfg.get("threads", 4)))
    model = model.to(device).float()
    tx, ty, _ = (v.to(device) for v in data["train"])
    vx, vy, _ = (v.to(device) for v in data["val"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(cfg.weight_decay))
    with torch.inference_mode():
        initial_mse = float((model(tx)-ty).square().mean())
        best = float((model(vx)-vy).square().mean())
    best_state, stale, history = copy.deepcopy(model.state_dict()), 0, []
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started, epoch, stop = time.perf_counter(), 0, False
    maximum = int(cfg.epochs) if budget == "params" else 1000000
    if maximum < 1 or float(cfg.wall_seconds) <= 0:
        raise ValueError("Positive epochs and wall_seconds required")
    while epoch < maximum and not stop:
        model.train()
        order = torch.randperm(len(tx), device=device)
        for indices in order.split(int(cfg.batch_size)):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(tx[indices])-ty[indices]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Controlled training produced nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if budget == "time" and time.perf_counter()-started >= float(cfg.wall_seconds):
                stop = True
                break
        model.eval()
        with torch.inference_mode():
            val_mse = float((model(vx)-vy).square().mean())
        elapsed = time.perf_counter()-started
        history.append({"epoch": epoch, "train_batch_mse": float(loss.detach()), "val_mse": val_mse, "wall_s": elapsed})
        if val_mse < best:
            best, best_state, stale = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        epoch += 1
        stop |= (budget == "params" and stale >= int(cfg.patience))
        stop |= (budget == "time" and elapsed >= float(cfg.wall_seconds))
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-started
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        final_mse = float((model(tx)-ty).square().mean())
    return model, {"val_mse": best, "initial_train_mse": initial_mse, "train_mse": final_mse,
                   "wall_s": elapsed, "epochs": epoch, "lr": float(lr), "history": history,
                   "requested_wall_s": float(cfg.wall_seconds) if budget == "time" else None,
                   "time_budget_relative_error": abs(elapsed-float(cfg.wall_seconds))/float(cfg.wall_seconds) if budget == "time" else None,
                   "peak_allocated_mb": torch.cuda.max_memory_allocated(device)/2**20 if device.startswith("cuda") else None}


def _test(model, data, task, device):
    x, observed, clean = (v.to(device) for v in data["test"])
    with torch.inference_mode():
        prediction = model(x)
        mse, clean_mse = float((prediction-observed).square().mean()), float((prediction-clean).square().mean())
    mask = torch.ones(len(x), dtype=torch.bool, device=device)
    if task == "piecewise_smooth":
        mask = (x[:, 0].abs() > .02) & ((x[:, 1]-.2).abs() > .02) & ((x[:, 2]+.3).abs() > .02)
    differentiable = x[mask].detach().requires_grad_(True)
    if len(differentiable):
        learned = torch.autograd.grad(model(differentiable).sum(), differentiable)[0]
        truth = torch.autograd.grad(target_function(differentiable, task).sum(), differentiable)[0]
        derivative_rmse = float((learned-truth).square().mean().sqrt())
    else:
        derivative_rmse = None
    return {"test_mse_observed": mse, "test_mse_noiseless": clean_mse,
            "test_gradient_rmse": derivative_rmse, "gradient_rows": int(mask.sum()), "test_rows": len(x),
            "gradient_scope": "underlying analytic mean; piecewise rows within .02 of kinks excluded"}


def _aggregate(rows, directory):
    table = pd.DataFrame(rows)
    table.to_csv(directory / "metrics.csv", index=False)
    atomic_json(directory / "metrics.json", rows)
    measures = ["test_mse_noiseless", "test_mse_observed", "test_gradient_rmse", "wall_s", "params"]
    summary = table.groupby(["task", "head", "budget"], sort=False)[measures].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    feasible = table.groupby(["task", "head", "budget"], as_index=False).parameter_budget_feasible.all()
    summary = summary.merge(feasible, on=["task", "head", "budget"], validate="one_to_one")
    summary.to_csv(directory / "comparison.csv", index=False)
    tasks = list(table.task.unique())
    fig, axes = plt.subplots(len(tasks), 2, figsize=(13, 3.2*len(tasks)), squeeze=False)
    for row, task in enumerate(tasks):
        subset = summary.loc[summary.task == task]
        heads, budgets = list(subset["head"].unique()), list(subset.budget.unique())
        width = .8/len(budgets)
        for col, measure in enumerate(["test_mse_noiseless", "test_gradient_rmse"]):
            ax = axes[row, col]
            for b, budget in enumerate(budgets):
                values = subset.loc[subset.budget == budget].set_index("head").reindex(heads)
                ax.bar(np.arange(len(heads))+(b-(len(budgets)-1)/2)*width, values[measure+"_mean"],
                       width, label=budget, yerr=values[measure+"_std"].fillna(0), capsize=2)
            labels = [head+("†" if not subset.loc[subset["head"] == head, "parameter_budget_feasible"].all() else "") for head in heads]
            ax.set_xticks(np.arange(len(heads)), labels, rotation=35, ha="right")
            ax.set_ylabel("MSE to analytic mean" if col == 0 else "Input derivative RMSE")
            ax.set_title(task)
            ax.set_yscale("log")
            ax.legend()
    fig.suptitle("Analytic functions; bars = seed SD; † unmatched parameter reference; no face-detection claims")
    fig.tight_layout()
    fig.savefig(directory / "comparison.svg", format="svg", dpi=300)
    plt.close(fig)


def run_controlled(cfg) -> Path:
    experiment = cfg.experiment
    tasks = list(experiment.get("tasks", TASKS))
    if any(task not in TASKS for task in tasks) or not tasks:
        raise ValueError(f"Select one or more tasks from {TASKS}")
    seeds, heads, budgets = list(experiment.seeds), list(experiment.heads), list(experiment.budgets)
    lrs = list(cfg.train.lrs)
    if not seeds or not heads or not budgets or not lrs or any(b not in {"params", "time"} for b in budgets):
        raise ValueError("Need nonempty seeds/heads/lrs and valid parameter/time budgets")
    settings = OmegaConf.to_container(cfg, resolve=True)
    runtime=environment()
    dependencies={n:digest_file(Path.cwd()/n) for n in ["uv.lock","pyproject.toml"] if (Path.cwd()/n).is_file()}
    key = fingerprint({"configuration": settings, "source": source_fingerprint(), "runtime":runtime,
                       "dependencies":dependencies,"suite": "analytic-regression-v1"})
    directory = Path(cfg.paths.output_dir)/"functions"/key
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "protocol.json", {"configuration": settings, "source": source_fingerprint(),"runtime":runtime,
                                              "data_unit": "independent uniform points on [-1,1]^8", "known_functions": True,
                                              "scientific_face_evidence": False, "HPO": "same learning-rate candidates per head/function/budget/seed; validation MSE only",
                                              "gradient_evaluation": "underlying analytic derivative; excludes piecewise kink neighborhoods"})
    rows = []
    total = len(tasks)*len(heads)*len(budgets)*len(seeds)
    with tqdm(total=total, desc="Controlled functions") as progress:
        for task in tasks:
            sizes = tuple(int(experiment.get(key, default)) for key, default in
                          [("train_points", 2048), ("val_points", 512), ("test_points", 2048)])
            data = generate_data(task, sizes, int(experiment.get("data_seed", 2026)), float(experiment.get("noise_std", .1)))
            for head in heads:
                for budget in budgets:
                    for seed in seeds:
                        run = directory/"runs"/task/head/budget/f"seed{seed}"
                        run_key = fingerprint({"suite": key, "task": task, "head": head, "budget": budget, "seed": seed})
                        if complete(run, run_key):
                            rows.append(json.loads((run/"metrics.json").read_text(encoding="utf-8")))
                            progress.update(1)
                            continue
                        run.mkdir(parents=True, exist_ok=True)
                        selected, trials, selected_info, selected_budget = None, [], None, None
                        for lr in lrs:
                            seed_everything(int(seed))
                            model, budget_info = build_model(head, 8, budget, int(cfg.train.target_hidden),cfg.get("model"))
                            model, info = _fit(model, data, cfg.train, float(lr), int(seed), budget)
                            model.cpu()
                            trials.append({k: v for k, v in info.items() if k != "history"})
                            if selected_info is None or info["val_mse"] < selected_info["val_mse"]:
                                selected, selected_info, selected_budget = model, info, budget_info
                        selected.to(str(cfg.train.device)).eval()
                        test = _test(selected, data, task, str(cfg.train.device))
                        record = {"task": task, "head": head, "budget": budget, "seed": int(seed),
                                  **{k: v for k, v in selected_info.items() if k != "history"}, **test,
                                  "params": count_params(selected), "parameter_budget_feasible": selected_budget["feasible"],
                                  "noise_std": float(experiment.get("noise_std", .1)) if task == "noisy_additive" else 0.0}
                        atomic_json(run/"metrics.json", record)
                        atomic_json(run/"hpo.json", {"trials": trials, "budget": selected_budget})
                        pd.DataFrame(selected_info["history"]).to_csv(run/"learning_curve.csv", index=False)
                        torch.save({"state_dict": {k: v.cpu() for k, v in selected.state_dict().items()},
                                    "head": head, "input_features": 8, "budget": selected_budget}, run/"model.pt")
                        mark_complete(run, run_key, ["metrics.json", "hpo.json", "learning_curve.csv", "model.pt"])
                        rows.append(record)
                        progress.update(1)
    _aggregate(rows, directory)
    return directory