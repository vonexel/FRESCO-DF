from __future__ import annotations

import time
import copy
import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.special import expit
from .utils import seed_everything
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from .heads import make_head, match_parameters, count_params


def training_conditions(conditions=None):
    if isinstance(conditions, str):
        raise ValueError("Training conditions must be a list, not a string")
    selected = list(["clean"] if conditions is None else conditions)
    if not selected or len(set(selected)) != len(selected) or "clean" not in selected:
        raise ValueError("Training conditions must be nonempty, unique, and include clean")
    if any(not isinstance(value, str) or not value for value in selected):
        raise ValueError("Training conditions must contain nonempty strings")
    return selected


def train_mask(meta, held_out=None, conditions=None):
    selected = training_conditions(conditions)
    mask = (meta.dataset == "ffpp") & (meta.split == "train") & meta.condition.isin(selected)
    if held_out is not None:
        mask &= meta.method != held_out
    if not mask.any():
        raise ValueError("No FF++ training rows for the requested conditions; dataset must use dataset=ffpp")
    assert not (meta.loc[mask, "dataset"] != "ffpp").any()
    return mask.to_numpy()


def validate_training_rows(meta, held_out=None, conditions=None):
    selected = training_conditions(conditions)
    if ((meta.dataset != "ffpp") & (meta.split != "test")).any():
        raise ValueError("Target dataset must be TEST ONLY")
    source = meta.loc[(meta.dataset == "ffpp") & (meta.split == "train")]
    if held_out is not None:
        source = source.loc[source.method != held_out]
    mask = train_mask(meta, held_out, selected)
    chosen = meta.loc[mask]
    if chosen.label.nunique() != 2:
        raise ValueError("Training needs real and fake FF++ videos")
    for column in ["label", "method", "group_id"]:
        if column in source and source.groupby("video_id")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent {column} within a source training video")
    if "group_id" in meta:
        train_groups = set(chosen.group_id)
        other = meta.loc[(meta.dataset == "ffpp") & (meta.split != "train"), "group_id"]
        if train_groups.intersection(other):
            raise ValueError("Source group leakage across training and validation/test")
    if "frame_idx" not in source:
        if len(selected) > 1:
            raise ValueError("Augmented training requires frame_idx for paired-condition validation")
        return mask
    keys = ["video_id", "frame_idx"]
    if chosen.duplicated(keys + ["condition"]).any():
        raise ValueError("Duplicate source training frame-condition rows")
    expected = source[keys].drop_duplicates().set_index(keys).index
    counts = chosen.groupby(keys).condition.nunique().reindex(expected, fill_value=0)
    if not counts.eq(len(selected)).all():
        raise ValueError("Missing paired source training conditions; regenerate the augmented feature cache")
    return mask


def training_row_weights(meta):
    counts = meta.groupby("video_id").size()
    videos_per_class = meta.drop_duplicates("video_id").groupby("label").size()
    weights = np.array([1 / (counts[v] * videos_per_class[y]) for v, y in
                        zip(meta.video_id, meta.label)], dtype=np.float32)
    return weights / weights.mean()


class Coordinates:
    def fit(self, features, covariates, meta, k, use_covariates, held_out=None):
        mask = train_mask(meta, held_out)
        idx = np.flatnonzero(mask)
        if len(idx) > 50000:
            idx = np.sort(np.random.default_rng(2026).choice(idx, 50000, replace=False))
        x = np.asarray(features[idx], dtype=np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        if k == x.shape[1]:
            self.center, self.components = x.mean(0), np.eye(k, dtype=np.float32)
            z = x-self.center
            self.native = True
        elif min(x.shape) <= k:
            raise ValueError(f"PCA k={k} needs more than k independent training rows/dimensions")
        else:
            pca = PCA(k, svd_solver="randomized", random_state=2026).fit(x)
            self.center, self.components = pca.mean_, pca.components_
            z = (x-self.center) @ self.components.T
            self.native = False
        self.zmean, self.zstd = z.mean(0), np.maximum(z.std(0), 1e-6)
        cov = np.asarray(covariates[mask], dtype=np.float32)
        self.cmean, self.cstd = cov.mean(0), np.maximum(cov.std(0), 1e-6)
        self.clow, self.chigh = np.quantile(cov, [.01, .99], axis=0)
        self.use_covariates = bool(use_covariates)
        self.train_videos = sorted(meta.loc[mask, "video_id"].unique().tolist())
        return self

    def transform(self, features, covariates):
        result = []
        for begin in range(0, len(features), 16384):
            x = np.array(features[begin:begin+16384], dtype=np.float32)
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
            projected = x-self.center if self.native else (x-self.center) @ self.components.T
            z = (projected-self.zmean)/self.zstd
            if self.use_covariates:
                z = np.concatenate((z, (covariates[begin:begin+16384]-self.cmean)/self.cstd), axis=1)
            result.append(z.astype(np.float32))
        return np.concatenate(result)

    def save(self, path):
        np.savez(path, **{k: v for k, v in vars(self).items() if k != "train_videos"})


@torch.inference_mode()
def predict(model, x, device, batch=4096):
    model.eval()
    return np.concatenate([model(torch.as_tensor(x[i:i+batch], device=device)).float().cpu().numpy().reshape(-1) for i in range(0, len(x), batch)])


def video_predictions(meta, logits, temperature=1.0):
    m = meta.copy()
    m["probability"] = expit(logits)
    m["calibrated"] = expit(logits/temperature)
    m["logit"] = logits
    keys = ["dataset", "condition", "video_id", "label", "split", "method", "scenario", "group_id"]
    return m.groupby(keys, as_index=False, dropna=False).agg(probability=("probability", "mean"), calibrated=("calibrated", "mean"),
                                                             logit=("logit", "mean"), frames=("probability", "size"))


def validation_mask(meta, held_out=None):
    mask = (meta.dataset == "ffpp") & (meta.split == "val") & (meta.condition == "clean")
    if held_out is not None:
        mask &= meta.method != held_out
    if not mask.any() or meta.loc[mask, "label"].nunique() != 2:
        raise ValueError("Need disjoint FF++ validation real and seen-method fake videos")
    return mask.to_numpy()


def build_model(name, dim, budget, hidden, head_kwargs=None, target_params=None):
    if target_params is not None and (isinstance(target_params, bool)
                                      or not isinstance(target_params, (int, np.integer)) or target_params < 1):
        raise ValueError("target_params must be a positive integer or null")
    options={"grid_size":5,"spline_degree":3,"grid_min":-3.0,"grid_max":3.0}
    if head_kwargs is not None:
        options.update(dict(head_kwargs))
    if budget == "params":
        target = (count_params(make_head("mlp_relu", dim, hidden, **options))
                  if target_params is None else int(target_params))
        model, metadata = match_parameters(name, dim, target, **options)
        metadata["head_kwargs"] = options
        metadata["target_source"] = "reference MLP at this input dimension" if target_params is None else "explicit common budget"
        return model, metadata
    if budget != "time":
        raise ValueError(f"Unknown budget {budget}; choose params or time")
    model = make_head(name, dim, hidden, **options)
    return model, {"hidden": hidden, "params": count_params(model), "feasible": True,
                   "head_kwargs": options,
                   "criterion": "equal training wall clock; parameter count unconstrained"}


def fit_head(model, x, meta, cfg, lr, seed, budget, held_out=None, progress=True):
    seed_everything(seed)
    device = str(cfg.device)
    is_cuda=torch.device(device).type=="cuda"
    if is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Set train.device=cpu explicitly for CPU tests.")
    torch.set_num_threads(int(cfg.threads))
    model = model.to(device)
    initial_state = copy.deepcopy(model.state_dict())
    conditions = training_conditions(cfg.get("conditions"))
    tm = validate_training_rows(meta, held_out, conditions)
    vm = validation_mask(meta, held_out)
    tx = torch.as_tensor(x[tm], device=device)
    ty = torch.as_tensor(meta.loc[tm, "label"].to_numpy(np.float32), device=device)
    trmeta, valmeta = meta.loc[tm], meta.loc[vm]
    weights = training_row_weights(trmeta)
    weights = torch.as_tensor(weights, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg.weight_decay))
    best, best_state, stale, log = -np.inf, None, 0, []
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start, epoch = time.perf_counter(), 0
    max_epochs = int(cfg.epochs) if budget == "params" else 1000000
    bar = tqdm(total=max_epochs if budget == "params" else None, disable=not progress,
               desc=f"{budget} seed={seed}", leave=False)
    stop = False
    while epoch < max_epochs and not stop:
        model.train()
        order = torch.randperm(len(tx), device=device)
        losses = []
        for begin in range(0, len(tx), int(cfg.batch_size)):
            batch = order[begin:begin+int(cfg.batch_size)]
            opt.zero_grad(set_to_none=True)
            logits = model(tx[batch]).reshape(-1)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, ty[batch], reduction="none")
                    * weights[batch]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; failed run retained")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
            if budget == "time" and time.perf_counter()-start >= float(cfg.wall_seconds):
                stop = True
                break
        vp = video_predictions(valmeta, predict(model, x[vm], device))
        auc = roc_auc_score(vp.label, vp.probability)
        elapsed = time.perf_counter()-start
        log.append({"epoch": epoch, "loss": np.mean(losses), "val_video_auc": auc, "wall_s": elapsed})
        if auc > best:
            best, best_state, stale = auc, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        epoch += 1
        bar.update(1)
        if budget == "params" and stale >= int(cfg.patience):
            break
        if budget == "time" and elapsed >= float(cfg.wall_seconds):
            break
    bar.close()
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    model.load_state_dict(best_state)
    return model, initial_state, {"validation_auc": best, "wall_s": elapsed,
                                  "requested_wall_s": float(cfg.wall_seconds) if budget == "time" else None,
                                  "time_budget_relative_error": abs(elapsed-float(cfg.wall_seconds))/float(cfg.wall_seconds) if budget == "time" else None,
                                  "epochs": epoch, "training_conditions": conditions, "training_rows": len(trmeta),
                                  "training_videos": int(trmeta.video_id.nunique()),
                                  "validation_selection": "FF++ clean seen-method video AUROC only",
                                  "weighting": "equal classes and equal videos within class; all selected conditions share video mass",
                                  "peak_allocated_mb": torch.cuda.max_memory_allocated(device)/2**20 if is_cuda else None,
                                  "log": log}