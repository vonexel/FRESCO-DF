from __future__ import annotations

import itertools
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .statistics import paired_auc_bootstrap
from .provenance import report_configuration, validate_run_provenance
from .evaluation import COVARIATE_NAMES
from .training import validate_covariate_indices
from .utils import atomic_json, digest_file, fingerprint, complete, mark_complete, source_fingerprint
from omegaconf import OmegaConf


plt.rcParams.update({"svg.fonttype": "none", "figure.dpi": 150, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _save(fig, directory, name, data):
    directory.mkdir(parents=True, exist_ok=True)
    tags=[]
    for column in ["k","covariates","held_out","budget"]:
        if column in data:
            values=data[column].fillna("none").astype(str).unique()
            if len(values)<=4:
                tags.append(column+"="+",".join(values))
    if tags:
        fig.text(.01,-.02,"; ".join(tags),fontsize=7)
    fig.savefig(directory/f"{name}.svg", format="svg", dpi=300, bbox_inches="tight")
    qa=directory.parent/"qa"
    qa.mkdir(exist_ok=True)
    fig.savefig(qa/f"{name}.png",dpi=120,bbox_inches="tight")
    plt.close(fig)
    data.to_csv(directory/f"{name}.csv", index=False)


def collect(out, synthetic=None):
    records, runs = [], []
    for path in sorted(out.rglob("metrics.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if not (path.parent / "DONE.json").exists():
            continue
        marker=json.loads((path.parent/"DONE.json").read_text(encoding="utf-8"))
        if not complete(path.parent,marker["key"]):
            raise ValueError(f"Corrupted/incomplete run artifacts: {path.parent}; rerun training to repair")
        run = r["run"]
        compute = r.get("compute", {})
        latency = compute.get("latency_batch256", {})
        if not isinstance(latency, dict):
            latency = {}
        common = {**run, "run_dir": str(path.parent.relative_to(out)),
                  "params": r["matching"].get("params"),
                  "budget_feasible": r["matching"].get("feasible", False),
                  "head_latency_ms": latency.get("p50_ms"), "train_wall_s": r["training"]["wall_s"],
                  "val_auc": r["training"]["validation_auc"]}
        if synthetic is not None:
            validate_run_provenance([common], {"synthetic": synthetic})
        runs.append({**common, "details": r})
        for row in r["eval"]:
            records.append({**common, **row})
    if not records:
        raise RuntimeError("No completed evaluation artifacts to aggregate")
    results = pd.DataFrame(records)
    results.to_csv(out/"results.csv", index=False)
    group = ["head", "budget", "k", "covariates", "held_out", "dataset", "condition", "level", "method", "calibration"]
    summary = results.groupby(group, dropna=False).agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        seeds=("seed", "nunique"), params=("params", "mean"), head_latency_ms=("head_latency_ms", "mean"),
        train_wall_s=("train_wall_s", "mean"), ece_mean=("ece", "mean"),
        brier_mean=("brier", "mean"), nll_mean=("nll", "mean")).reset_index()
    summary.to_csv(out/"results_summary.csv", index=False)
    novelty=[]
    symbolic=[]
    for r in runs:
        for measurement in r["details"].get("openset",{}).get("metrics",[]):
            novelty.append({k:v for k,v in r.items() if k!="details"}|measurement)
        surrogate=r["details"].get("interp",{}).get("symbolic",{})
        for measurement in surrogate.get("test_video_fidelity",[]):
            symbolic.append({k:v for k,v in r.items() if k!="details"}|measurement|
                            {"degree":surrogate.get("degree"),"test_logit_rmse":surrogate.get("test_rmse")})
    pd.DataFrame(novelty).to_csv(out/"openset_results.csv",index=False)
    pd.DataFrame(symbolic).to_csv(out/"symbolic_fidelity.csv",index=False)
    return results, summary, runs


def response_audit_table(out, runs):
    rows = []
    identifiers = ["head", "budget", "k", "covariates", "held_out", "seed", "run_dir"]
    for run in runs:
        audit = run["details"].get("interp", {}).get("response_audit", {})
        measurements = audit.get("curve_response", {}).get("by_dataset", [])
        for measurement in measurements:
            rows.append({**{name: run.get(name) for name in identifiers}, **measurement})
    columns = identifiers + ["dataset", "status", "coverage", "coverage_complete", "gap", "rmse",
                             "total_rms_covered", "total_rms_all", "sign_agreement"]
    result = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    result.to_csv(out / "response_audit_summary.csv", index=False)
    return result


def _analysis_role(cfg):
    role = cfg.experiment.get("analysis_role")
    if role:
        return str(role)
    if set(cfg.get("train", {}).get("conditions") or ["clean"]) != {"clean"}:
        return "Changed source training recipe; original clean-only hypothesis criteria are diagnostic"
    return None


def _aligned_scores(out, selected, dataset, condition):
    arrays, base = [], None
    for run in sorted(selected, key=lambda r: r["seed"]):
        scores = pd.read_csv(out/run["run_dir"]/"scores.csv", keep_default_na=False)
        scores = scores[(scores.dataset == dataset) & (scores.condition == condition) & (scores.split == "test")]
        scores = scores.set_index("video_id").sort_index()
        if scores.index.duplicated().any():
            raise ValueError("Duplicate video scores")
        if base is None:
            base = scores
        elif not scores.index.equals(base.index) or not np.array_equal(scores.label, base.label):
            raise ValueError("Runs have different test video sets; cannot use paired inference")
        arrays.append(scores.probability.to_numpy())
    return base, np.stack(arrays)


def agreement_test(out, runs, cfg):
    chosen = validate_covariate_indices(cfg.train.get("covariate_indices")) if "train" in cfg else np.arange(4)
    if not np.array_equal(chosen, np.arange(4)):
        return {"status": "NOT_APPLICABLE", "reason": "Historical primary contrast requires all four covariates; this is a subset ablation"}
    cell=[r for r in runs if r["k"]==16 and r["covariates"] and r["budget"]=="params"
          and r["held_out"] is None and r["head"] in {"kan_spline","mlp_gelu"}]
    a=[r for r in cell if r["head"]=="kan_spline"]
    b=[r for r in cell if r["head"]=="mlp_gelu"]
    result={"status":"INCONCLUSIVE","contrast":"kan_spline minus mlp_gelu, k16, quality covariates, params",
            "endpoint":"mean seed-specific sqrt(mean_video(error_sse/nframes) / mean_video(total_sse/nframes))",
            "dataset":"celebdfpp","conditions":"all predeclared paired image degradations"}
    if not a or not b or {r["seed"] for r in a}!={r["seed"] for r in b}:
        return {**result,"reason":"Primary configuration absent or incomplete"}
    errors,totals,groups,ids=[],[],None,None
    for chosen in [a,b]:
        es,ts=[],[]
        for r in sorted(chosen,key=lambda r:r["seed"]):
            d=pd.read_csv(out/r["run_dir"]/"intervention.csv")
            d=d[d.dataset=="celebdfpp"]
            if not len(d):
                return {**result,"reason":"No paired target interventions"}
            d=d.groupby(["video_id","group_id"],as_index=False)[["error_sse","total_sse","frame_count"]].sum().sort_values("video_id")
            if ids is None:
                ids=d.video_id.to_list()
                groups=d.group_id.to_numpy()
            elif ids!=d.video_id.to_list():
                return {**result,"reason":"Different paired video sets across models/seeds"}
            es.append(d.error_sse.to_numpy()/d.frame_count.to_numpy())
            ts.append(d.total_sse.to_numpy()/d.frame_count.to_numpy())
        errors.append(np.stack(es))
        totals.append(np.stack(ts))

    def statistic(idx):
        denominator=[t[:,idx].mean(axis=1) for t in totals]
        if any(np.any(d<1e-10) for d in denominator):
            return None
        gaps=[np.sqrt(e[:,idx].mean(axis=1)/d).mean() for e,d in zip(errors,denominator)]
        return gaps
    point=statistic(np.arange(len(ids)))

    if point is None:
        return {**result,"reason":"Near-zero measured total response; relative agreement is undefined"}
    unique=np.unique(groups)

    if len(unique)<5:
        return {**result,"reason":"Fewer than five identity/source clusters"}
    blocks=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(2026)
    samples=[]

    for _ in range(int(cfg.eval.bootstrap)):
        idx=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        value=statistic(idx)
        if value is not None:
            samples.append(value[0]-value[1])
    if len(samples)<.9*int(cfg.eval.bootstrap):
        return {**result,"reason":"Degenerate response-bootstrap denominators"}
    ci=np.quantile(samples,[.025,.975]).tolist()
    _,aa=_aligned_scores(out,a,"celebdfpp","clean")
    labels,bb=_aligned_scores(out,b,"celebdfpp","clean")
    discrimination=paired_auc_bootstrap(labels.label,aa,bb,labels.group_id,
                        resamples=int(cfg.eval.bootstrap),margin=float(cfg.eval.margin))

    _,kval=_curves(out,a,cfg)
    defined=kval.median_seed_correlation.dropna()
    stability=float(defined.median()) if len(defined) else None
    coverage_complete=bool(len(kval)==len(COVARIATE_NAMES) and kval.coverage_complete.all())
    coverage={"complete":coverage_complete,"expected_covariates":len(COVARIATE_NAMES),
              "complete_covariates":int(kval.coverage_complete.sum()),
              "expected_pairs":int(kval.expected_pairs.sum()),"defined_pairs":int(kval.pairs.sum()),
              "features":[{"feature":row.feature,"expected_seed_curves":int(row.expected_seed_curves),
                           "valid_seed_curves":int(row.valid_seed_curves),
                           "constant_seed_curves":int(row.constant_seed_curves),
                           "expected_pairs":int(row.expected_pairs),"defined_pairs":int(row.pairs),
                           "complete":bool(row.coverage_complete),
                           "reasons":json.loads(row.coverage_reasons)} for row in kval.itertuples()]}
    result.update({"gap_kan":point[0],"gap_mlp":point[1],"difference":point[0]-point[1],"ci95":ci,
                   "clusters":len(unique),"seeds":len(a),"conditional_fixed_seeds":True,
                   "auc_noninferiority":discrimination,"median_curve_stability":stability,
                   "curve_stability_coverage":coverage,
                   "preregistered_gap_threshold":.3})
    if not coverage_complete:
        return {**result,"reason":"Incomplete or undefined curve stability; see curve_stability_coverage"}
    feasible=all(r["budget_feasible"] for r in a+b)
    scientific=(not cfg.experiment.synthetic and len(a)>=int(cfg.eval.min_seeds_confirmatory))
    pass_conditions=(feasible and ci[1]<0 and point[0]<=.3 and stability is not None and stability>=.9
                     and discrimination.get("ci95",[-1])[0]>-float(cfg.eval.margin))
    if scientific:
        result["status"]="SUPPORTED_CONDITIONAL_ON_SEEDS" if pass_conditions else "NOT_SUPPORTED"
        result["reason"]="Joint predeclared agreement, noninferiority, stability and budget criteria"
    else:
        result["reason"]="Pilot/synthetic run; no confirmatory verdict"
    return result


def hypothesis_tests(out, runs, cfg):
    comparisons, selection = [], []
    keys = sorted({(r["budget"], r["k"], r["covariates"], str(r["held_out"])) for r in runs})
    for budget, k, cov, held in keys:
        cell = [r for r in runs if (r["budget"], r["k"], r["covariates"], str(r["held_out"])) == (budget,k,cov,held)]
        bases = [r for r in cell if r["head"] in {"mlp_relu", "mlp_gelu", "mlp_bspline"}]
        if not bases:
            continue
        mean_val = pd.DataFrame(bases).groupby("head").val_auc.mean()
        comparator = mean_val.idxmax()
        reference = [r for r in bases if r["head"] == comparator]
        selection.append({"budget": budget, "k": k, "covariates": cov, "held_out": held,
                          "comparator": comparator, "criterion": "mean FF++ seen validation AUC"})
        for head in sorted({r["head"] for r in cell if r["head"].startswith("kan_")}):
            selected = [r for r in cell if r["head"] == head]
            info = {"head": head, "comparator": comparator, "budget": budget, "k": k,
                    "covariates": cov, "held_out": held, "status": "INCONCLUSIVE"}
            if {r["seed"] for r in selected} != {r["seed"] for r in reference}:
                info["reason"] = "Incomplete paired seed sets"
                comparisons.append(info)
                continue
            declared = cfg.experiment.get("seeds")
            if declared is not None:
                expected = set(int(seed) for seed in declared)
                if ({r["seed"] for r in selected} != expected
                        or len(selected) != len(expected) or len(reference) != len(expected)):
                    info["reason"] = "Incomplete or duplicate declared seed sets"
                    comparisons.append(info)
                    continue
            a, aa = _aligned_scores(out, selected, "celebdfpp", "clean")
            b, bb = _aligned_scores(out, reference, "celebdfpp", "clean")
            if not a.index.equals(b.index):
                raise ValueError("Unpaired model video predictions")
            test = paired_auc_bootstrap(a.label, aa, bb, groups=a.group_id,
                     resamples=int(cfg.eval.bootstrap), margin=float(cfg.eval.margin))
            info.update(test)
            seed_n = len(selected)
            feasible = all(r["budget_feasible"] for r in selected+reference)
            if budget == "time":
                feasible &= all(r["details"]["training"].get("time_budget_relative_error", 1) <= .10
                                for r in selected+reference)
            info["budget_feasible"] = feasible
            info["seeds"] = seed_n
            info["multiple_comparisons"] = "exploratory; primary prespecified kan_spline, k16, covariates=true, params"
            if cfg.experiment.synthetic:
                info["reason"] = "Synthetic integration fixture; no scientific verdict"
            elif seed_n < int(cfg.eval.min_seeds_confirmatory):
                info["reason"] = "Pilot: fewer than article's ten seeds; conditional video CIs only"
            elif not feasible:
                info["reason"] = "Requested budget match not achieved"
            elif test.get("equivalent_tost_percentile"):
                info["status"] = "EQUIVALENT_CONDITIONAL_ON_SEEDS"
            else:
                info["reason"] = "Equivalence not demonstrated; this does not establish inequality"
            comparisons.append(info)
    result = {"primary_paired_response": agreement_test(out,runs,cfg),
              "H2a": comparisons, "comparator_selection": selection,
              "H2b": {"status": "EXPLORATORY", "reason": "Report paired degradation scores; no unadjusted multiple-testing PASS"},
              "H2c": {"status": "INCONCLUSIVE", "reason": "PD surrogate is not edge replacement; inspect stability and actual surrogate fidelity separately"},
              "H2c_intervention": {"status": "MEASURED_NO_AUTOMATIC_VERDICT", "reason": "Agreement gap tests partial explanation coverage, not causality"},
              "H2d": {"status": "EXPLORATORY", "reason": "Adapted first-layer plasticity; dataset/method shift is confounded"},
              "H2e": {"status": "EXPLORATORY", "reason": "Dimension association requires k_ablation; no causal dimensionality claim"},
              "inference": "Mean seed-specific AUC differences; source/target identity cluster bootstrap conditional on fixed seeds. Secondary co-source dependence remains.",
              "TOST": "alpha=.05, 90% percentile CI strictly inside +/-margin; superiority uses 95% CI"}
    role = _analysis_role(cfg)
    if role:
        result["analysis_role"] = str(role)
        for assessment in [result["primary_paired_response"], *result["H2a"]]:
            assessment["diagnostic_criteria_status"] = assessment["status"]
            assessment["status"] = "EXPLORATORY"
            assessment["reason"] = ("Follow-up recipe after target inspection; original criteria are diagnostics, "
                                    "not a new confirmatory test. " + assessment.get("reason", ""))
    atomic_json(out/"stats.json", result)
    return result


def _curves(out, runs, cfg=None):
    rows, stability, frames, load_errors, cells = [], [], {}, {}, {}
    cellcols=["head", "budget", "k", "covariates", "held_out"]
    for run in runs:
        if not run["covariates"]:
            continue
        cells.setdefault(tuple(run[c] for c in cellcols),[]).append(run)
        path = out/run["run_dir"]/"pd_curves.csv"
        try:
            frame = pd.read_csv(path)
            response=next((c for c in ["response","mean","delta"] if c in frame),None)
            grid="grid" if "grid" in frame else "value"
            if response is None or not {"feature","grid_type",grid}.issubset(frame):
                raise ValueError("missing feature/grid_type/grid/response columns")
            frame=frame.loc[frame.grid_type=="shared_standardized"].copy()
            frame["grid"]=pd.to_numeric(frame[grid],errors="coerce")
            frame["response"]=pd.to_numeric(frame[response],errors="coerce")
            frames[run["run_dir"]]=frame
            for key in cellcols+["seed"]:
                frame[key]=run[key]
            rows.append(frame)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            load_errors[run["run_dir"]]=str(exc)
    allrows=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    declared_seeds=cfg.experiment.get("seeds") if cfg is not None else None
    for keys, selected in cells.items():
        seeds=sorted(set(int(s) for s in (declared_seeds if declared_seeds is not None
                                         else [r["seed"] for r in selected])))
        expected_pairs=len(seeds)*(len(seeds)-1)//2
        common_grid=(np.linspace(-2,2,int(cfg.eval.get("grid_points",31))) if cfg is not None else None)
        unexpected=sorted(set(r["seed"] for r in selected)-set(seeds))
        selected_indices = (validate_covariate_indices(cfg.train.get("covariate_indices"))
                            if cfg is not None and "train" in cfg else np.arange(4))
        for feature in [COVARIATE_NAMES[int(i)] for i in selected_indices]:
            curves, reasons, constants={},[],0
            if len(seeds)<2:
                reasons.append("fewer than two declared seeds")
            if unexpected:
                reasons.append(f"unexpected seeds: {unexpected}")
            for seed in seeds:
                matches=[r for r in selected if r["seed"]==seed]
                if len(matches)!=1:
                    reasons.append(f"seed {seed}: expected one run, found {len(matches)}")
                    continue
                run=matches[0]
                if run["run_dir"] not in frames:
                    reasons.append(f"seed {seed}: unavailable curves ({load_errors.get(run['run_dir'],'missing')})")
                    continue
                frame=frames[run["run_dir"]]
                curve=frame.loc[frame.feature==feature].sort_values("grid")
                grid=curve.grid.to_numpy()
                response=curve.response.to_numpy()
                if len(grid)<2 or not np.isfinite(grid).all() or not np.isfinite(response).all():
                    reasons.append(f"seed {seed}: missing or nonfinite curve")
                    continue
                if curve.grid.duplicated().any():
                    reasons.append(f"seed {seed}: duplicate grid points")
                    continue
                if common_grid is None:
                    common_grid=grid
                if len(grid)!=len(common_grid) or not np.allclose(grid,common_grid,rtol=0,atol=1e-12):
                    reasons.append(f"seed {seed}: incomplete or different shared grid")
                    continue
                if np.std(response)<=1e-10:
                    constants+=1
                    reasons.append(f"seed {seed}: constant curve")
                    continue
                curves[seed]=response
            corrs=[float(np.corrcoef(curves[a],curves[b])[0,1])
                   for a,b in itertools.combinations(curves,2)]
            if not np.isfinite(corrs).all():
                reasons.append("nonfinite pair correlation")
                corrs=[v for v in corrs if np.isfinite(v)]
            complete_coverage=not reasons and len(corrs)==expected_pairs and expected_pairs>0
            stability.append({**dict(zip(cellcols,keys)),"feature":feature,
                              "median_seed_correlation":np.median(corrs) if corrs else np.nan,
                              "pairs":len(corrs),"expected_pairs":expected_pairs,
                              "expected_seed_curves":len(seeds),"valid_seed_curves":len(curves),
                              "constant_seed_curves":constants,"constant_curves":constants>0,
                              "coverage_complete":complete_coverage,"coverage_reasons":json.dumps(reasons)})
    return allrows, pd.DataFrame(stability)


def figures(out, results, summary, runs, cfg=None):
    directory = out/"figures"
    all_primary = summary[(summary.dataset == "celebdfpp") & (summary.condition == "clean") &
                      (summary.level == "video") & (summary.method == "all") & (summary.calibration == "raw")]
    display_k=16 if 16 in set(summary.k) else int(summary.k.min())
    display_cov=True if True in set(summary.covariates) else False
    cell=summary[(summary.k==display_k)&(summary.covariates==display_cov)]
    if cell.held_out.isna().any():
        cell=cell[cell.held_out.isna()]
        display_held=None
    else:
        display_held=sorted(cell.held_out.astype(str).unique())[0]
        cell=cell[cell.held_out.astype(str)==display_held]
    primary = cell[(cell.dataset == "celebdfpp") & (cell.condition == "clean") &
                   (cell.level == "video") & (cell.method == "all") & (cell.calibration == "raw")]
    display_budget="params" if "params" in set(cell.budget) else str(cell.budget.iloc[0])
    onebudget=cell[cell.budget==display_budget]
    display_runs=[r for r in runs if r["k"]==display_k and r["covariates"]==display_cov and
                  r["held_out"]==display_held and r["budget"]==display_budget]
    if len(primary):
        fig, ax = plt.subplots(figsize=(10,4))
        for budget, frame in primary.groupby("budget"):
            frame = frame.sort_values("head")
            ax.errorbar(frame["head"], frame.auc_mean, yerr=frame.auc_std.fillna(0), marker="o", label=budget, capsize=3)
        ax.set(ylabel="Video AUROC (mean ± seed SD)", title="Celeb-DF++ transfer")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
        _save(fig,directory,"auc_celebdfpp_by_head",primary)
        fig, ax = plt.subplots(figsize=(7,4))
        for _, row in primary.iterrows():
            if np.isfinite(row.head_latency_ms):
                ax.scatter(row.head_latency_ms,row.auc_mean,s=max(20,np.sqrt(row.params)))
                ax.annotate(row["head"]+"/"+row.budget,(row.head_latency_ms,row.auc_mean),fontsize=6)
        ax.set(xlabel="Head latency per batch of 256 (ms, fp32)",ylabel="Video AUROC")
        _save(fig,directory,"pareto_auc_latency",primary)
    bymethod = onebudget[(onebudget.dataset == "celebdfpp") & (onebudget.condition == "clean") &
              (onebudget.level == "video") & (onebudget.method != "all") & (onebudget.calibration == "raw")]
    if len(bymethod):
        mat = bymethod.pivot_table(index="head",columns="method",values="auc_mean")
        fig,ax = plt.subplots(figsize=(max(7,len(mat.columns)*.42),4))
        im=ax.imshow(mat,aspect="auto",vmin=0,vmax=1)
        ax.set_xticks(range(len(mat.columns)),mat.columns,rotation=90)
        ax.set_yticks(range(len(mat.index)),mat.index)
        fig.colorbar(im,ax=ax,label="AUROC with shared real controls")
        _save(fig,directory,"auc_by_method_heatmap",bymethod)
    robust = onebudget[(onebudget.level == "video") & (onebudget.method == "all") & (onebudget.calibration == "raw")]
    datasets=sorted(robust.dataset.unique())
    fig,axes=plt.subplots(len(datasets),3,figsize=(12,3.3*len(datasets)),squeeze=False)
    families=[("JPEG Q",[("clean",100),("jpeg90",90),("jpeg70",70),("jpeg50",50),("jpeg30",30)]),
              ("Blur sigma",[("clean",0),("blur1",1),("blur2",2),("blur3",3)]),
              ("Resize factor",[("resize0.25",.25),("resize0.5",.5),("clean",1)])]
    for row,dataset in enumerate(datasets):
        for ax,(label,conditions) in zip(axes[row],families):
            available=set(robust.loc[robust.dataset==dataset,"condition"])
            if not any(condition!="clean" and condition in available for condition,_ in conditions):
                ax.text(.5,.5,"No paired condition in this run",ha="center",va="center",transform=ax.transAxes)
                ax.set_axis_off()
                continue
            for head,d in robust[robust.dataset==dataset].groupby("head"):
                lookup=d.set_index("condition").auc_mean
                points=[(value,lookup[condition]) for condition,value in conditions if condition in lookup]
                if points:
                    ax.plot([p[0] for p in points],[p[1] for p in points],marker=".",label=head)
            ax.set(xlabel=label,ylabel="Video AUROC",title=dataset)
    handles,labels=axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.98),fontsize=7,ncols=4)
    fig.text(.01,.99,"JPEG x=100 denotes native clean crop, not an uncompressed original",fontsize=7)
    fig.tight_layout(rect=[0,0,1,.87])
    _save(fig,directory,"robustness_curves",robust)
    _,stability=_curves(out,runs,cfg)
    pdrows,_=_curves(out,display_runs,cfg)
    stability.to_csv(out/"stability.csv",index=False)
    if len(pdrows) and "feature" in pdrows:
        response=next((c for c in ["response","mean","delta"] if c in pdrows),None)
        grid="grid" if "grid" in pdrows else "value"
        if response:
            feats=pdrows.feature.unique()
            heads=sorted(pdrows["head"].unique())
            fig,axes=plt.subplots(len(heads),len(feats),figsize=(3.5*len(feats),2.1*len(heads)),squeeze=False)
            for row,head in enumerate(heads):
                for col,feature in enumerate(feats):
                    ax=axes[row,col]
                    selected=pdrows[(pdrows.feature==feature)&(pdrows["head"]==head)]
                    for seed,frame in selected.groupby("seed"):
                        f=frame.groupby(grid,as_index=False)[response].mean()
                        ax.plot(f[grid],f[response],label=f"seed {seed}")
                    if row==0:
                        ax.set_title(str(feature),fontsize=8)
                    if col==0:
                        ax.set_ylabel(head+"\nlogit change",fontsize=8)
                    if row==len(heads)-1:
                        ax.set_xlabel("Standardized observed covariate")
            axes[0,0].legend(fontsize=7)
            fig.tight_layout()
            _save(fig,directory,"response_curves_covariates",pdrows)
    if len(stability):
        stability_display=stability[(stability.k==display_k)&(stability.covariates==display_cov)&
                                    (stability.budget==display_budget)]
        stability_display=(stability_display[stability_display.held_out.isna()] if display_held is None else
                           stability_display[stability_display.held_out==display_held])
        fig,ax=plt.subplots(figsize=(8,3))
        for i,(head,frame) in enumerate(stability_display.groupby("head")):
            ax.scatter([i]*len(frame),frame.median_seed_correlation,label=head)
        ax.axhline(.9,color="black",ls="--",lw=.8)
        ax.set_xticks(range(stability_display["head"].nunique()),sorted(stability_display["head"].unique()),rotation=30)
        ax.set(ylabel="Median pairwise seed correlation",ylim=(-1.05,1.05),title=f"k={display_k}, covariates={display_cov}, {display_budget}")
        _save(fig,directory,"stability",stability_display)
    interventions=[]
    for r in display_runs:
        p=out/r["run_dir"]/"intervention.csv"
        if p.exists():
            d=pd.read_csv(p)
            if len(d):
                for key in ["head","seed","budget","k","covariates","held_out"]:
                    d[key]=r[key]
                interventions.append(d)
    if interventions:
        joined=pd.concat(interventions,ignore_index=True)
        joined.to_csv(out/"intervention_all.csv",index=False)
        if {"total","covariate"}.issubset(joined):
            f=joined.groupby(["head","dataset","condition"],as_index=False)[["total","covariate"]].mean()
            f["k"],f["covariates"],f["held_out"],f["budget"]=display_k,display_cov,display_held,display_budget
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for (head,dataset),d in f.groupby(["head","dataset"]):
                axes[0].plot(d.condition,d.total,marker=".",label=f"{head}/{dataset}")
                axes[1].plot(d.condition,d.covariate,marker=".",label=f"{head}/{dataset}")
            for ax,title in zip(axes,["Real paired image transformation","Covariate-only partial response"]):
                ax.set(title=title,ylabel="Mean logit change")
                ax.tick_params(axis="x",rotation=45)
            axes[0].legend(fontsize=6)
            _save(fig,directory,"intervention_response",f)
    calibrations=[]
    for r in display_runs:
        p=out/r["run_dir"]/"calibration.csv"
        if p.exists():
            d=pd.read_csv(p)
            if len(d):
                for key in ["head","seed","budget","k","covariates","held_out"]:
                    d[key]=r[key]
                calibrations.append(d)
    if calibrations:
        d=pd.concat(calibrations,ignore_index=True)
        d=d[(d.dataset=="celebdfpp")&(d.condition=="clean")&(d.level=="video")&(d.calibration=="temperature")]
        fig,ax=plt.subplots(figsize=(6,4))
        if {"confidence","frequency"}.issubset(d):
            for head,f in d.groupby("head"):
                ax.scatter(f.confidence,f.frequency,s=10,alpha=.35,label=head)
        ax.plot([0,1],[0,1],"k--",lw=.7)
        ax.set(xlabel="Mean predicted fake probability",ylabel="Observed fake frequency")
        _save(fig,directory,"calibration_reliability",d)
    # Dimension and covariate ablations retain all conditions in companion CSVs.
    for variable,name in [("k","dimension_ablation"),("covariates","covariate_ablation")]:
        if len(all_primary) and all_primary[variable].nunique()>1:
            fig,ax=plt.subplots(figsize=(7,4))
            ablation=all_primary[(all_primary.budget==display_budget)&all_primary.held_out.isna()]
            for head,d in ablation.groupby("head"):
                f=d.groupby(variable,as_index=False).auc_mean.mean()
                ax.plot(f[variable],f.auc_mean,marker="o",label=head)
            ax.set(xlabel=variable,ylabel="Celeb-DF++ video AUROC")
            ax.legend(fontsize=7)
            _save(fig,directory,name,ablation)
    novelty=pd.read_csv(out/"openset_results.csv") if (out/"openset_results.csv").stat().st_size>2 else pd.DataFrame()
    if len(novelty):
        n=novelty[(novelty.k==display_k)&(novelty.covariates==display_cov)&
                  (novelty.budget==display_budget)&(novelty.condition=="clean")]
        n=n[n.held_out.isna()] if display_held is None else n[n.held_out==display_held]
        fig,axes=plt.subplots(1,max(1,n.comparison.nunique()),figsize=(6*max(1,n.comparison.nunique()),4),squeeze=False)
        for ax,(comparison,d) in zip(axes.flat,n.groupby("comparison")):
            means=d.groupby(["head","score"],as_index=False).auc.mean()
            for score,f in means.groupby("score"):
                ax.plot(f["head"],f.auc,marker="o",label=score)
            ax.set(title=comparison,ylabel="Novelty AUROC (seed mean)")
            ax.tick_params(axis="x",rotation=45)
            ax.axhline(.5,color="gray",ls="--",lw=.7)
            ax.legend(fontsize=6)
        _save(fig,directory,"openset_auroc",n)
    symbolic=pd.read_csv(out/"symbolic_fidelity.csv") if (out/"symbolic_fidelity.csv").stat().st_size>2 else pd.DataFrame()
    if len(symbolic):
        d=symbolic[(symbolic.k==display_k)&(symbolic.covariates==display_cov)&
                   (symbolic.budget==display_budget)&(symbolic.dataset=="celebdfpp")&(symbolic.condition=="clean")]
        d=d[d.held_out.isna()] if display_held is None else d[d.held_out==display_held]
        fig,ax=plt.subplots(figsize=(8,3.5))
        for i,(head,f) in enumerate(d.groupby("head")):
            ax.scatter([i]*len(f),f.auc_drop)
        ax.set_xticks(range(d["head"].nunique()),sorted(d["head"].unique()),rotation=30)
        ax.axhline(.005,color="gray",ls="--",lw=.8)
        ax.set(ylabel="AUROC drop after additive surrogate",title="Diagnostic surrogate, not edge symbolification")
        _save(fig,directory,"symbolic_fidelity",d)


def pareto_table(out,results,runs):
    rows=[]
    for r in runs:
        d=results[(results.run_dir==r["run_dir"])&(results.dataset=="celebdfpp")&
                  (results.level=="video")&(results.method=="all")&(results.calibration=="raw")]
        clean=d[d.condition=="clean"]
        degraded=d[d.condition!="clean"]
        path=out/r["run_dir"]/"intervention.csv"
        gap=None
        if path.exists() and path.stat().st_size>2:
            iv=pd.read_csv(path)
            if len(iv):
                iv=iv[iv.dataset=="celebdfpp"]
                if len(iv) and iv.total_sse.sum()>1e-10:
                    by_video=iv.groupby("video_id")[["error_sse","total_sse","frame_count"]].sum()
                    numerator=(by_video.error_sse/by_video.frame_count).mean()
                    denominator=(by_video.total_sse/by_video.frame_count).mean()
                    gap=float(np.sqrt(numerator/denominator)) if denominator>1e-10 else None
        rows.append({k:r[k] for k in ["head","budget","k","covariates","held_out","seed"]}|
                    {"A":float(clean.auc.mean()),"P":r["params"],"C":r["head_latency_ms"],
                     "R":float(degraded.auc.mean()) if len(degraded) else None,"I_gap":gap,
                     "train_wall_s":r["train_wall_s"]})
    measured=pd.DataFrame(rows)
    keys=["head","budget","k","covariates","held_out"]
    objectives=["A","P","C","R","I_gap"]
    numeric=objectives+["train_wall_s"]
    measured[numeric]=measured[numeric].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    grouped=measured.groupby(keys,dropna=False)
    counts=grouped[objectives].count().add_suffix("_observed_runs")
    counts["run_count"]=grouped.size()
    means=grouped[numeric].mean().join(counts).reset_index()
    means["pareto_nondominated"]=pd.Series(pd.NA,index=means.index,dtype="boolean")
    means["pareto_missing_objectives"]=[
        json.dumps([name for name in objectives if row[name+"_observed_runs"]<row["run_count"]])
        for _,row in means.iterrows()]
    means["pareto_status"]="UNAVAILABLE"
    for _,cell in means.groupby(["budget","k","covariates","held_out"],dropna=False):
        valid=cell[cell.pareto_missing_objectives=="[]"]
        values=valid[objectives].to_numpy()*np.array([-1,1,1,-1,1])
        for i,index in enumerate(valid.index):
            dominates=(values<=values[i]).all(1)&(values<values[i]).any(1)
            means.loc[index,"pareto_nondominated"]=not bool(dominates.any())
            means.loc[index,"pareto_status"]="DESCRIPTIVE_COMPLETE_CASES"
    means["interpretation"]=("Descriptive point-estimate front among rows with all objectives observed for every run; "
                             "missing objectives are not ranked; no uncertainty-aware architecture recommendation")
    means.to_csv(out/"pareto.csv",index=False)
    return means


def build_report(out: Path, cfg):
    cfg, manifest = report_configuration(out, cfg)
    sources=sorted(p for p in out.rglob("*") if p.is_file() and p.parent.name.startswith("seed")
                   and p.suffix in {".json",".csv"})
    reportkey=fingerprint({"inputs":{str(p.relative_to(out)):digest_file(p) for p in sources},
                           "eval":OmegaConf.to_container(cfg.eval,resolve=True),
                           "synthetic":bool(cfg.experiment.synthetic),"source":source_fingerprint(),
                           "manifest": digest_file(out / "manifest.json")})
    reportdir=out/"report_stage"
    if not cfg.force and complete(reportdir,reportkey):
        record=json.loads((reportdir/"artifacts.json").read_text(encoding="utf-8"))
        if all((out/n).is_file() and digest_file(out/n)==sha for n,sha in record.items()):
            print("SKIP aggregate/figures/report")
            return
    results,summary,runs=collect(out, synthetic=manifest["synthetic"])
    response_audits = response_audit_table(out, runs)
    pareto_table(out,results,runs)
    figures(out,results,summary,runs,cfg)
    stats=hypothesis_tests(out,runs,cfg)
    primary=summary[(summary.dataset=="celebdfpp") & (summary.condition=="clean") &
            (summary.level=="video") & (summary.method=="all") & (summary.calibration=="raw")]
    lines=["# FRESCO empirical report", "", "SYNTHETIC INTEGRATION FIXTURE — NO SCIENTIFIC EVIDENCE." if cfg.experiment.synthetic
           else "Measured results from the supplied face videos. No target data were used for model selection.", "",
           f"Seeds = {len(set(r['seed'] for r in runs))} (paper §6 recommends ≥10). "
           f"Seeds: {sorted(set(r['seed'] for r in runs))}.", "",
           "Primary interpretation question: how much of the real paired image response is captured by an observed-quality partial response? "
           "A disagreement is an explanation-coverage gap; the algebraic decomposition does not establish causality.", "",
           "## Measurements", "", "| Head | Budget | k | Covariates | Held-out method | Video AUROC mean ± seed SD | Parameters | Head b256 ms |",
           "|---|---|---:|---|---|---:|---:|---:|"]
    for _,r in primary.iterrows():
        lines.append(f"|{r['head']}|{r.budget}|{r.k}|{r.covariates}|{r.held_out}|{r.auc_mean:.4f} ± {r.auc_std:.4f}|{r.params:.0f}|{r.head_latency_ms:.4f}|")
    if _analysis_role(cfg):
        lines += ["", "Analysis role: **EXPLORATORY FOLLOW-UP**. " + _analysis_role(cfg),
                  "The original main experiment and its negative result remain separate.", ""]
    lines += ["", "## Hypothesis assessment", "", "See [stats.json](stats.json) for paired conditional intervals and budget feasibility. "
              "TOST uses 90% intervals. Failure to demonstrate equivalence is inconclusive. Comparators are selected with FF++ validation only. "
              "Mean single-seed AUC contrasts are used; scores are not ensembled for inference.", "",
              "| Comparison | Budget | k | Covariates | Held-out method | Verdict | Reason |", "|---|---|---|---|---|---|---|"]
    for r in stats["H2a"]:
        lines.append(f"|{r['head']} vs {r['comparator']}|{r['budget']}|{r['k']}|{r['covariates']}|{r['held_out']}|{r['status']}|{r.get('reason','')}|")
    paired=stats["primary_paired_response"]
    lines += ["", "## Primary paired response agreement", "",
              f"Status: **{paired['status']}**. {paired.get('reason','')}", "",
              "Prespecified contrast: spline KAN minus GELU MLP, k16, observed quality covariates, parameter budget. "
              "Each video has equal weight; seed-specific agreement ratios are averaged.", ""]
    if "gap_kan" in paired:
        lines += [f"G(KAN) = {paired['gap_kan']:.4f}; G(MLP) = {paired['gap_mlp']:.4f}; "
                  f"difference = {paired['difference']:.4f}, 95% cluster CI = {paired['ci95']}.", "",
                  f"Median curve stability = {paired['median_curve_stability']}. "
                  f"AUROC noninferiority contrast: {paired['auc_noninferiority']}.", ""]
        coverage=paired["curve_stability_coverage"]
        lines += [f"Curve stability coverage complete: {coverage['complete']}; "
                  f"covariates {coverage['complete_covariates']}/{coverage['expected_covariates']}; "
                  f"defined seed pairs {coverage['defined_pairs']}/{coverage['expected_pairs']}. "
                  "Constant, unavailable or incomplete curves make the primary verdict INCONCLUSIVE. "
                  "See stats.json for per-feature reasons.", ""]
    lines += ["", "## Frozen curve prediction audit", "",
              "G_block (the original endpoint) evaluates the full head with hybrid covariates. "
              "G_curve evaluates an additive predictor built from frozen source-validation PD curves. "
              "The original 0.30 threshold is not a success rule for G_curve.", "",
              "[Per-run curve fidelity and coverage](response_audit_summary.csv). "
              "Out-of-grid predictions are unavailable; covered-only gaps must be read alongside coverage. "
              "Unchanged finite covariate contributions cancel exactly without interpolation. "
              "Covered-only G_curve and full-test G_block are not numerically comparable on different populations. "
              "Marginal 1%-99% box excursions and actual per-layer spline clipping are separate diagnostics.", ""]
    if response_audits.empty:
        lines += ["These saved runs predate the curve audit; report-only mode cannot manufacture new measurements. "
                  "Use the separate replay_audit command with the verified original cache and a new output directory.", ""]
    else:
        lines += ["Per-run artifacts: curve_predictor.csv, curve_intervention.csv, response_support.csv, spline_support.csv. "
                  "Actual spline-grid support uses deterministic capped samples; covariate-box diagnostics use all rows.", ""]
    lines += ["", "## Article criteria and limitations", "",
        "A: frame/video AUROC, AP, EER, Brier, NLL and equal-width ECE. "
        "EER is the FPR=FNR crossing of the linearly interpolated empirical ROC, retaining all thresholds. "
        "P: every trainable coefficient counted; "
        "fixed CLIP and PCA state are not trainable head parameters. C: separate parameter and train-wall-time budgets, "
        "head b1/b256 latency and training memory. Encoder/crop timing must be read from the feature/preprocessing manifests; "
        "head latency alone is not pipeline latency.", "",
        "R: paired degradations and cross-dataset transfer. I: observed-covariate response stability and intervention decomposition. "
        "Symbolic PD approximation is a diagnostic, not validated full-model symbolification. "
        "No universal monotonic direction of fake score versus JPEG quality is assumed. "
        "E: resolved configs, source/data hashes, initialization checkpoints, completed-stage checksums and failed-run logs.", "",
        "Intervals are conditional on the fixed seeds; training-seed dispersion must be reported separately. "
        "Identity clustering holds target identities together; "
        "cross-role identity/source dependencies can remain. DeLong video intervals are secondary descriptive intervals. "
        "Unknown-method detection and cross-dataset shift are distinct tasks; the latter confounds dataset and generator. "
        "The adapted first-layer plasticity score does not reproduce the complete Canevaro pipeline.", "",
        "Non-goals retained transparently: no LN tuning, no claim of exact upstream KAN implementations, "
        "no fabricated c40 condition when c40 videos are absent, no automatic Pareto recommendation from point estimates alone.", "",
        "## Artifacts", "", "[Results](results.csv), [seed summary](results_summary.csv), "
        "[descriptive Pareto table](pareto.csv), [novelty results](openset_results.csv), "
        "[surrogate fidelity](symbolic_fidelity.csv), [environment and data provenance](manifest.json).", ""]
    for path in sorted((out/"figures").glob("*.svg")):
        lines.append(f"- [{path.stem}](figures/{path.name}) · [source data](figures/{path.stem}.csv)")
    failures=list(out.rglob("FAILED.json"))
    lines += ["", f"Completed runs: {len(runs)}. Failure records: {len(failures)}. Negative outcomes are retained.", ""]
    (out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    names=["REPORT.md","results.csv","results_summary.csv","stats.json","stability.csv",
           "openset_results.csv","symbolic_fidelity.csv","pareto.csv","response_audit_summary.csv"]
    names += [str(p.relative_to(out)) for p in sorted((out/"figures").glob("*"))]
    atomic_json(reportdir/"artifacts.json",{n:digest_file(out/n) for n in names})
    mark_complete(reportdir,reportkey,["artifacts.json"])