import hashlib
import json
import xml.etree.ElementTree as ET
import pandas as pd
import pytest
from fresco.extensions.inventory import sha256
from fresco.extensions.representation_run import (_label_factorial_report, _measurement,
                                                  _saved_seed_valid, validate_reuse_candidates,)
from fresco.extensions.representation_visuals import representation_figures, representation_tables


def test_validation_threshold_is_not_reselected_on_target():
    value = _measurement([0, 0, 1, 1], [.1, .7, .5, .9], threshold=.6)
    assert value["balanced_accuracy"] == pytest.approx(.5)
    assert value["source_validation_threshold"] == .6
    assert value["tpr_at_fpr_1pct"] is None
    assert value["tpr_at_fpr_1pct_status"] == "unstable_insufficient_negatives"
    with pytest.raises(FileNotFoundError, match="will not be retrained"):
        validate_reuse_candidates(None, None, None, None)


def test_extension_keeps_its_prespecified_primary_diagnostic():
    old = {"primary_diagnostic": "augmentation change in architecture x covariates interaction"}
    target = _label_factorial_report(old, 16, "celebdfpp")
    assert target["primary_diagnostic"] == "clean: architecture x covariates interaction"
    assert target["historical_engine_primary_diagnostic"] == old["primary_diagnostic"]
    assert _label_factorial_report(old, 128, "celebdfpp")["primary_diagnostic"] is None
    assert _label_factorial_report(old, 16, "ffpp")["primary_diagnostic"] is None


def test_completed_seed_requires_real_artifact_hashes(tmp_path):
    names = ("checkpoint.pt", "metrics.json", "scores.csv", "temperature.json")
    for name in names:
        (tmp_path/name).write_text(name, encoding="utf-8")
    (tmp_path/"DONE.json").write_text(json.dumps({"files": {n: sha256(tmp_path/n) for n in names}}))
    assert _saved_seed_valid(tmp_path)
    (tmp_path/"scores.csv").write_text("changed", encoding="utf-8")
    assert not _saved_seed_valid(tmp_path)


def test_factorial_figures_are_reproducible(tmp_path):
    rows = []
    for k in (16, 128):
        for recipe in ("clean", "augmented"):
            for head in ("kan_spline", "mlp_gelu"):
                for cov in (False, True):
                    for seed in range(2):
                        for dataset in ("ffpp", "celebdfpp"):
                            for condition in ("clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30",
                                              "blur1", "blur2", "blur3", "resize0.5", "resize0.25"):
                                rows.append({"k": k, "recipe": recipe, "head": head, "covariates": cov,
                                             "seed": seed, "dataset": dataset, "condition": condition,
                                             "auc": .6 + .01*cov + .001*seed})
    contrasts = [{"contrast": f"{recipe}: covariate effect, {head}", "difference": .01,
                  "crossed_seeds_clusters": {"ci95": [-.01, .03]}}
                 for recipe in ("clean", "augmented") for head in ("kan_spline", "mlp_gelu")]
    reports = {f"k{k}/celebdfpp": {"contrasts": contrasts, "videos": 100, "clusters": 20}
               for k in (16, 128)}
    representation_tables(pd.DataFrame(rows), reports, tmp_path)
    representation_figures(tmp_path)
    files = sorted(tmp_path.glob("*.svg"))
    assert len(files) == 4
    for file in files:
        assert ET.parse(file).getroot().tag.endswith("svg")
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    representation_figures(tmp_path)
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]