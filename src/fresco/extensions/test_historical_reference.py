import json
import pandas as pd
import pytest
from fresco.extensions.historical_reference import load_full_feature_reference
from fresco.extensions.inventory import sha256


def test_reference_requires_complete_checksum_verified_table(tmp_path):
    stage = tmp_path/"report_stage"
    stage.mkdir()
    (tmp_path/"manifest.json").write_text("{}", encoding="utf-8")
    frame = pd.DataFrame([{"head": head, "covariates": cov, "dataset": dataset, "budget": "params", "k": 1024,
                           "condition": "clean", "level": "video", "method": "all", "calibration": "raw",
                           "auc_mean": .6, "auc_std": .1, "seeds": 10, "params": 2000}
                          for head in ("linear", "mlp_gelu") for cov in (False, True)
                          for dataset in ("ffpp", "celebdfpp")])
    table = tmp_path/"results_summary.csv"
    frame.to_csv(table, index=False)
    artifacts = stage/"artifacts.json"
    artifacts.write_text(json.dumps({"results_summary.csv": sha256(table)}), encoding="utf-8")
    (stage/"DONE.json").write_text(json.dumps({"files": {"artifacts.json": sha256(artifacts)}}), encoding="utf-8")
    output, audit = load_full_feature_reference(tmp_path)
    assert len(output) == 8 and output.role.eq("historical_reference_only_not_factorial").all()
    assert audit["status"] == "VERIFIED_HISTORICAL_REFERENCE"
    table.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_full_feature_reference(tmp_path)