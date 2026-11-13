from hydra import compose, initialize_config_dir
from pathlib import Path
import json
from fresco.run import execute


def test_primary_configuration_end_to_end_and_resume(tmp_path):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1]/"configs")):
        cfg=compose(config_name="config",overrides=["experiment=smoke","experiment.ks=[16]",
                    "experiment.heads=[mlp_gelu,kan_spline]","experiment.budgets=[params]",
                    "experiment.seeds=[0,1]"])
    cfg.paths.data_dir=str(tmp_path/"data")
    cfg.paths.output_dir=str(tmp_path/"outputs")
    out=execute(cfg)
    stats=json.loads((out/"stats.json").read_text())
    assert stats["primary_paired_response"]["status"]=="INCONCLUSIVE"
    assert "gap_kan" in stats["primary_paired_response"]
    assert all(row["status"]=="INCONCLUSIVE" for row in stats["H2a"])
    checkpoints=list(out.rglob("checkpoint.pt"))
    assert len(checkpoints)==4
    old={str(p):p.stat().st_mtime_ns for p in checkpoints+[out/"REPORT.md"]}
    assert len(list((out/"figures").glob("*.svg")))>=8
    again=execute(cfg)
    assert again==out
    assert old=={str(p):p.stat().st_mtime_ns for p in checkpoints+[out/"REPORT.md"]}

