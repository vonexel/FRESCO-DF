import hashlib
import xml.etree.ElementTree as ET
import pandas as pd
import pytest
from fresco.extensions.surrogate_run import _summary, _surrogate_video, _teacher_distribution
from fresco.extensions.surrogate_visuals import surrogate_figures


def test_surrogate_contrasts_use_same_seed_and_video():
    frame = pd.DataFrame([
        {"dataset": d, "condition": "jpeg70", "label": v % 2, "video_hash": f"v{v}",
         "group_hash": f"g{v}", "total": 2., "zero": 0., "ridge": 1., "gam": 1.5,
         "ebm": 1.75, "ebm_interactions": 2.}
        for d in ("ffpp", "celebdfpp") for v in range(6)])
    video = _surrogate_video(frame)
    tables = {(head, seed): video.copy() for head in ("kan_spline", "mlp_gelu") for seed in range(10)}
    report, fidelity, transfer = _summary(tables, resamples=20, seed=13)
    assert report["kan_spline/ffpp"]["gam_minus_ridge"]["difference"] == pytest.approx(-.25)
    assert report["kan_spline/ffpp"]["ebm_minus_ridge"]["difference"] == pytest.approx(-.375)
    assert fidelity.query("method == 'zero'").G.eq(1).all()
    assert transfer.difference.abs().max() == pytest.approx(0)


def test_teacher_score_distribution_keeps_classes_separate():
    frame = pd.DataFrame({"dataset": ["ffpp"]*3, "condition": ["jpeg70"]*3, "label": [0, 1, 1],
                          "total": [1., 2., 3.], "clean_teacher_logit": [0., 1., 2.],
                          "changed_teacher_logit": [1., 3., 5.]})
    table = _teacher_distribution(frame, "kan_spline", 0).set_index("label")
    assert table.loc[0, "pairs"] == 1 and table.loc[1, "pairs"] == 2
    assert table.loc[1, "changed_logit_median"] == 4.


def test_surrogate_svg_rebuilds_deterministically(tmp_path):
    methods = ("zero", "ridge", "gam", "ebm", "ebm_interactions")
    pd.DataFrame([{"head": h, "dataset": d, "method": m, "G": .9,
                   "ci_low": .8, "ci_high": 1., "videos": 6}
                  for h in ("kan_spline", "mlp_gelu") for d in ("ffpp", "celebdfpp") for m in methods]).to_csv(
                      tmp_path / "surrogate_fidelity.csv", index=False)
    pd.DataFrame([{"head": h, "dataset": d, "seed": 0, "observed_bin": 0., "n": 5, "observed": .1,
                   "ridge": .1, "gam": .2, "ebm": .3}
                  for h in ("kan_spline", "mlp_gelu") for d in ("ffpp", "celebdfpp")]).to_csv(
                      tmp_path / "surrogate_agreement.csv", index=False)
    pd.DataFrame([{"head": h, "method": m, "difference": .1, "ci_low": -.1, "ci_high": .2,
                   "source_videos": 10, "target_videos": 20}
                  for h in ("kan_spline", "mlp_gelu") for m in methods]).to_csv(
                      tmp_path / "surrogate_transfer.csv", index=False)
    pd.DataFrame([{"feature": f, "method": m, "grid": x, "response": x+s*.1, "source_bin_count": 5}
                  for f in range(8) for m in ("gam", "ebm") for x in (0., 1.) for s in range(2)]).to_csv(
                      tmp_path / "surrogate_functions.csv", index=False)
    surrogate_figures(tmp_path)
    files = sorted(tmp_path.glob("*.svg"))
    assert len(files) == 4
    for file in files:
        assert ET.parse(file).getroot().tag.endswith("svg")
    initial = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    surrogate_figures(tmp_path)
    assert initial == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]