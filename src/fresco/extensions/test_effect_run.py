import hashlib
import xml.etree.ElementTree as ET
import pandas as pd
import pytest
from fresco.extensions.effect_run import _effect_bootstrap, _effect_video
from fresco.extensions.effect_visuals import effect_figures


def test_paired_effect_bootstrap_uses_same_videos():
    frame = pd.DataFrame([
        {"dataset": dataset, "condition": "jpeg70", "label": v % 2, "video_hash": f"v{v}",
         "group_hash": f"g{v}", "total": 2., "pd": 1., "ale": 1.5}
        for dataset in ("ffpp", "celebdfpp") for v in range(6)])
    video = _effect_video(frame)
    tables = {(h, s): video.copy() for h in ("kan_spline", "mlp_gelu") for s in range(10)}
    result, figure = _effect_bootstrap(tables, 20, 44)
    assert result["kan_spline/ffpp"]["ale_minus_pd"]["difference"] == pytest.approx(-.25)
    assert sorted(figure.G.unique()) == pytest.approx([.25, .5])
    again, _ = _effect_bootstrap(tables, 20, 44)
    assert result == again


def test_effect_svg_source_tables_reproduce_output(tmp_path):
    pd.DataFrame([{"head": h, "dataset": d, "method": m, "G": value,
                   "ci_low": value-.1, "ci_high": value+.1, "videos": 10}
                  for h in ("kan_spline", "mlp_gelu") for d in ("ffpp", "celebdfpp")
                  for m, value in (("pd", .9), ("ale", .8))]).to_csv(
                      tmp_path / "pd_ale_fidelity.csv", index=False)
    pd.DataFrame([{"head": h, "feature": f, "method": m, "grid": x, "response": x+s*.1, "bin_count": 5}
                  for h in ("kan_spline", "mlp_gelu") for f in ("a", "b", "c", "d") for m in ("pd", "ale")
                  for x in (0., 1.) for s in (0, 1)]).to_csv(tmp_path / "pd_ale_curves.csv", index=False)
    pd.DataFrame([{"dataset": d, "condition": "jpeg70", "paired": 10, "pd_supported": 9,
                   "ale_supported": 8, "common_supported": 8} for d in ("ffpp", "celebdfpp")]).to_csv(
                       tmp_path / "pd_ale_coverage.csv", index=False)
    effect_figures(tmp_path)
    files = sorted(tmp_path.glob("*.svg"))
    assert len(files) == 3
    for file in files:
        assert ET.parse(file).getroot().tag.endswith("svg")
    original = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    effect_figures(tmp_path)
    assert original == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]