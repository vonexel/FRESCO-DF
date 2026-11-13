import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pytest
from fresco.extensions.response_run import _bootstrap, _strata, _video_statistics, common_coverage
from fresco.extensions.runtime import FrozenInputs
from fresco.extensions.visuals import response_figures


def test_common_support_requires_every_curve_and_preserves_unchanged_outside():
    c0 = np.array([[2., 0.], [0., 0.], [0., 0.]])
    ca = np.array([[2., 0.], [2., 0.], [.5, 0.]])
    curves = {("a", 0): ([np.array([-1., 1.])]*2, [np.array([0., 1.])]*2),
              ("b", 0): ([np.array([-.75, .75])]*2, [np.array([0., 1.])]*2)}
    assert common_coverage(c0, ca, curves).tolist() == [True, False, True]


def test_frame_pairing_fails_closed():
    inp = object.__new__(FrozenInputs)
    inp.meta = pd.DataFrame([
        {"row": 0, "dataset": "ffpp", "video_id": "v", "frame_idx": 0, "split": "test",
         "condition": "clean", "label": 1, "method": "fake", "scenario": "s", "group_id": "g"},
        {"row": 1, "dataset": "ffpp", "video_id": "v", "frame_idx": 0, "split": "test",
         "condition": "jpeg70", "label": 1, "method": "fake", "scenario": "s", "group_id": "g"}])
    assert inp.pairs().iloc[0].clean_row == 0
    inp.meta = inp.meta.iloc[1:]
    with pytest.raises(ValueError, match="anchor"):
        inp.pairs()


def test_cluster_bootstrap_is_paired_and_reproducible():
    data = []
    for dataset in ("ffpp", "celebdfpp"):
        for video in range(6):
            data.append({"dataset": dataset, "video_hash": f"v{video}", "group_hash": f"g{video}",
                         "label": video % 2, "condition": "jpeg70", "total": 2., "cov": 1.,
                         "embed": .5, "interaction": .5, "curve": 1.})
    frame = pd.DataFrame(data)
    video = _video_statistics(frame)
    tables = {(head, seed): video.copy() for head in ("kan_spline", "mlp_gelu") for seed in range(10)}
    first = _bootstrap(tables, resamples=20, seed=123)
    second = _bootstrap(tables, resamples=20, seed=123)
    assert first == second
    assert first["ffpp"]["head_difference_curve"]["difference"] == pytest.approx(0)
    assert first["ffpp"]["kan_spline_curve_minus_zero"]["difference"] == pytest.approx(-.5)
    rows = _strata(frame, "kan_spline", 0, 1e-8)
    assert rows.query("dataset == 'ffpp' and condition == 'all' and label == 'all' and predictor == 'curve'").G.iloc[0] == pytest.approx(.5)


def test_svg_figures_are_vector_xml_and_reproducible(tmp_path):
    pd.DataFrame([{"dataset": d, "condition": "jpeg70", "label": 0, "paired": 10, "supported": 5}
                  for d in ("ffpp", "celebdfpp")]).to_csv(tmp_path / "coverage.csv", index=False)
    pd.DataFrame([{"dataset": d, "head": h, "seed": s, "support_bin": 0, "G": .8, "pairs": 5}
                  for d in ("ffpp", "celebdfpp") for h in ("kan_spline", "mlp_gelu") for s in range(2)]).to_csv(
                      tmp_path / "fidelity_vs_support.csv", index=False)
    pd.DataFrame([{"dataset": d, "head": h, "seed": s, "observed_bin": 0, "n": 5,
                   "observed": .1, "predicted": .2}
                  for d in ("ffpp", "celebdfpp") for h in ("kan_spline", "mlp_gelu") for s in range(2)]).to_csv(
                      tmp_path / "response_agreement.csv", index=False)
    pd.DataFrame([{"dataset": d, "head": h, "seed": s, "condition": "jpeg70", "pairs": 10,
                   "cov_mean": .1, "cov_std": .05, "embed_mean": .2, "embed_std": .1,
                   "interaction_mean": .3, "interaction_std": .1}
                  for d in ("ffpp", "celebdfpp") for h in ("kan_spline", "mlp_gelu") for s in range(2)]).to_csv(
                       tmp_path / "response_components.csv", index=False)
    response_figures(tmp_path)
    files = sorted(tmp_path.glob("*.svg"))
    assert len(files) == 4
    for file in files:
        assert ET.parse(file).getroot().tag.endswith("svg")
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    response_figures(tmp_path)
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]