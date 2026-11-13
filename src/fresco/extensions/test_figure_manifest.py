import json
import pytest
from fresco.extensions.figure_manifest import PLOTS, write_figure_manifest


def test_figure_manifest_requires_matching_source_snapshot(tmp_path, monkeypatch):
    monkeypatch.setitem(PLOTS, "fixture", ("effect_visuals.py", ("plot",)))
    (tmp_path/"plot.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    (tmp_path/"plot.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    write_figure_manifest(tmp_path, "fixture")
    data = json.loads((tmp_path/"figure_manifest.json").read_text())
    assert data["figures"][0]["source"] == "plot.csv"
    assert (tmp_path/data["script_snapshot"]).is_file()
    (tmp_path/data["script_snapshot"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="generator changed"):
        write_figure_manifest(tmp_path, "fixture")