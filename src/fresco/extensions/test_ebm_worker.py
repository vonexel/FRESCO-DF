import json
import os
import subprocess
from pathlib import Path
import numpy as np
import pytest


def test_separate_ebm_worker_has_no_target_response_labels(tmp_path):
    default = Path.cwd() / "new_task_work" / "ebm-env" / "Scripts" / "python.exe"
    python = Path(os.environ.get("FRESCO_EBM_PYTHON", str(default)))
    if not python.is_file():
        pytest.skip("Separate InterpretML environment not installed")
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(180, 8)).astype(np.float32)
    y = 2*x[:, 0]-x[:, 1]
    np.savez(tmp_path / "shared.npz", train_x=x[:140], val_x=x[140:160], test_x=x[160:],
             train_video=np.array([f"t{i}" for i in range(140)]),
             train_group=np.array([f"t{i}" for i in range(140)]),
             val_video=np.array([f"v{i}" for i in range(20)]), val_group=np.array([f"v{i}" for i in range(20)]))
    np.savez(tmp_path / "targets.npz", train_y=y[:140], val_y=y[140:160])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    command = [str(python), "-m", "fresco.extensions.ebm_worker", "--shared", str(tmp_path / "shared.npz"),
               "--targets", str(tmp_path / "targets.npz"), "--destination", str(tmp_path), "--seed", "0"]
    # InterpretML can contend with concurrent CPU/GPU scientific runs
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False, timeout=600)
    assert result.returncode == 0, result.stderr
    metadata = json.loads((tmp_path / "worker.json").read_text(encoding="utf-8"))
    assert metadata["test_teacher_responses_supplied_to_worker"] is False
    assert metadata["primary_ebm_interactions"] == 0
    assert all(len(term) == 1 for term in metadata["selection"]["ebm"]["term_features"])
    assert np.load(tmp_path / "predictions.npy").shape == (5, 20)