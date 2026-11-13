"""Exercise the actual Bash analyze entry point without a cache or GPU job."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _bash():
    found = shutil.which("bash")
    candidates = ([Path(found)] if found else []) + [Path("C:/Program Files/Git/bin/bash.exe")]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    pytest.skip("Bash is required to exercise the deployment runner")


def _preflight_module():
    spec = importlib.util.spec_from_file_location("runner_preflight", ROOT / "scripts/server_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(tmp_path):
    project = tmp_path / "project with spaces"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shell = scripts / "run_research_gpu.sh"
    shutil.copyfile(ROOT / "scripts/run_research_gpu.sh", shell)
    output = project / "outputs" / "gpu_extension"
    clean = output / "response_clean" / "completed_clean"
    augmented = output / "response_augmented" / "completed_augmented"
    for experiment in (clean, augmented):
        experiment.mkdir(parents=True)
        (experiment.parent / "LATEST.json").write_text(json.dumps({"path": str(experiment)}), encoding="utf-8")
    cache = project / "data/cache/response"
    cache.mkdir(parents=True)
    bin_dir = project / "bin"
    bin_dir.mkdir()
    log = project / "calls.txt"
    shim = project / "check_destination.py"
    shim.write_text('''import argparse, importlib.util, os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--output-root", required=True)
p.add_argument("--experiment", required=True)
p.add_argument("--cache", required=True)
a,_=p.parse_known_args()
spec=importlib.util.spec_from_file_location("real_preflight", os.environ["FRESCO_TEST_PREFLIGHT"])
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.reject_output_overlap(Path(a.output_root), [Path(a.experiment), Path(a.cache)])
''', encoding="utf-8")
    # This intercepts uv only. The actual production shell and actual overlap
    # guard run, while expensive runtime/data checks and research jobs do not.
    uv = bin_dir / "uv"
    uv.write_text('''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' CALL "$@" END >> "$FRESCO_TEST_LOG"
[[ "$1" == run ]] || exit 81
shift
[[ "$1" == --frozen ]] && shift
[[ "$1" == python ]] || exit 82
shift
case "$1" in
  -c)
    case "${@: -1}" in
      */response_clean/LATEST.json) printf '%s\\n' "$FRESCO_TEST_CLEAN" ;;
      */response_augmented/LATEST.json) printf '%s\\n' "$FRESCO_TEST_AUGMENTED" ;;
      *) exit 83 ;;
    esac ;;
  scripts/server_preflight.py)
    shift
    "$FRESCO_TEST_PYTHON" "$FRESCO_TEST_SHIM" "$@" ;;
  -m)
    case "$2" in
      fresco.factorial_audit|fresco.explanation_audit) exit 0 ;;
      *) exit 84 ;;
    esac ;;
  *) exit 85 ;;
esac
''', encoding="utf-8", newline="\n")
    uv.chmod(0o755)
    env = dict(os.environ)
    env.update({
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
        "FRESCO_OUTPUT_ROOT": output.as_posix(),
        "FRESCO_RESPONSE_CACHE": cache.as_posix(),
        "FRESCO_DEVICE": "cpu", "FRESCO_RUN_TAG": "regression",
        "FRESCO_TEST_LOG": log.as_posix(), "FRESCO_TEST_SHIM": shim.as_posix(),
        "FRESCO_TEST_PYTHON": Path(sys.executable).as_posix(),
        "FRESCO_TEST_CLEAN": clean.as_posix(), "FRESCO_TEST_AUGMENTED": augmented.as_posix(),
        "FRESCO_TEST_PREFLIGHT": (ROOT / "scripts/server_preflight.py").as_posix(),
        "MSYS2_ARG_CONV_EXCL": "*",
    })
    return project, shell, output, clean, augmented, env, log


def _run(runner):
    project, _, _, _, _, env, _ = runner
    return subprocess.run([_bash(), "scripts/run_research_gpu.sh", "analyze"], cwd=project,
                          env=env, text=True, capture_output=True, timeout=30)


def _calls(log):
    calls, current = [], None
    for line in log.read_text(encoding="utf-8").splitlines():
        if line == "CALL":
            current = []
        elif line == "END":
            calls.append(current)
            current = None
        else:
            current.append(line)
    return calls


def test_analyze_checks_exact_new_destination_and_never_retrains(runner):
    _, _, output, clean, augmented, _, log = runner
    result = _run(runner)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = _calls(log)
    preflight = next(call for call in calls if "scripts/server_preflight.py" in call)
    assert preflight.count("--output-root") == 1
    destination = Path(preflight[preflight.index("--output-root") + 1])
    assert destination == output / "factorial_regression"
    assert destination != output
    _preflight_module().reject_output_overlap(destination, [clean, augmented])
    factorial = next(call for call in calls if "fresco.factorial_audit" in call)
    assert factorial[factorial.index("--output") + 1] == destination.as_posix()
    assert sum("fresco.explanation_audit" in call for call in calls) == 2
    assert all("fresco" not in call and "stages=[features]" not in call
               and "stages=[train,report]" not in call for call in calls)


def test_old_container_check_reproduces_failure_before_analysis(runner):
    _, shell, _, _, _, _, log = runner
    source = shell.read_text(encoding="utf-8")
    assert 'preflight_at "$factorial_output"' in source
    shell.write_text(source.replace('preflight_at "$factorial_output"', 'preflight_at "$OUTPUT_ROOT"'),
                     encoding="utf-8", newline="\n")
    result = _run(runner)
    assert result.returncode != 0
    assert "Output overlaps protected" in result.stderr
    assert not any("fresco.factorial_audit" in call for call in _calls(log))


def test_new_destination_inside_saved_experiment_still_rejected(runner):
    _, _, _, clean, _, env, log = runner
    env["FRESCO_OUTPUT_ROOT"] = clean.as_posix()
    result = _run(runner)
    assert result.returncode != 0
    assert "Output overlaps protected" in result.stderr
    assert not any("fresco.factorial_audit" in call for call in _calls(log))


def test_overlap_guard_keeps_rejecting_ancestor_and_direct_destination(runner):
    _, _, output, clean, _, _, _ = runner
    reject = _preflight_module().reject_output_overlap
    for forbidden in (output, clean, clean / "analysis"):
        with pytest.raises(ValueError, match="overlaps"):
            reject(forbidden, [clean])
    reject(output / "factorial_new", [clean])
