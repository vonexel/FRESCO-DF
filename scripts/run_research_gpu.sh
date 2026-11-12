#!/usr/bin/env bash


# run from an unpacked source bundle
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"

STAGE="${1:-help}"
DEVICE="${FRESCO_DEVICE:-cuda}"
OUTPUT_ROOT="${FRESCO_OUTPUT_ROOT:-$PROJECT_DIR/outputs/gpu_extension}"
RESPONSE_CACHE="${FRESCO_RESPONSE_CACHE:-$PROJECT_DIR/data/cache/clip_vitl14_response}"
BUDGETS="${FRESCO_BUDGETS:-[params]}"
RUN_TAG="${FRESCO_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'FRESCO_RUN_TAG may contain only letters, digits, underscores and hyphens.' >&2
  exit 2
fi
for value in "$OUTPUT_ROOT" "$RESPONSE_CACHE" "${FRESCO_DATA_DIR:-}" "${FRESCO_CACHE:-}"; do
  if [[ "$value" == *"'"* || "$value" == *$'\n'* ]]; then
    echo 'Paths must not contain apostrophes or newlines (Hydra command-line quoting).' >&2
    exit 2
  fi
done

require_env() {
  if [[ -z "${!1:-}" ]]; then
    echo "Set $1 as described in docs/UBUNTU_GPU_RUN.md" >&2
    exit 2
  fi
}
preflight_at() {
  local destination="$1"
  shift
  uv run --frozen python scripts/server_preflight.py --device "$DEVICE" --output-root "$destination" "$@"
}
preflight() {
  preflight_at "$OUTPUT_ROOT" "$@"
}
run_preset() {
  local preset="$1" cache="$2"
  [[ -f "configs/experiment/$preset.yaml" ]] || { echo "Missing preset: $preset" >&2; exit 2; }
  uv run --frozen fresco "experiment=$preset" 'stages=[train,report]' \
    "paths.feature_cache='$cache'" "paths.output_dir='$OUTPUT_ROOT'" \
    "train.device=$DEVICE" "experiment.budgets=$BUDGETS"
}
latest_for() {
  uv run --frozen python -c \
    'import json, pathlib, sys; p=pathlib.Path(json.loads(pathlib.Path(sys.argv[1]).read_text())["path"]); assert p.is_dir(), p; print(p)' \
    "$OUTPUT_ROOT/$1/LATEST.json"
}

case "$STAGE" in
  verify)
    python3 scripts/build_transfer_bundle.py --verify-directory "$PROJECT_DIR"
    ;;
  preflight)
    preflight --mode environment
    ;;
  smoke)
    preflight --mode environment
    uv run --frozen --extra dev pytest -q
    uv run --frozen fresco experiment=smoke 'stages=[train,report]' \
      'experiment.seeds=[0,1]' 'experiment.heads=[linear,mlp_gelu,kan_spline]' \
      'experiment.budgets=[params]' "paths.data_dir='$OUTPUT_ROOT/smoke_data'" \
      "paths.output_dir='$OUTPUT_ROOT/smoke_$RUN_TAG'" eval.bootstrap=50
    ;;
  audit)
    require_env FRESCO_CACHE
    require_env FRESCO_MAIN_EXPERIMENT
    preflight --mode audit --cache "$FRESCO_CACHE" --experiment "$FRESCO_MAIN_EXPERIMENT"
    uv run --frozen python -m fresco.research_audit --experiment "$FRESCO_MAIN_EXPERIMENT" \
      --output "$OUTPUT_ROOT/main_uncertainty_$RUN_TAG" --resamples 10000
    uv run --frozen python -m fresco.explanation_audit --experiment "$FRESCO_MAIN_EXPERIMENT" \
      --cache "$FRESCO_CACHE" --output "$OUTPUT_ROOT/main_ladder_$RUN_TAG" --device "$DEVICE"
    ;;
  native-audit)
    require_env FRESCO_CACHE
    require_env FRESCO_FULL_EXPERIMENT
    preflight --mode audit --cache "$FRESCO_CACHE" --experiment "$FRESCO_FULL_EXPERIMENT"
    uv run --frozen python -m fresco.explanation_audit --experiment "$FRESCO_FULL_EXPERIMENT" \
      --cache "$FRESCO_CACHE" --output "$OUTPUT_ROOT/native_linear_$RUN_TAG" --device "$DEVICE" \
      --heads linear --k 1024 --without-covariates --allow-additional-conditions
    ;;
  features)
    require_env FRESCO_DATA_DIR
    require_env FRESCO_CACHE
    preflight --mode features --data-dir "$FRESCO_DATA_DIR" --cache "$RESPONSE_CACHE" \
      --original-cache "$FRESCO_CACHE"
    uv run --frozen fresco experiment=response_augmented 'stages=[features]' \
      "paths.data_dir='$FRESCO_DATA_DIR'" "paths.feature_cache='$RESPONSE_CACHE'" \
      "paths.output_dir='$OUTPUT_ROOT'" "features.device=$DEVICE"
    ;;
  train)
    preflight --mode train --cache "$RESPONSE_CACHE" --source-conditions clean jpeg70 blur1 resize0.5
    run_preset response_clean "$RESPONSE_CACHE"
    run_preset response_augmented "$RESPONSE_CACHE"
    ;;
  support)
    preflight --mode train --cache "$RESPONSE_CACHE" --source-conditions clean jpeg70 blur1 resize0.5
    for preset in ${FRESCO_SUPPORT_PRESETS:-support_quality3 support_range}; do
      run_preset "$preset" "$RESPONSE_CACHE"
    done
    ;;
  representations)
    require_env FRESCO_CACHE
    preflight --mode train --cache "$FRESCO_CACHE"
    for preset in ${FRESCO_REPRESENTATION_PRESETS:-representation_control basis_control}; do
      if [[ "$preset" == basis_control ]]; then
        preflight --mode train --cache "$RESPONSE_CACHE" --source-conditions clean jpeg70 blur1 resize0.5
        run_preset "$preset" "$RESPONSE_CACHE"
      else
        run_preset "$preset" "$FRESCO_CACHE"
      fi
    done
    ;;
  analyze)
    clean_experiment="$(latest_for response_clean)"
    augmented_experiment="$(latest_for response_augmented)"
    factorial_output="$OUTPUT_ROOT/factorial_$RUN_TAG"
    preflight_at "$factorial_output" --mode audit --cache "$RESPONSE_CACHE" --experiment "$clean_experiment" \
      --source-conditions clean jpeg70 blur1 resize0.5
    uv run --frozen python -m fresco.factorial_audit --clean "$clean_experiment" \
      --augmented "$augmented_experiment" --output "$factorial_output" --resamples 10000
    for preset in response_clean response_augmented; do
      experiment="$(latest_for "$preset")"
      uv run --frozen python -m fresco.explanation_audit --experiment "$experiment" \
        --cache "$RESPONSE_CACHE" --output "$OUTPUT_ROOT/${preset}_ladder_$RUN_TAG" --device "$DEVICE"
    done
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage: bash scripts/run_research_gpu.sh STAGE
  verify          Verify transferred source files.
  preflight       Check Python/locked CUDA runtime, finite head gradients and free disk.
  smoke           Preflight + tests + small CPU synthetic integration fixture.
  audit           Seed/cluster uncertainty and matched explanation ladder on saved main heads.
  native-audit    Evaluate saved native linear head on existing paired degradations.
  features        Build a NEW augmented FF++ cache using already prepared crops.
  train           Matched-step clean vs augmented training (params budget by default).
  support         Quality-only and source-range support controls.
  representations Linear dimension and spline-MLP basis controls.
  analyze         Factorial and matched explanation-ladder analysis after train.
Required paths vary by stage; see docs/UBUNTU_GPU_RUN.md.
HELP
    ;;
  *) echo "Unknown stage: $STAGE (use --help)" >&2; exit 2 ;;
esac