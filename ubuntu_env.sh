#!/usr/bin/env bash
FRESCO_BUNDLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FRESCO_BUNDLE_ROOT
export FRESCO_CACHE="$FRESCO_BUNDLE_ROOT/data/cache/clip_vitl14"
export FRESCO_MAIN_EXPERIMENT="$FRESCO_BUNDLE_ROOT/legacy_results/original_outputs/main/074d119aafb7b25e8e28"
export FRESCO_REPLAY_PROVENANCE="$FRESCO_BUNDLE_ROOT/legacy_results/gpu_extension_outputs/main_ladder_20261028T134949Z/provenance.json"
export FRESCO_CURVES_V2="$FRESCO_BUNDLE_ROOT/legacy_results/original_outputs/research_followup_20261028/curves_v2"
export FRESCO_SOURCE_CACHE="$FRESCO_BUNDLE_ROOT/outputs/extensions/representation/source_pairs_20261106/source_pairs"
export FRESCO_EBM_PYTHON="$FRESCO_BUNDLE_ROOT/.venv-ebm/bin/python"
export FRESCO_CLEAN_RUN="$FRESCO_BUNDLE_ROOT/legacy_results/gpu_extension_outputs/response_clean/badcfa2dfe5dac0e1d18"
export FRESCO_AUGMENTED_RUN="$FRESCO_BUNDLE_ROOT/legacy_results/gpu_extension_outputs/response_augmented/83e15134a4249c71f3f9"
export FRESCO_HISTORICAL_CODE="$FRESCO_BUNDLE_ROOT/research_gpu_extension_20261028/src/fresco"
export FRESCO_SYMBOLIC_ATTEMPT="$FRESCO_BUNDLE_ROOT/outputs/extensions/explanations/symbolic_provenance_20261107_source/symbolic"
export FRESCO_DEVICE=cuda