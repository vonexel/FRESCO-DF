EXP ?= main
ARGS ?=
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
.PHONY: setup test smoke prepare-data extract-faces extract-features run-all functions report
setup:
	uv sync --extra dev --frozen
test:
	uv run pytest -q
smoke:
	uv run fresco experiment=smoke $(ARGS)
prepare-data:
	uv run fresco 'stages=[index]' $(ARGS)
extract-faces:
	uv run fresco 'stages=[faces]' $(ARGS)
extract-features:
	uv run fresco 'stages=[features]' $(ARGS)
run-all:
	$(MAKE) functions
	uv run fresco experiment=full_features $(ARGS)
	uv run fresco experiment=main $(ARGS)
	uv run fresco experiment=k_ablation $(ARGS)
	uv run fresco experiment=covariate_ablation $(ARGS)
	uv run fresco experiment=lomo $(ARGS)
functions:
	uv run fresco experiment=functions 'stages=[train,report]' train.epochs=100 train.patience=15 train.wall_seconds=2 $(ARGS)
report:
	uv run fresco experiment=$(EXP) 'stages=[report]' $(ARGS)