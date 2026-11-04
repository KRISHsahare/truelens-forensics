# One-word entry points for the whole repository. Every target is a line you could type yourself,
# and each one says which directory it runs in, because `uv` and `npm` resolve their environment
# from the current directory and a Makefile that hides that is a Makefile that breaks on a
# different machine.
#
# GNU make with a POSIX shell. On Windows that means WSL or Git Bash with `make` installed
# (choco install make); this box has neither, so the README lists the same commands unwrapped.
# `SHELL := /usr/bin/env bash` makes the multi-line recipes behave the same on every platform.

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help
BACKEND := backend
FRONTEND := frontend
# The API port this repository uses locally. 8000 is free on a clean machine; on this one another
# project already holds it, hence 8010 and `frontend/.env.local`.
API_PORT ?= 8010
WEB_PORT ?= 3000
# The interpreter the ML scripts run under: the backend's venv, because that is where torch lives.
# Windows: `make ml-smoke PY=backend/.venv/Scripts/python.exe`, or export PY once in your shell.
PY ?= $(BACKEND)/.venv/bin/python

.PHONY: help install dev api web test test-pool lint smoke ml-smoke artifacts build clean docker-up docker-down

help: ## Show this list.
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create both environments and install dependencies.
	cd $(BACKEND) && uv sync --extra dev
	cd $(FRONTEND) && npm ci

api: ## Run the API with reload on http://127.0.0.1:$(API_PORT).
	cd $(BACKEND) && uv run uvicorn app.main:app --reload --port $(API_PORT)

web: ## Run the frontend dev server on http://localhost:$(WEB_PORT).
	cd $(FRONTEND) && npm run dev

dev: ## Both servers, API first; Ctrl-C stops them together.
	@cd $(BACKEND) && uv run uvicorn app.main:app --port $(API_PORT) & \
	 API=$$!; trap 'kill 0' INT TERM EXIT; \
	 sleep 3; cd $(FRONTEND) && npm run dev; \
	 kill $$API 2>/dev/null

test: ## Backend suite: analysers, fusion, verdict, API, ML contracts.
	cd $(BACKEND) && uv run pytest -q

test-pool: ## The spawn-process-pool tests, which catch pickling regressions the in-process run cannot.
	cd $(BACKEND) && uv run pytest -q -m pool

lint: ## Ruff over the backend and ml, ESLint over the frontend.
	cd $(BACKEND) && uv run ruff check app scripts ../ml
	cd $(FRONTEND) && npm run lint

smoke: ## End-to-end checks that need no server: analysers, then the fusion layer, then the API app.
	cd $(BACKEND) && uv run python scripts/smoke_techniques.py
	cd $(BACKEND) && uv run python scripts/smoke_fusion.py
	cd $(BACKEND) && uv run python scripts/smoke_api.py

ml-smoke: ## The whole ML pipeline on generated data. Writes ml/.smoke, needs no downloads.
	$(PY) ml/scripts/make_corpus.py --out ml/.smoke/corpus --generated --images 48 --per-group 2
	$(PY) ml/scripts/train_cnn.py --corpus ml/.smoke/corpus --out ml/.smoke/model.pt --epochs 3 --batch 8 --no-pretrained
	$(PY) ml/scripts/evaluate_cnn.py --corpus ml/.smoke/corpus --weights ml/.smoke/model.pt --out-dir ml/.smoke --generated --limit 48
	$(PY) ml/scripts/calibrate_fusion.py --generated --corpus ml/.smoke/fusion_corpus --out ml/.smoke/fusion.joblib --report ml/.smoke/fusion_report.json --weights ml/.smoke/model.pt
	$(PY) ml/scripts/train_clip_probe.py --self-test --out ml/.smoke/clip_probe.joblib

artifacts: ## Check each produced artifact against what the running service expects.
	$(PY) ml/scripts/check_artifacts.py --weights artifacts/model.pt --metrics artifacts/metrics.json \
	  --fusion artifacts/fusion.joblib --clip-probe artifacts/clip_probe.joblib

build: ## Production build of the frontend (type-checks and prerenders every route).
	cd $(FRONTEND) && npm run build

docker-up: ## Whole stack in containers. Needs OVERLAY_SIGNING_KEY in the environment.
	docker compose up --build -d
	@echo "web http://localhost:$(WEB_PORT)  api http://127.0.0.1:$(API_PORT)/api/health"

docker-down: ## Stop the stack and remove the containers, not the images.
	docker compose down

clean: ## Remove build output and scratch. Never touches artifacts/ or a corpus.
	rm -rf $(FRONTEND)/.next $(FRONTEND)/tsconfig.tsbuildinfo
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache $(BACKEND)/app/**/__pycache__
	rm -rf ml/.smoke ml/.probe
