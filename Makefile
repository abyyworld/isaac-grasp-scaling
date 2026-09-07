# isaac-grasp-scaling - common workflows.
#
# Two machines are involved and the split matters for cost:
#   * dataset generation with the Isaac backend needs an RTX GPU (rented)
#   * everything else, including the MuJoCo control arm, runs on CPU
# See docs/setup-cloud.md.

PYTHON  ?= python3
VENV    ?= .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
RUFF    := $(VENV)/bin/ruff

BACKEND  ?= mujoco
SCENES   ?= 6000
WORKERS  ?= 4
NUM_ENVS ?= 1024
ANGLES   ?= 3
DATA     ?= data/mj18k
OUT      ?= results/scaling/mujoco
SIZES    ?= 128 256 512 1024 2048
EPOCHS   ?= 10
INPUT    ?= 96

.PHONY: help install check collect scaling plot throughput test lint clean

help:
	@echo "make install     create $(VENV) and install the package"
	@echo "make check       verify the install by running it (add ISAAC=1 on a GPU box)"
	@echo "make collect     generate a dataset      (BACKEND=$(BACKEND) SCENES=$(SCENES))"
	@echo "make scaling     train the curve         (DATA=$(DATA) SIZES='$(SIZES)')"
	@echo "make plot        redraw the curve from a finished run"
	@echo "make throughput  compare samples/hour between backends"
	@echo "make test        run the test suite"
	@echo ""
	@echo "Override any of: BACKEND SCENES WORKERS NUM_ENVS ANGLES DATA OUT SIZES EPOCHS INPUT"

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -q --upgrade pip

install: $(VENV)
	$(PIP) install -e ".[dev]"

check:
	$(PY) scripts/check_setup.py $(if $(ISAAC),--isaac,)

collect:
	$(PY) scripts/collect.py --backend $(BACKEND) --scenes $(SCENES) \
		--angles-per-scene $(ANGLES) --split seen --out $(DATA) \
		$(if $(filter isaac,$(BACKEND)),--num-envs $(NUM_ENVS),--workers $(WORKERS))

scaling:
	$(PY) scripts/run_scaling.py --data $(DATA) --out $(OUT) --sizes $(SIZES) \
		--epochs $(EPOCHS) --input-size $(INPUT)

plot:
	$(PY) scripts/plot_scaling.py --result $(OUT)/scaling.json

throughput:
	$(PY) scripts/throughput.py --dataset $(DATA) --hardware "$(shell uname -m), CPU" \
		--workers $(WORKERS)

test:
	$(PY) -m pytest tests/ -q

lint:
	$(RUFF) check src scripts tests

clean:
	rm -rf $(DATA) $(OUT)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
