# SO-101 sim2real — common workflows.
# Activate the env first:  conda activate sim2real   (or override PY=...)
PY ?= python
CONFIG ?= configs/reach.yaml
RUN ?= outputs/latest

.PHONY: help setup fetch-assets smoke train train-pickplace train-sac eval eval-dr watch video export deploy-sim test tb clean

help:
	@echo "setup         create/refresh the conda env and install the package"
	@echo "fetch-assets  re-download the SO-101 MuJoCo model + meshes"
	@echo "smoke         tiny end-to-end training run (pipeline sanity)"
	@echo "train         full PPO training     (CONFIG=$(CONFIG))"
	@echo "train-pickplace  PPO pick-and-place training (configs/pickplace.yaml)"
	@echo "train-sac     full SAC training"
	@echo "eval          evaluate a run        (RUN=path)"
	@echo "eval-dr       evaluate under domain randomization"
	@echo "watch         open the interactive MuJoCo viewer for a run"
	@echo "video         render an mp4 of a run (headless, MUJOCO_GL=egl)"
	@echo "export        export a run's policy to ONNX"
	@echo "deploy-sim    run the deploy loop against the MuJoCo stand-in"
	@echo "test          run the test suite"
	@echo "tb            launch tensorboard on outputs/"

setup:
	bash scripts/setup.sh

fetch-assets:
	bash scripts/fetch_assets.sh

smoke:
	$(PY) -m sim2real.train --smoke

train:
	$(PY) -m sim2real.train --config $(CONFIG)

train-pickplace:
	$(PY) -m sim2real.train --config configs/pickplace.yaml

train-sac:
	$(PY) -m sim2real.train --config configs/sac_reach.yaml

eval:
	$(PY) -m sim2real.eval --run $(RUN) --episodes 50

eval-dr:
	$(PY) -m sim2real.eval --run $(RUN) --episodes 50 --randomize

watch:
	$(PY) -m sim2real.visualize --run $(RUN)$(if $(SEED), --seed $(SEED))$(if $(EP), --episode $(EP))

video:
	MUJOCO_GL=egl $(PY) -m sim2real.visualize --run $(RUN) --video $(RUN)/rollout.mp4

export:
	$(PY) -m sim2real.export_policy --run $(RUN)

deploy-sim:
	$(PY) -m sim2real.deploy.deploy_so101 --run $(RUN)

test:
	$(PY) -m pytest

tb:
	tensorboard --logdir outputs --bind_all

clean:
	rm -rf outputs/* **/__pycache__ .pytest_cache
