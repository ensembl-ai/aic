# AIC cable insertion: working notes for Claude

Team prototype for the AI for Industry Challenge (cable insertion in Gazebo).
Our code lives in `aic_utils/aic_perception` on `main` (the
`rmalhan/far_approach` work was fast-forwarded into `main` on 2026-11-20
and the branch deleted).
The running history of results and decisions is in [history.md](history.md);
read it at the start of a session and append to it after every benchmark,
training run, or design decision.

## How we work (user preferences)

- Questions get answers and proposals only. Edit code or configs only on an
  explicit instruction ("implement", "fix", "do it").
- Prototype lab: work is sequential. Retraining and recollecting are fine
  when they are the right fix.
- Bare-minimum MVP, config-driven scripts (no CLI args), single `pixi run`
  commands. No leading-underscore names. Helpers go in `utils`, and use
  standard libraries (e.g. transforms3d) instead of handwritten math.
- READMEs hold the big picture only.
- Competition honesty: the policy may use only what the competition
  provides. Ground truth (`/scoring/tf`) is for logging, never control.
- Never touch system or NVIDIA drivers. Everything runs in the existing pixi
  env and the `aic_eval:phase1` image; install nothing.
- Long jobs (collection, training): give terminal commands unless the user
  delegates the run.
- Sim trials: run them only when the user asks. Leave no stale containers or
  processes. Kill by PID found with a bracketed grep
  (`ps -eo pid,cmd | grep -E "lib/aic_mode[l]|run_infer[e]nce"`); never
  `pkill -f` with a pattern that appears in the same command.
- Git: only `origin` (ensembl-ai/aic) and only the current working branch
  (now `main`). Never push, PR, or merge against `parent` (intrinsic-dev/aic);
  touch other branches only when the user asks.
  Commit a policy change only when it beats the current best benchmark.
- Before every commit run the system `black --check .` from the repo root
  (CI's Python style check uses the Ubuntu `black` package, default line
  length 88) and the tests (`pixi run --locked python -m pytest -q` in
  `aic_utils/aic_perception`).
- Keep the dev environment clean: nothing outside the repo (no saved
  patches, backups or scratch runners). Run logs and tooling go in the
  gitignored `aic_utils/aic_perception/runs/`; `runs/benchmark/run_trials.sh
  <trials.yaml> <log_dir>` runs one engine config and cleans up.
- Keep answers short; give the immediate next step, not everything at once.

## Running the six-trial benchmark

From the repo root, with `S` a log directory:

```bash
docker run -d --rm --name aic_eval_run --network host \
  -v $PWD/aic_utils/aic_perception/configs/engine_trials.yaml:/engine_trials.yaml:ro \
  --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -e GALLIUM_DRIVER=d3d12 -e MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  aic_eval:phase1 ground_truth:=true start_aic_engine:=true gazebo_gui:=false launch_rviz:=false \
  aic_engine_config_file:=/engine_trials.yaml model_discovery_timeout_seconds:=300
pixi run --locked python aic_utils/aic_perception/scripts/run_inference.py > $S/inference.log 2>&1 &
PYTHONPATH=$PWD/aic_utils/aic_perception pixi run --locked ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=aic_perception.policies.PerceptionInsert > $S/policy.log 2>&1 &
```

Scores: `docker logs aic_eval_run | grep -E "Trial '|Total Score"`. The policy
log has `[ground truth]` lines (port error, plug offset and rotation at each
phase) and per-trial JSONL records.

Build the sim image from this repo (phase-1 board, `sc_port_1` on rail 0):
see the README. The old ghcr image has a different SC layout and must not be
used.
