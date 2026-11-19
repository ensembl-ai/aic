# History of the build

Append new entries at the bottom: date, what changed, result, decision.

## Pipeline as of 2026-11-18

- **Scene estimator v5** (`runs/scene_estimator_v5`): DINOv2 ViT-S/14
  backbone, per-patch ray embeddings from camera intrinsics and TF,
  transformer with 13 learned queries (schema `aic_v2`: board, 10 SFP ports,
  SC ports by rail rank). It outputs presence, position, 6D rotation and
  confidence per target. 2.55 mm mean port error on validation. Trained on
  `aic_v3` (5000 boards x 6 views, 25% SC) with early stopping.
- **Close-range refiner v1** (`runs/refiner_v1`): plug-to-port offset from
  336 px wrist crops, 0.2 mm median within 2 mm. Trained on `aic_v3` plus
  `aic_v4_close` (3000 x 8 views, 3 to 40 mm, 30% on axis). Disabled in
  `infer.json`: the servo built on it oscillated, and one model alone already
  works.
- **Policy `PerceptionInsert`** (staged, the default): estimate the port at
  the start pose, move to a 5 cm hover, re-estimate, move to 1 cm, descend
  until contact, spiral search, press with soft impedance until seated, and
  retry snags. Moves settle with rest-then-correct.

## Problems found and fixed

- Data collection workers died on OOM restarts; OOM no longer counts as a
  failure and memory is capped.
- Hover oscillation was integrator windup in `move_plug` (correcting while
  the arm still moved). Fixed with rest-then-correct (correct only at rest,
  bounded to 8 mm, at most 4 times).
- SC trials failed because the old ghcr sim image had `sc_port_1` on the
  other rail. Built `aic_eval:phase1` from the repo; the collector now uses
  the image's own board and cable descriptions.
- A v4 checkpoint was overwritten by a later run; retrained as v5.
- The engine counts an insertion only after about 1 s of steady touch-sensor
  contact plus a lock delay. A policy that returns right after seating can
  score "partial insertion" even at full depth.

## Benchmarks (six trials: sample 1-3, eval 1-3; sample3 and eval3 are SC)

| Date | Variant | s1 | s2 | s3 | e1 | e2 | e3 | Total |
|---|---|---|---|---|---|---|---|---|
| 11-18 | Staged (default) | ok | ok | ok | ok | ok | ok | **556.7** |
| 11-18 | Tracking v1 | 70.3 | 96.3 | 95.5 | 96.6 | 96.6 | 1.0 | 456.3 |
| 11-18 | Tracking v2 (close-weighted, hold) | 92.2 | 94.3 | 94.4 | 94.4 | 95.8 | 1.0 | 472.0 |
| 11-18 | Tracking v1 + seat hold | 93.2 | 95.6 | 95.1 | 71.0 | 96.3 | 1.0 | 452.1 |
| 11-18 | + rotation averaging + yaw wiggle | 93.3 | 95.1 | 1.0 | 96.0 | 96.3 | 1.0 | 382.6 |

Tracking = continuous approach that follows a running average of port
estimates instead of stopping at the 5 cm hover.

## Findings

- **SC needs the plug within about 1 degree of the port.** It seated at 1.0
  degrees or less in all cases (6 of 6) and jammed on the internal ledges
  (about 10.5 mm deep) at 1.8 to 2.2 degrees (all cases).
- Tracking kept the start-pose rotation, about 2 degrees off. That jammed SC
  and caused a partial SFP insertion. Averaging the rotation fixed the SFP
  case (2.0 to 0.74 degrees) but not eval3 SC (1.8 degrees): the estimator's
  close-up yaw for that port is biased. Staged re-estimates at 5 cm and gets
  it right.
- The yaw wiggle on snags did nothing (same 8.3 mm stall each retry). It ran
  under the soft-rotation recovery impedance, so the yaw likely never reached
  the plug.
- Web research: the winning team (Flex) used a lateral swing for SC;
  several teams reported SC partial insertions; SC connectors snap-latch.

## Decisions

- 2026-11-18: staged approach is the default (commit `58fa07d`). The
  tracking experiments (hold, rotation averaging, wiggle) were not committed;
  the patch is kept outside the repo.
- Next: a larger randomized benchmark, so changes are judged on a real
  success rate instead of six trials.

## 2026-11-18: benchmark generator and presentation clips

- `scripts/make_benchmark.py` (config `benchmark` in `collect.json`) writes
  40 randomized trials (26 SFP, 14 SC, seed 2026) as four engine configs of
  10 into `runs/benchmark/batch_{0..3}.yaml`. Trials come from the
  collector's scene sampler, so they follow the training ranges but are new
  boards. With two SC ports present, the lower-translation port is written
  as `sc_port_1`, matching `sc_rank_for_slot` and the phase-1 eval trials;
  the policy cannot tell the two apart any other way (known risk).
  Estimated run time: about 7 minutes per batch of 10.
- Presentation clips of five SFP trials (staged policy, all full
  insertions, about 95 each) are in `runs/videos/sfp_successes/<trial>/`:
  close-up isometric on the target port, medium isometric, side, wide shot
  of the cell, and the robot's center wrist camera. Recording tooling is
  in `runs/videos/tooling/` (`run.sh`): static Gazebo cameras spawned at run
  time with the CameraVideoRecorder system; the close-up is moved onto each
  trial's port from ground-truth TF. Gazebo cannot move its temp video onto
  a bind mount, so clips are saved inside the container and copied out.
- Staged is not immune to the partial-insertion timing issue: in one
  recording take, sample1 and bench_16 scored about 70 ("partial
  insertion" at full depth); in the next take all five scored about 95.
  The seat hold should be added to staged and checked on the benchmark.
