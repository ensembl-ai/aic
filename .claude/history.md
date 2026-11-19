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

## 2026-11-19: first 40-trial benchmark (staged policy, v5, refiner off)

Total 2125 of 40 trials (53 per trial). The machine restarted at 00:44 for a
Windows Update right after the run; it was not caused by the run (the
cleanup had finished at 00:42).

| Type | Trials | Full (>90) | Partial (61-70) | Failed (1) | Mean |
|---|---|---|---|---|---|
| SFP | 26 | 14 | 3 | 9 | 59.4 |
| SC | 14 | 4 | 3 | 7 | 41.6 |

Failure causes:
- **Target port never detected at the start pose: 9 SFP + 1 SC.** The
  model saw other parts of the board but reported the target card's ports
  absent (presence 0.00 for 30 s), often missing several cards at once. On
  validation home views the same model has 97% presence recall, so this is
  a gap between the collection scenes and the engine's trials, not a
  model weakness seen in training data. Not yet explained; next step is to
  capture the home images of a failing trial and compare them with a
  training image of the same board.
- **Partial insertion at full depth: 3 SFP + 3 SC** (61 to 70 points). The
  plug was fully seated (SFP -46.7 mm, SC -15.6 mm) but the engine did not
  count it: the seat hold would recover about 25 points each.
- **SC did not enter: 4 trials** stalled on the surface (-1.5 to -2.2 mm)
  with the plug 1.3 to 2.0 mm off axis and 0.1 to 1.3 degrees rotated; the
  spiral search did not find the opening.
- **SC drift or wrong depth: 2 trials** (plug drifted 15 mm sideways; plug
  went 35 mm deep, i.e. beside the port).

## 2026-11-19: search when the target is not seen at the start pose

Change: if the target port has no present estimate within 4 s at the start
pose, the plug moves 8 cm above the port's expected position (mean
board-frame position from the aic_v3 labels, placed with the board
estimate, which is always detected) and looks again; then ±1.5 cm along
the slot. SFP targets only (SC rank resolution happens before this).

- The 9 SFP trials that failed on presence: 9 -> 751 (6 full, 3 partial).
  The first search viewpoint found the port every time; its first estimate
  was 1.5 to 14.7 mm off, corrected by the 5 cm re-estimate.
- **40-trial benchmark: 2125 -> 2956** (74 per trial). Batches: 911, 607,
  695, 744.
  - SFP 26: none failed; 17 full, 9 below 90 (mostly partial insertions
    at 65 to 70, i.e. seated but not counted).
  - SC 14: 6 full, 2 partial, 5 failed (1 point), 1 wrong port
    (bench_09, -11).
- Committed. Next: the seat hold for the partial insertions (~13 trials of
  about 25 points each), then SC entry.

## 2026-11-19: presentation clips re-recorded (policy `5e79dc3`)

Old clips deleted and re-recorded. First take: three trials full (about 95),
but sample1 and eval1 scored 70 (the partial-insertion timing issue), and
bench_16's clips were lost: the copy ran 5 s after the last stop, before
Gazebo had finalized the files. The wait before the copy in
`tooling/run.sh` is now 20 s. Those three trials were retaken: all about 95.
All five trials now have full insertions in `runs/videos/sfp_successes/`,
plus a contact sheet. bench_16's side camera misses its board: the cameras
are fixed and that board sits elsewhere.
- Clips re-shot for the presentation on harder boards: bench_29, 38, 16, 18,
  31 (3 to 5 NIC cards, board yaws from -179 to +150 degrees), all full
  insertions (92 to 96; bench_16 and bench_18 retaken after partials). The
  wide camera moved right to (2.175, -0.379, 2.05) so the cell pillar no
  longer blocks the view. `sfp_successes/all_trials.mp4` combines them:
  wide view as the main panel, plus the close-up, the wrist camera and a
  text box.
- LinkedIn story cut `runs/videos/linkedin_story.mp4` (2:08; problem, why it is hard, solution captions): title card, three SFP
  clips (bench_38, 16, 18), two SC clips (bench_00 and bench_04, newly
  recorded at 94 and 95 and kept in `runs/videos/sc_successes/`), end card.
  Plain-English captions, plus a "Plugged in" badge on the last frame. Each
  clip ends before the engine deletes the board. Of the four SC trials
  recorded, bench_07 was partial (70) and bench_12 failed (1).

## 2026-11-19: seat hold on the 40 trials; smarter search

- Seat hold (keep pressing 2.5 s after seating): 40-trial total 2743, below
  2956. It fixed every partial (about +220), but six other trials failed
  before seating (search not finding the port, a bad viewpoint estimate, the
  plug knocked in the gripper, two off-limit contacts with a NIC card). The
  start-pose detection and search vary run to run by about ±200, so a single
  40-trial run cannot judge a change of this size. Reverted, not committed.
- Search rework (`find_port`): expected pose from the sibling port when
  seen, else the board; the plug takes the orientation of the nearest seen
  SFP port; each viewpoint is reached through a 15 cm transit height; four
  viewpoints (`search_viewpoints`); detections more than 30 mm from expected
  are rejected. On the five trials that failed in the hold run (bench_18,
  23, 24, 27, 39): -19 -> 384, all at full depth, no contacts; three
  partials (the timing issue). Not yet run on all 40.
- **40-trial benchmark with the new search: 3078** (new best, was 2956).
  Batches 850, 782, 659, 786. SFP 2349 (was 2258): 23 full (two at 89.5),
  3 partial at full depth (bench_16, 27, 32: the timing issue), 0 failed.
  The search ran 9 times; all found the port at the first viewpoint, the
  position always from the board estimate. SC 730 (search not used for SC;
  6 failures as before: bench_01, 02, 03, 06, 10, and 09 at 37).
  Committed. Next: seat hold (partials), then SC entry.
