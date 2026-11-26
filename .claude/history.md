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

## 2026-11-20: presentation content campaign

- Rich recorder (wrist images, camera poses, estimates, wrench, ground truth
  for logging) and renderers for annotated overlay and 3D videos, in
  `runs/videos/tooling/`; content in `runs/videos/content/` (README there).
- Failure modes re-created with their code: tracking jams SC at 2.0° (yaw
  locked from far away while the close-range estimate was 0.2° off); before
  the search the target card's presence stays 0 and the robot never moves;
  continuous correction while moving makes the plug swing 50-110 mm over
  the port. The old search's plug knock did not reproduce (4 of 4 inserted).
- Current policy, 15 showcase trials: 996; SC failures seen: spiral misses
  at 1.8 and 3.1 mm off axis (sample_trial_3, bench_06), arm stops 46 mm
  short (bench_02), drift after entry (bench_09).
- Decks (exec, technical, model deep dive) and a long-form article published
  as private artifacts; charts in `runs/figures/` (training curve, accuracy
  by distance, confidence calibration). Inference latency: 44 ms median.

## 2026-11-21: seat hold on the 20-trial targeted set (branch rmalhan/seat_hold_v2, uncommitted)

Report: `runs/benchmark/seat_hold_v2/report.md` (10 runs vs the 20-run baseline).
- Partials on the set 30% -> 8%; seated plugs scored partial 30% -> 9%; full
  insertions 67% -> 84%. Mean score flat (82.3 -> 82.0) because failures rose
  3% -> 8%, all before seating (the hold never ran in them).
- Failures cluster in runs 5 and 10 (run 10: 7 failures, 1561 s instead of
  ~850 s; a "483 mm" drift in run 5), the same stale-state symptoms seen under
  parallel load. Likely machine load during the day; the baseline ran overnight.
- Lesson: compare A/B interleaved on the same set, not against a baseline
  recorded at another time.

## 2026-11-23 to 25: subsystem diagnostics and SC fixes (rmalhan/seat_hold_v2, uncommitted)

Diagnostics report (5 pages): `runs/analysis/insertion_diagnostics.pdf`
(`report_pdf.py`; metrics from `subsystems.py` over the 800-trial study).
Experiments in `runs/experiments/` (interleaved A/B via `ab.sh`, report via
`ab_report.py`), all on the 7 SC trials bench_00/01/02/03/04/09/13.
- E1 seat hold soft rotation vs stiff: not conclusive. E2 refiner probe live
  (clean trials): SC 0.19 mm / 0.13 deg median, SFP 0.27 mm / 0.25 deg. The
  probe harness broke after trial 5 (trials start disturbed); rerun pending.
- E3 SC viewpoint + high transit (code): resolve SC rank from `sc_view_point`
  above the SC rail (board frame, 50 mm above card tops), vote over the last
  `sc_vote_window_s`; SC approach crosses at `approach_transit_height_m` and
  descends vertically. bench_01 now picks the right port; bench_02/13 reach
  the hover. bench_04 lands ~1 mm worse at pre-contact (open-loop step).
- E4 refiner servo on vs off (3 runs each): full 19% -> 67%, mean score per
  trial 20 -> 60. Servo converges in 2 steps to 0.1-0.4 mm and 0.1-0.2 deg,
  no oscillation. bench_01/03/13/04 now insert.
- Remaining SC mechanisms found:
  1. Lip jam at exactly 10.5 mm depth (12 of 13 entered-not-seated study
     trials, and E4): the SC port model has a 0.25 mm lip
     (`cube_collider_box.001`) over the middle 10.8 mm of the channel; plug
     housing 7.3 mm in an 8.1 mm channel (7.85 under the lip), so it passes
     only within about -0.4..+0.15 mm of centre, lip side = -y of the entrance
     frame. After 3 stalls the press retracts out of the correct hole.
  2. Lateral drift during the 12 mm descent after the servo (bench_09): plug
     aligned to 0.1-0.2 mm lands 1.3-2.3 mm off. Lateral impedance is 300 N/m.
     E5 tests 800 N/m lateral for servo and descent.
  3. bench_02: descent stops by force 1-7 mm above the entrance with the plug
     aligned to 0.1-0.4 mm: something other than the plug tip touches first
     (SC port at rail end -55 mm, next to NIC card 4). Not yet diagnosed.
- Policy logs now include the true plug offset in the port frame (x, y).
- Decks updated: system chain + root causes (exec), diagnostics section of
  five slides (technical), live refiner accuracy (model).
- E5 lateral stiffness 800 vs 300 N/m for servo and descent (servo on in
  both): full 62% -> 76%, mean 58.6 -> 71.5. bench_09 descent drift 2.6 ->
  1.1-1.4 mm (halved, not gone). Adopted in infer.json; refiner enabled.
- E6 lip handling on lip-prone boards (06/10/11/13 + 09/01/00/04): lip stalls
  6 -> 0, bench_10/11/13 full 4/9 -> 9/9; mean 66.7 -> 72.8, full flat at 71%
  because of two new losses, neither caused by the change:
  * bench_06 scores partial (65) exactly when the seated plug rests at port x
    = -0.35 mm, i.e. against the channel side wall (9.0 mm housing in a
    9.7 mm channel); full at x = +0.2 / -0.17. Seated-against-the-wall is not
    counted by the touch sensor. E7 holds the seated plug on the servo's port
    axis (`seat_hold_on_axis`).
  * Stale trial start (B run 2): bench_01 began with the plug already slipped
    in the gripper (66 mm off axis at the first reading, 21-23 deg rotated,
    kinematic plug estimate 18 mm off); the refiner was out of range and
    "aligned" it, the plug went in beside the port, and the next two trials
    (bench_00/04, normally 100%) saw no SC port. Needs a grasp sanity check
    (refiner plug vs kinematics) and safe abort.
- bench_02 (SC port at the -55 mm rail limit): the plug tip touches something
  ~5 mm above the port face and is pushed toward the slot centre; hypothesis:
  duplex plug body vs the slot end. Open.
- Refiner servo now runs for SFP too (refiner.enabled is global): SFP must be
  validated before any of this goes to main.
- E7 seat hold on the port axis: refuted (plug centred to 0.1 mm, bench_06
  still partial 2/3; A runs at x = -0.34 counted). Removed. The wall-contact
  explanation from E6 is wrong; cause of bench_06 never-firing touch unknown.
- E8 full benchmark x4 with the stack (SC viewpoint + transit, refiner servo
  at 0.2 mm, 800 N/m lateral servo/descent, lip bias + no retract inside,
  2.5 s seat hold), `runs/experiments/e8_full/`:
  * Healthy runs: 3502 and 3430 vs main 2876 +- 141 (study). Per trial over
    all 150 scored trials: SFP 87% full (main 71%), SC 81% (main 47%).
  * run 3 batch 1: simulator setup failure (TareFt timeout, board not
    deleted): infrastructure, excluded.
  * Collapses in 2 of 15 batches (run 3 batch 2 from bench_22, run 4 batch 3
    from bench_15): from one trial on, every trial fails; at the start the
    TF arm pose reads TCP z 0.159 (home ~0.33), plug estimate 14 mm off,
    board estimate 144 mm off. The engine logged a clean reset (cable
    deleted, robot homed). Main never collapsed (0 of 80 batches); the
    seat_hold_v2 runs showed the same signature in 2 of 10 runs, so the
    seat hold (commands sent while the CablePlugin has locked the end
    effector static) is the prime suspect. Not resolved: not for main.
- E9 full benchmark, stack with vs without the 2.5 s seat hold (2 runs each,
  interleaved; first attempt void: seat_hold_s 0.0 fails validation, used
  0.05): no hold 3374 / 3350, hold 3248 / 3434. Hold turns partials into
  fulls (18% -> 4% partial, 79% -> 89% full) but failures rise 4% -> 8% (as
  in seat_hold_v2): score-neutral (84.1 vs 83.5 per trial). No collapse in
  these 4 runs nor in the 2 earlier hold runs, so the hold is not shown to
  cause the E8 collapses; cause still open.
- Stack totals over 10 full runs: 3502, 3430, 3445, 3189, 3374, 3248, 3434,
  3350 clean; 2235 and 2703 with a collapsed batch (2235 also lost a batch to
  a simulator setup failure). Main: 2876 +- 141 over 20 runs.

## 2026-11-25: batch collapses explained (simulator reset fault), return-home fix, seat hold dropped

- Collapse cause: a simulator fault on the post-trial reset. After a
  successful insertion the CablePlugin holds the cable and end effector with
  static joints; the engine deletes the board, then the cable. Occasionally
  physics keeps referencing the deleted board's entities ("Physics.cc:3188
  physics entity ptr ... does not exist", millions of lines), the
  ResetJointsPlugin homing fails, and every later trial starts where the
  previous one ended (bench_30 started at bench_26's final TCP, z 0.14).
  Perception from that pose is out of distribution (board 144 mm off).
- Not stack-specific; correction of the earlier claim "main never
  collapsed": 9 faulty batches across all logs, including main's
  reliability_v1 (3 of 80; its two corrupted batch_2 runs are that batch's
  two lowest scores of 20), seat_hold_v2 1/10, e1 1/12, e2 1/1, e6 1/6, e8 2/16.
- Fix (policy side, competition-legal: TCP from TF only): `return_home` at
  the start of each trial. If the TCP is more than `home_tolerance_m`
  (0.05) from `home_tcp`, rise vertically to home height, then move home,
  then perceive. Start distance is logged in every record ("start" phase).
- Seat hold removed (code, schema, config): score-neutral in E9.
- E10: faulty start reproduced by setting the engine's home joints to the
  pose recorded at bench_30's faulty start (5 trials).
- E10 result (`runs/experiments/e10_return_home/run_1`): return_home fired on
  4 of 5 trials (start 276-282 mm off; the first trial began at true home)
  and brought the TCP back to within 0.6-0.9 mm of home in 14 s (one SC
  trial 166 mm, still moving); perception afterwards normal (2.5-4 mm port
  error). All 4 still failed (57 total): the plug sits in the hand far
  from the calibrated grasp (13-20 mm, ~82 deg in this test). The real
  faulty batches show the same second effect, smaller: E8 run 3 batch 2
  plug estimate error 7-15 mm and 12-22 deg after the fault (normal: <2 mm,
  <1 deg). So a faulty reset breaks two things, the start pose (now
  handled) and the grasp (the cable spawns at the gripper while the arm is
  elsewhere). Fixing the second needs a per-trial grasp measurement
  (position and rotation) from the wrist cameras; the refiner's
  grasp_update covers <= 6 mm and no rotation.
