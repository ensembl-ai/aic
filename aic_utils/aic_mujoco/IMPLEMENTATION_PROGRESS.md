# MJWarp foundation implementation checkpoint

Last updated: 2026-08-09

## Objective

Produce the smallest mergeable AIC SFP/NIC simulation foundation with strict JSON configuration, independent MJWarp worlds, AIC HOLD control, native RGB cameras, native wrist F/T observations, per-world domain randomization, visualization, and recording. Runtime CPU MuJoCo stepping/rendering, cables, plugins, prototype training code, CLI options, fallbacks, and duplicate configuration are out of scope.

## Locked decisions

- Preserve the canonical `main` AIC XML conversion outputs; generate one reduced MJWarp scene from them.
- Remove the cable, LC plug, cable plugin, SC task objects, distractors, enclosure, walls, and floor from the reduced scene.
- Retain the six UR5e joints, three AIC cameras, wrist force/torque sensors, fixed gripper geometry, standalone SFP, board, target NIC, and explicit light.
- Fix the gripper and SFP in MJCF; use six arm actuators only.
- Bake the AIC 0.0073 m SFP grasp into both fixed finger body transforms.
- Exclude internal gripper/SFP contacts while retaining SFP/NIC contact.
- Use batched mocap poses for independently randomized board and target NIC fixtures.
- Use `mjw.step1 -> AIC HOLD Warp kernel -> mjw.step2` so control uses current `qfrc_bias`.
- Use AIC HOME and AIC HOLD gains from `aic_engine`.
- Use AIC RGB only: 1152 x 1024, 20 Hz, center/left/right.
- Use per-world raw and tared six-axis wrench observations.
- Use `base.json` plus `run.json`, recursively deep merged and strictly validated with no fallbacks.
- Initial configurable joint reset offsets are `[-0.02, 0.02]` radians per joint because AIC specifies no reset-noise range.

## Work phases

- [x] Inventory canonical files and exact source body/asset relationships.
- [x] Restore unrelated branch modifications and delete branch-only prototypes.
- [x] Implement strict configuration and schema validation.
- [x] Implement deterministic reduced-scene preparation.
- [x] Implement MJWarp runtime/reset/HOLD/cameras/F/T/taring.
- [x] Implement visualization and recording path without CPU physics/rendering.
- [x] Add tests and run full verification.
- [x] Replace README with algorithmic-design and software-design sections.
- [x] Final diff and minimum-file audit.
- [x] Restore explicit robot/joints/command/controller boundaries with only the HOLD path.
- [x] Make the rollout continuous until Ctrl+C and add real reduced-scene meshes to Viser.
- [x] Support explicit `visualization.env_ids: "all"` for displaying all `N` worlds.

## Verification evidence

- Reduced MJCF: nq=6, nv=6, nu=6, ncam=3, nsensor=2,
  nsensordata=6, nmocap=2, nplugin=0.
- CPU MJWarp contract test (not CPU MuJoCo): 2 independent worlds, 3 non-empty
  RGB tensors, raw/tared wrench tensors, stable HOLD, selected reset, and
  asynchronous per-world retare.
- `3 passed in 5.38s` after the Warp kernel cache was populated.
- Final hardening added compiled-physics/actuator validation, per-world episode
  clocks, selected-image device gathering, atomic scene generation, and the
  fixed AIC SFP finger transform.
- `git diff --check` and the final shell-level source/config audit pass.
- The post-refactor foundation suite passes: `3 passed` using the
  already-installed pinned environment interpreter.
- A targeted one-world CPU-MJWarp smoke test constructed and stopped the new
  Viser mesh path successfully. This was output validation only; the
  committed application configuration remains CUDA-only.
