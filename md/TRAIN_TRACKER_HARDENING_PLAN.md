# Plan: Harden `train_tracker.py` Into a Correct Baseline

## Summary
Fix the tracker training pipeline in two passes.

Pass 1 is correctness-only and should land first: make validation truly held-out, make the script safe for its advertised configs, and align DDP/runtime behavior with the existing `train.py` conventions.

Pass 2 is robustness/usability: remove misleading dead config paths, make evaluation mode-agnostic, and add guardrails around known edge cases so failures are explicit instead of silent.

## Key Changes

### 1. Validation and split correctness
- Change the validation construction in `train_tracker.py` so eval scenes come from `val_split`, not `train_split`.
- Implement this in the dataset layer, not as an ad hoc subset hack in the trainer.
- Add an explicit dataset mode or split selector for EVS datasets so training and validation can be constructed unambiguously from the same codepath.
- Preserve the current `strict_split=False` default unless the config explicitly asks otherwise.
- Make `eval_seqs` actually cap the number of evaluated sequences after unique-scene selection.
- Keep the current "one window per sequence" evaluation policy unless changed later.

### 2. Runtime/config guardrails
- Port the missing config assertions from `train.py` into `train_tracker.py`.
- At minimum enforce: `datapath` exists, `train_split` and `val_split` exist, `fgraph_pickle` exists when set, `n_frames > 7`, `gpu_num > 0`, `steps % gpu_num == 0` if using global-step semantics, and `batch == 1` until batching is truly implemented.
- If `patch_selector != scorer`, either disable score-map export/eval scorer visualization cleanly or fail fast with a clear message.
- Update docstrings and CLI help to match actual behavior.

### 3. DDP and scheduler semantics
- Match `train.py` step semantics: treat config `steps` as global steps, divide by `gpu_num` before spawn, and keep checkpoint/log naming in global-step units.
- Add `sampler.set_epoch(epoch_like_counter)` each time the distributed loader is re-entered so shard order changes across epochs.
- Keep save/eval frequency in global-step units for consistency with existing checkpoints and TensorBoard curves.
- Ensure resume restores the same interpretation of `steps` and scheduler progress.

### 4. NaN and empty-supervision handling
- Add an explicit guard in the scorer-loss branch for `valid_mask.any()`.
- If there are no valid close edges on the last iteration, skip scorer loss for that batch and report zero scorer loss.
- Keep the existing NaN bailout, but do not advance the scheduler on skipped/invalid optimization steps.
- Log a counter or warning metric for skipped batches so this failure mode is observable.

### 5. Batch-size support decision
- Short term: explicitly lock tracker training to `batch=1` with an assertion and config/docs update.
- Do not attempt partial "support" for `batch>1`; the current patch extraction/scoring path is structurally batch-0-centric and needs a deliberate rewrite.
- Create a follow-up task for real batched support in `Patchifier` and any downstream score indexing logic, with no behavior changes in this fix set.

### 6. Evaluation cleanup
- Make `evaluate_tracker()` conditional on selector capabilities.
- For scorer mode, keep score-map PNG export.
- For non-scorer modes, skip score-map generation and still compute final EPE/tracks.
- Remove or correct the stale NPZ-output claims in the eval docstring unless NPZ export is intentionally added now.
- If NPZ export is desired, add it explicitly for last-iteration `coords_est`, `coords_gt`, `valid`, and `weight`, and keep filenames stable.

## Test Plan
- Unit-level/static checks:
  - Parser/config assertions reject `n_frames <= 7`, missing splits, and `batch > 1`.
  - Validation dataset built from `val_split` contains only held-out scenes.
  - `eval_seqs` limits the number of evaluated unique sequences.
- Functional smoke checks:
  - Single-GPU run with scorer selector completes one save/eval cycle and writes checkpoint plus eval artifacts.
  - Single-GPU run with non-scorer selector completes eval without touching scorer-only codepaths.
  - DDP smoke run with `gpu_num=2` uses the intended global-step budget and reshuffles sampler order across epochs.
- Edge-case checks:
  - Batch with zero valid close edges does not produce NaN and does not advance scheduler on skipped update.
  - Resume from checkpoint preserves global-step naming and scheduler state.

## Assumptions
- The intended behavior is to keep the current tracker model and loss design, not redesign patch tracking or supervision.
- `steps`, `save_freq`, and eval reporting should follow the original `train.py` global-step convention.
- Real `batch>1` support is out of scope for this fix and should be deferred rather than approximated.
- Validation should remain lightweight: one representative window per held-out sequence, capped by `eval_seqs`.
