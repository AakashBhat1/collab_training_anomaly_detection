# Current Step

Last updated: 2026-04-10 00:18 (local)
Owner: GitHub Copilot

## Where We Are
- Step ID: PHASE2-COLAB-GITHUB-SCRIPTS
- Status: READY
- Summary: GitHub-oriented Colab shell wrappers are now in place for clone/bootstrap, run, and export packaging.

## Completed In This Pass
- Added mp4-to-frame extraction in dataset materialization.
- Added stale split cleanup and split-plan key validation.
- Added split-ratio bounds validation for safer split planning inputs.
- Added split contract tests and raised changed-module coverage above 80%.
- Updated bootstrap and README for both supported folder layouts.
- Added GitHub-first Colab helper shell scripts and README command examples.
- Added working-session docs and verification log.

## Next Exact Action
- Command: `bash collab_scripts/colab_clone_and_bootstrap.sh https://github.com/<org>/<repo>.git /content/intruder_detection_system main`
- File to edit next: `collab_scripts/pipeline_config.json`
- Expected result: Repo is cloned in Colab, dependencies are installed, and environment is ready for `bash collab_scripts/colab_run_training.sh /content/intruder_detection_system`.

## If Blocked
- Blocker: Missing/invalid raw dataset layout under configured `paths.raw_dataset_dir`.
- Needed from human/partner: Provide or fix class-labeled clips for `fight`, `theft`, `intrusion`, and `normal` in the configured dataset root.
