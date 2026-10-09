# Baseline v1: completed Qwen3-8B acceptance experiments

This snapshot precedes the v2 timing and long-context experiment changes.

- Engine changes are committed in serve/.
- workspace/ preserves scripts, environment setup documentation, and handoff files from the outer vericache workspace. Restore its contents two directories above this repository to reproduce the original layout.
- experiment_logs.tar.gz contains the complete log/ tree: 90 cases, per-verification rounds, output tokens, analysis, runtime logs, and proposed v2 design. Extract with `tar -xzf research/baseline_v1/experiment_logs.tar.gz` from repository root.
- Model weights, Python/CUDA environments and downloaded source datasets are external dependencies, not included. The validated 100-prompt sample and hashes are in the log archive.
- Historical PID/status files are archival records, not live jobs after restoration.
