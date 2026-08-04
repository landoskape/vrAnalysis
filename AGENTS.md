# Codex project instructions

## Testing

- Run tests with the `vrAnalysis` Conda environment, not the system Python. On this Windows workstation the interpreter is `C:\Users\Andrew\AppData\Local\miniforge3\envs\vrAnalysis\python.exe`.
- Disable automatic third-party pytest plugin loading for ordinary repository tests. In PowerShell, set `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"` before invoking `python -m pytest`. Plugin autoload has caused focused tests to exceed four minutes, whereas the same tests complete in seconds with it disabled.
- Enable plugin autoload only when a test explicitly depends on an installed pytest plugin.
- Prefer focused tests for the changed code before broader suites.
- `speedystats>=0.1.1` no longer uses Numba's disk cache for its generated functions. Its import now takes roughly two seconds in the managed Codex environment; do not assume the former multi-minute `speedystats` cache stall is still present unless measurements show a regression.
