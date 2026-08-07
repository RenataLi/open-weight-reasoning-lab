# Contributing

Contributions that improve reproducibility, evaluation coverage, or experiment integrity are welcome.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest
```

## Pull requests

Please keep each pull request focused and include:

- the motivation and expected effect;
- tests for changed behavior;
- the exact command used for validation;
- updated documentation for any CLI, configuration, or artifact-schema change.

Changes that alter dataset composition, selection thresholds, confirmation semantics, or hashing rules should be treated as a new protocol version. Never rewrite an existing selection lock or reuse an exposed confirmation split as a fresh holdout.

## Data and artifacts

Do not commit model weights, raw benchmark snapshots, generated datasets, predictions, checkpoints, local caches, experiment runs, credentials, or confirmation exposure registries. Publish compact aggregate summaries under `reports/` instead.
