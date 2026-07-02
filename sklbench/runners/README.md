# Benchmark Runner

`sklbench.runners.*` modules execute one already-expanded benchmark case.

It is intentionally separate from config parsing and orchestration:

- a Python config script generates validated benchmark cases;
- the orchestrator records environments, launches runner subprocesses, captures
  logs/errors, and writes result files;
- the runner validates one case file, loads data, runs repetitions, and writes
  JSONL.

## CLI Contract

```bash
python -m sklbench.runners.estimator \
  --case-file /path/to/case.json \
  --n-runs 3 \
  --output-jsonl /tmp/sklbench-result.jsonl
```

Each runner accepts `--case-file`, `--n-runs`, and `--output-jsonl`. The output
file contains one JSON object per repetition. The orchestrator owns environment
capture, timeout handling, stdout/stderr capture, and final record writing.
