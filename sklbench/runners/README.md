# Benchmark Runner

The `sklbench.runners.*` modules run one expanded benchmark case.

They are kept separate from config parsing and orchestration:

- a Python config script generates validated benchmark cases;
- the orchestrator records environments, launches runner subprocesses,
  captures logs and errors, and writes result files;
- the runner validates one case file, loads data, runs the repetitions, and
  writes JSONL.

## CLI Contract

```bash
python -m sklbench.runners.estimator \
  --case-file /path/to/case.json \
  --n-runs 3 \
  --output-jsonl /tmp/sklbench-result.jsonl
```

Each runner accepts `--case-file`, `--n-runs` and `--output-jsonl`. The output
file has one JSON object per repetition. The orchestrator handles environment
capture, timeouts, stdout/stderr capture and writing the final record.
