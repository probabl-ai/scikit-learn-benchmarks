# Benchmark Config Scripts

Each Python file in this directory defines a benchmark matrix through a
`generate_cases()` function. The function must return a list of plain
dictionaries or pydantic case models exported by `sklbench.config`.

Shared generator helpers live in `sklbench.config.generators`; keep top-level
config files focused on selecting workloads and implementations.
