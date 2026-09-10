Ruff is a CI gate. `.github/workflows/lint.yaml` runs `ruff check` and
`ruff format --check` pinned to the version `.pre-commit-config.yaml` uses, so
the hooks and CI cannot disagree. The ten outstanding lint errors are fixed and
62 files are formatted; `examples/**` joins `docs/examples/**` in a top-level
`[tool.ruff] exclude`, since notebooks are demonstrations and one of them does
not parse as Python source.
