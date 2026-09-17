Data now ships as **packages described by a YAML manifest** instead of one
monolithic file: `mceq_base_air_1d_v2.h5` (grid, decays, losses, FLUKA20251 +
SIBYLL23E), `mceq_extra_models_air_1d_v2.h5` (all other 1D models),
`mceq_base_air_2d_v2.h5` (2D FLUKA20251 on the 150-bin × 48-mode lattice),
`mceq_model_SIBYLL23E_air_2d_v2.h5` and the paper's 2D `rc7` database, with
`mceq_db_manifest_v2.yaml` recording grid, sha256 and models per medium for
each. `config.mceq_db_fname` names the primary (spine) database; a model it
lacks is resolved through the manifest (`config.mceq_db_manifest`) to the
package on the same energy grid, fetched once and race-safely (temp file +
sha256 + atomic rename + lock sentinel with an actionable collision message),
and read after a byte-identical energy-grid check. Registering a new model is
one manifest entry pointing at a single-model file. `python -m
MCEq.data.download --defaults | --model M [--dim 2d] | --file NAME | --all`
pre-fetches packages before submitting cluster jobs. Monolithic databases are
unaffected and remain the default; `pyyaml` becomes a dependency. The test suite, goldens and example notebooks run on a CI-sized cut of the release data (`mceq_ci_*_v2.h5`, `ci-assets`), cached in CI under a manifest-digest key.
