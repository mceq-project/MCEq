The example notebooks are a CI gate. `.github/workflows/notebooks.yaml` runs
`pytest --nbmake --nbmake-kernel=python3` over `docs/examples/` and `examples/`
from the new `notebooks` dependency group, forcing the kernel so a notebook's
stale `python2` or `mceq_dev_396` kernelspec cannot decide which interpreter
executes it.

Ten of the eleven notebooks are `--ignore`d, one line each with the reason in
the workflow: seven request an interaction model no shipped database carries
(`SIBYLL23C`, `DPMJETIII191`) or one absent from the reduced CI database
(`SIBYLL23D`), one is Python 2 source, one imports the pre-`crflux`
`CRFluxModels`, and one needs the 2D FLUKA database. Only
`Plot_density_depth_relations.ipynb` executes; it touches no database. A
separate step validates the notebook JSON of all eleven, since `--ignore`
prunes a path before collection and the ignored notebooks are otherwise never
opened. The ignore list is the shrink list.
