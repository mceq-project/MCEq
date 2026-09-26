(data-installation)=

# Data installation

MCEq ships its yield, cross-section, decay and energy-loss tables as HDF5
databases that are downloaded on first use into `config.data_dir` (the
package's `MCEq/data` directory unless you point it elsewhere). Since v2 the
data comes in **packages** described by a small YAML **manifest**, so an
install carries only the models it runs; the single-file databases of v1
keep working unchanged.

## The packages of the v2 data release

| file | grid | contents |
|---|---|---|
| `mceq_base_air_1d_v2.h5` | 1D, 150 bins (1 MeV–1 EeV) | energy grid, decays, continuous losses (air, hydrogen, ice, rock, water); FLUKA20251 and SIBYLL23E (+ SIBYLL23E on hydrogen). **Default 1D package.** |
| `mceq_extra_models_air_1d_v2.h5` | 1D, 150 bins | the same spine plus every other 1D model: DPMJETIII193, EPOSLHC, EPOSLHCR, PYTHIA8CASCADE, PYTHIA8ANGANTYR, QGSJETII04, QGSJETIII, SIBYLL21, SIBYLL23D, SIBYLL23ESTAR{BAR,MIXED,RHO,STRANGE} |
| `mceq_base_air_2d_v2.h5` | 2D, 150 bins × 48 Hankel modes | 2D spine (decays, losses) and FLUKA20251. **Default 2D package.** |
| `mceq_model_SIBYLL23E_air_2d_v2.h5` | 2D, 150 × 48 | SIBYLL23E 2D yields only (a *model file*: needs the 2D base) |
| `mceq_base_air_2d_rc7_v2.h5` | 2D, 80 bins × 48 modes | the FLUKA-only 2D database of the MCEq v2 paper, unchanged |
| `mceq_db_manifest_v2.yaml` | — | the manifest: grid, sha256, size and models per medium of every file |

A copy of the manifest ships inside the package; the files are assets of the
`v2.0.0` release of the MCEq repository and every one has a `.sha256`
sidecar.

## How a run finds its data

`config.mceq_db_fname` names the **primary** database. It supplies the energy
grid, the decay and loss tables and whatever models it holds — so it must be
a *base* package or any monolithic database. When you ask for a model the
primary does not carry:

```python
from MCEq import config
from MCEq.core import MCEqRun
import crflux.models as pm

config.mceq_db_fname = "mceq_base_air_1d_v2.h5"
run = MCEqRun(interaction_model="SIBYLL23ESTARRHO", theta_deg=0.0,
              primary_model=(pm.HillasGaisser2012, "H3a"))
```

the loader reads the manifest (`config.mceq_db_manifest`, looked up in
`config.data_dir`, else the packaged copy), picks the package on the **same
energy grid** that lists `SIBYLL23ESTARRHO` for the run's medium
(`mceq_extra_models_air_1d_v2.h5`), downloads it once if it is missing, checks
that its `common/e_grid` is byte-identical to the primary's, and reads the
yields and cross sections from it. Decays and losses always come from the
primary. Numerics are unaffected: a model solved from a package is bitwise
equal to the same model solved from the monolithic source database.

Downloads are safe for cluster jobs that start together: the file is written
to a temporary name, verified against the manifest's sha256 and renamed
atomically, and a `<file>.lock` sentinel lets exactly one process fetch while
the others fail fast with the pre-fetch commands below. No half-written
database is ever read.

For 2D runs point the primary at the 2D base:

```python
config.mceq_db_fname = "mceq_base_air_2d_v2.h5"      # FLUKA20251, 48 modes
run = MCEqRun(interaction_model="SIBYLL23E", low_energy_model="FLUKA20251", ...)
#  -> mceq_model_SIBYLL23E_air_2d_v2.h5 (1.1 GB) is fetched on first use
```

A model that no package on the primary's grid provides raises an error that
names the model, the manifest and the models the release does offer on that
grid.

## Pre-fetching on clusters

```bash
# the default 1D and 2D packages
python -m MCEq.data.download --defaults
# every package that provides a model (1D grid by default; --dim 2d for the 2D family)
python -m MCEq.data.download --model SIBYLL23ESTARRHO --model QGSJETIII
python -m MCEq.data.download --model SIBYLL23E --dim 2d
# one file by name, or everything in the manifest (~2 GB)
python -m MCEq.data.download --file mceq_extra_models_air_1d_v2.h5
python -m MCEq.data.download --all
# into a custom data directory (the one you set config.data_dir to)
python -m MCEq.data.download --defaults --dir /project/mceq-data
```

## Registering your own model file

The manifest is plain YAML and the loader trusts it, so a database of your
own becomes selectable by adding one entry to the copy in `config.data_dir`
(copy the packaged file there first if it is not present):

```yaml
files:
  my_model_air_1d.h5:
    path: /data/mceq/my_model_air_1d.h5   # local file: used in place, never fetched
    grid: 1d-150                          # must be a grid id the manifest already defines
    spine: false
    models:
      air: [MYMODEL]
```

The file needs the `common` group of the grid it declares (copy it from the
base package) and the model's `hadronic_interactions/air/MYMODEL`
(+ `_indptrs`) and `cross_sections/air/MYMODEL` datasets in the layout of the
release packages — mceq-maintenance-tools step 6 (`6_split_bundles.py`)
writes exactly this layout from any compact database.

## The CI package set

The test suite and the example notebooks run on a CI-sized cut of the same
release data: `mceq_ci_base_air_1d_v2.h5` (FLUKA20251 + SIBYLL23E),
`mceq_ci_extra_models_air_1d_v2.h5` (the other 13 models) and
`mceq_db_manifest_ci_v2.yaml`, the 1D release restricted to the 31-bin window
891 GeV – 891 TeV (assets of the `ci-assets` release; not usable for
physics). A package declares the manifest it belongs to in its root attribute
`mceq_db_manifest`, and that declaration takes precedence over
`config.mceq_db_manifest` when the file is present in `config.data_dir`, so
selecting `mceq_ci_base_air_1d_v2.h5` resolves SIBYLL21 or QGSJETII04 through
the CI manifest without further configuration.

## Monolithic databases

A single-file database (`mceq_db_lext_dpm193_v142.h5`, the retired v1.4 CI
database, your own builds) is resolved against its own contents only and
never touches the manifest, so every existing script, pin and CI cache keeps
working. The v1 default file remains the default primary for now.
