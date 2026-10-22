(data-installation)=

# Data installation

MCEq reads its yields, cross sections, decays and energy-loss tables from HDF5
databases. They are downloaded on first use into `config.data_dir` (by default
the `MCEq/data` directory of the installed package). The v2 data are split into
packages, listed with their checksums and models in a YAML manifest.

## The v2 data release

| file | grid | contents |
|---|---|---|
| `mceq_base_air_1d_v2.h5` | 1D, 150 bins (1 MeV–1 EeV) | decays, continuous losses (air, hydrogen, ice, rock, water), FLUKA20251 and SIBYLL23E for air, SIBYLL23E for hydrogen. **Default 1D database.** |
| `mceq_extra_models_air_1d_v2.h5` | 1D, 150 bins | the same tables plus DPMJETIII193, EPOSLHC, EPOSLHCR, PYTHIA8CASCADE, PYTHIA8ANGANTYR, QGSJETII04, QGSJETIII, SIBYLL21, SIBYLL23D, SIBYLL23ESTAR{BAR,MIXED,RHO,STRANGE} |
| `mceq_base_air_2d_legendre_v2.h5` | 2D, 150 bins × 90 Legendre modes | 2D decays, continuous losses, FLUKA20251 (both hemispheres), muon scattering rates. **Default 2D database.** |
| `mceq_model_SIBYLL23E_air_2d_legendre_v2.h5` | 2D, 150 × 90 | SIBYLL23E 2D yields only; used together with the 2D base file. |
| `mceq_db_manifest_v2.yaml` | — | grid, sha256, size and models per medium of each file |

The FLUKA20251 yields cover projectile energies up to 10^5 GeV. Use FLUKA as the
low-energy model together with a high-energy model, for example
`interaction_model="SIBYLL23E", low_energy_model="FLUKA20251"`. SIBYLL23E yields
start at 70 GeV.

The files are assets of the `v2.0.0` release of the MCEq repository, each with a
`.sha256` file. A copy of the manifest is installed with the package.

2D databases carry the angular distribution as Legendre moments
⟨P_l(cos θ)⟩.

## How a run finds its data

`config.mceq_db_fname` names the primary database. It provides the energy grid,
decays and continuous losses, and the models it contains. A model that is not in
the primary database is looked up in the manifest (`config.mceq_db_manifest`,
read from `config.data_dir` if present, else the installed copy):

```python
from MCEq import config
from MCEq.core import MCEqRun
import crflux.models as pm

config.mceq_db_fname = "mceq_base_air_1d_v2.h5"
run = MCEqRun(interaction_model="SIBYLL23ESTARRHO", theta_deg=0.0,
              primary_model=(pm.HillasGaisser2012, "H3a"))
```

Here `mceq_extra_models_air_1d_v2.h5` is downloaded if missing and the yields and
cross sections of SIBYLL23ESTARRHO are read from it. Decays and losses always
come from the primary database. The result is identical to a run on a single
file that contains all models.

A download is checked against the sha256 in the manifest before it is put in
place. When several jobs start at once, one downloads and the others stop with a
message to pre-fetch the data (below).

A file already on disk is checked against its manifest entry whenever a run
uses it: its size, and the digest of the database it was cut from (the root
attribute `split_from_sha256` against the entry's `source.sha256`). A copy left
over from an earlier release under the same name is refused with
`StaleDataError`; delete it and the run downloads the listed file. A local build
under a release name passes with `config.verify_data_packages = False`. A
manifest in a separate `config.data_dir` that is older than the installed copy
gives `StaleManifestWarning`. For a full check of every file on disk:

```bash
python -m MCEq.data.download --verify             # exit 1 if a file is stale
```

For 2D runs point `config.mceq_db_fname` at a Legendre 2D base file; a model
it does not hold is resolved through the manifest in the same way, from the
packages on its grid.

If no file in the release provides the requested model, the error lists the
models that are available.

## Pre-fetching on clusters

```bash
# the default 1D and 2D databases
python -m MCEq.data.download --defaults
# the files that provide given models (1D by default, --dim 2d for 2D)
python -m MCEq.data.download --model SIBYLL23ESTARRHO --model QGSJETIII
# one file by name, or all files (~2 GB)
python -m MCEq.data.download --file mceq_extra_models_air_1d_v2.h5
python -m MCEq.data.download --all
# into the directory that config.data_dir points to
python -m MCEq.data.download --defaults --dir /project/mceq-data
```

## Adding your own model file

Add an entry to the manifest in `config.data_dir` (copy the installed manifest
there first):

```yaml
files:
  my_model_air_1d.h5:
    path: /data/mceq/my_model_air_1d.h5   # local file, never downloaded
    grid: 1d-150                          # a grid defined in the manifest
    models:
      air: [MYMODEL]
```

The file needs the `common` group of the base file and the datasets
`hadronic_interactions/air/MYMODEL`, `hadronic_interactions/air/MYMODEL_indptrs`
and `cross_sections/air/MYMODEL`, in the same layout as the release files.

## Test data

The test suite and the example notebooks use `mceq_ci_*` databases with
31 energy bins from 891 GeV to 891 TeV, listed in `mceq_db_manifest_ci_v2.yaml`.
They are for tests only, not for physics. `tests/data/ci_assets.py` pins them
by sha256: every `pytest` session first replaces files that differ from the
pins (`python tests/data/ci_assets.py` does the same by hand, `--check` only
reports), and the CI workflows fetch through it.

## Older databases

v2 reads the single-file v1.4 databases (for example
`mceq_db_lext_dpm193_v142.h5`): set `config.mceq_db_fname` to the file name. A
single file is used as is, without the manifest.
