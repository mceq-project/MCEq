"""Optional elastic collision tables, kept separate from inelastic rates.

Kernels use differential-flux normalization, including the parent/child bin
width ratio. Model mixing acts on cross sections and cross-section-weighted
kernels together, never on an inelastic collision clock.
"""

from dataclasses import dataclass, field

import numpy as np

from MCEq.data.model_names import normalize_hadronic_model_name


@dataclass
class ElasticTables:
    cross_sections: dict = field(default_factory=dict)
    production: dict = field(default_factory=dict)

    @property
    def particles(self):
        return sorted(set(self.cross_sections) | {c for _, c in self.production})

    def restrict_parents(self, parents):
        """Match an explicit interaction parent list, retaining its daughters."""
        allowed = set(parents)
        return ElasticTables(
            {p: s for p, s in self.cross_sections.items() if p in allowed},
            {key: v for key, v in self.production.items() if key[0] in allowed},
        )


def load_elastic(backend, model):
    """Return mb loss columns and mb-weighted redistribution on the active grid.

    An absent group preserves legacy behavior. A present group must cover all
    active parent columns with nonzero model weight. There are no implicit
    material or projectile fallbacks. Both 1D and 2D use the DB's channel layout.
    """
    result = ElasticTables()
    if backend._em_standalone:
        return result
    model = normalize_hadronic_model_name(model)
    le = backend.low_energy_model
    selections = [(model, np.ones(backend._energy_grid.d))]
    if le is not None and le != model:
        weight = backend._he_le_weight()
        selections = [(model, weight), (le, 1.0 - weight)]
    for name, weight in selections:
        path = f"elastic_scattering/{backend.medium}/{name}"
        if not backend._had.has(path):
            continue
        attrs = backend._had.attrs(path)
        required = dict(
            schema_version=1,
            process="elastic",
            cross_section_units="mb",
            energy_kind="kinetic",
            normalization="differential-per-collision",
        )
        if any(attrs.get(k) != v for k, v in required.items()):
            raise ValueError(f"Invalid elastic table metadata at {path}")
        sig = backend._had.read_dataset(path + "/cross_sections")
        parents = backend._had.read_dataset(path + "/projectiles")
        covered = backend._had.read_dataset(path + "/coverage")
        if (
            parents.ndim != 2
            or parents.shape[1] != 2
            or sig.shape != (len(backend._e_grid_full), len(parents))
            or covered.shape != sig.shape
            or covered.dtype.kind != "b"
            or parents.dtype.kind not in "iu"
            or len(set(map(tuple, parents))) != len(parents)
            or not np.isfinite(sig).all()
            or np.any(sig < 0)
        ):
            raise ValueError(f"Invalid elastic cross sections or coverage at {path}")
        sig, covered = sig[backend._cuts], covered[backend._cuts]
        disabled = backend._physics.filters["disabled_particles"]
        selected = {}
        for j, p in enumerate(parents):
            p = tuple(map(int, p))
            if abs(p[0]) in disabled:
                continue
            if np.any((weight > 0) & ~covered[:, j]):
                raise ValueError(
                    f"Elastic coverage missing for {p} at {path}; restrict the energy grid"
                )
            selected[p] = sig[:, j] * weight
            result.cross_sections[p] = result.cross_sections.get(p, 0) + selected[p]
        index = backend._gen_db_dictionary(
            backend._had.read_channel_pack(path, "yields")
        )
        for (p, c), kernel in index["index_d"].items():
            if p not in selected:
                raise ValueError(f"Elastic kernel has no cross section for {p}")
            if not np.isfinite(kernel).all():
                raise ValueError(f"Nonfinite elastic kernel for {p}, {c}")
            zero = kernel[0] if backend.is_2d else kernel
            if np.any(zero < -1e-12) or np.any(np.abs(kernel) > zero + 1e-10):
                raise ValueError(f"Invalid elastic Hankel moments for {p}, {c}")
            key = (p, c)
            result.production[key] = (
                result.production.get(key, 0) + kernel * selected[p]
            )
    return result
