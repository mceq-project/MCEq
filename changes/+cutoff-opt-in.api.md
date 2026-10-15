The geomagnetic cutoff of `solve_fullsky` is off by default. It used to switch
itself on for MSIS atmospheres, which made `solve_fullsky` disagree with
`solve`. Pass `geomagnetic_cutoff=True` to `MCEqRun` or `solve_fullsky` to
apply it.
