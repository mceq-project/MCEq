`cNRLMSISE00` wrote the local apparent solar time `inp.lst` once at
construction and never updated it, so `set_location`/`set_location_coord` left
it holding the previous site's value while `inp.sec` stayed at 43200 — breaking
the consistency NRLMSISE-00 asks for between `lst`, `sec` and `g_long`
(`lst = sec/3600 + g_long/15`).

Named locations are now evaluated at **local noon**: `lst` is held at 12.0 and
`sec`, which is UT and has no setter, is derived from the longitude. The old
code already pinned `lst` at 12.0, so this restores column depths to within
~1e-9 relative of their pre-fix values; what changes is the model's separate
UT term, up to 0.6 % in density near 110 km and 0.07 % at 95 km, and nothing
measurable at sea level. South Pole results, including every IceCube one, are
unchanged.

Note that MSIS 2.1 derives `lst` from a fixed UT second instead, so it samples
12:00 UT. At non-zero longitude the two backends therefore describe different
times of day; the gap is pinned by
`tests/geometry/test_environment_pins.py::test_msis00_is_local_noon_and_msis21_is_ut_noon`.
