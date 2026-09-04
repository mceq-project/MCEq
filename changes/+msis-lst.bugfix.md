`cNRLMSISE00` wrote the local apparent solar time `inp.lst` once at
construction and never updated it, so every site was evaluated at the *default*
location's solar time — 12.0, the South Pole at longitude 0 — while NRLMSISE-00
documents `lst`, `sec` and `g_long` as inputs that should be kept consistent
(`lst = sec/3600 + g_long/15`). `set_location_coord` now recomputes it.

Densities move: over the 12 shipped `LOCATIONS` × 12 months, up to 9.34 h of
solar time (Tsukuba), 0.41 % at sea level (LynnLake, January), 22.97 % at 95 km
(LynnLake, July) and up to 46 % near 107 km (KSC, July). Column depth `max_X`
moves by at most 4.1e-03 relative. The error was exactly zero at longitude 0,
so South Pole results — including every IceCube one — are unchanged.

`sec` is UT and has no setter, so consistency is restored by moving `lst`:
sites are now evaluated at 12:00 UT rather than at local noon, which is what
MSIS 2.1 already did. MSIS 2.1 was never affected.
