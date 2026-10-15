The NRLMSISE-00 and NRLMSIS 2.1 atmospheres now behave the same and differ
only in the model. MSIS21 samples local noon at each site, as MSIS00 always
has; it used to sample 12:00 UT, which is a night atmosphere at Tokyo. Both
backends pass the height through without clamping, accept day of year 0 to
366, and raise `ValueError` for an unknown location or season, coordinates
out of range, or an invalid day. When `season` and `doy` are both given, `doy`
now wins at construction too, as it already did in `update_parameters`.
