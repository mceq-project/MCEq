`cNRLMSISE00` memoised its result on altitude alone, so `set_location`,
`set_location_coord`, `set_season` and `set_doy` were silently ignored when the
next `get_density` used the altitude of the previous call. A day-of-year change
from 1 to 200 returned the old density, 12.5 % high at 20 km, and the
azimuth-averaged `MSIS00LocationCentered.get_density` averaged a corrupted set
of directions. The setters now drop the memo. MSIS 2.1 was never affected.
