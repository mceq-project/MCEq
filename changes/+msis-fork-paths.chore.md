`path_workers > 1` now works with MSIS atmospheres. The paths a forked worker
returns are bitwise equal to the serial ones (16/16 conditions, 7.8x on four
workers); what used to look like nrlmsise-00 not being fork-safe was the
altitude memo making the azimuth-averaged spline depend on the order the
directions were evaluated in, which also affected serial runs.
