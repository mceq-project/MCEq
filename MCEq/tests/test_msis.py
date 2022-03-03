from __future__ import print_function

result_expected = """6.665177E+05 1.138806E+08 1.998211E+07 4.022764E+05 3.557465E+03 4.074714E-15 3.475312E+04 4.095913E+06 2.667273E+04 1.250540E+03 1.241416E+03 
3.407293E+06 1.586333E+08 1.391117E+07 3.262560E+05 1.559618E+03 5.001846E-15 4.854208E+04 4.380967E+06 6.956682E+03 1.166754E+03 1.161710E+03 
1.123767E+05 6.934130E+04 4.247105E+01 1.322750E-01 2.618848E-05 2.756772E-18 2.016750E+04 5.741256E+03 2.374394E+04 1.239892E+03 1.239891E+03 
5.411554E+07 1.918893E+11 6.115826E+12 1.225201E+12 6.023212E+10 3.584426E-10 1.059880E+07 2.615737E+05 2.819879E-42 1.027318E+03 2.068878E+02 
1.851122E+06 1.476555E+08 1.579356E+07 2.633795E+05 1.588781E+03 4.809630E-15 5.816167E+04 5.478984E+06 1.264446E+03 1.212396E+03 1.208135E+03 
8.673095E+05 1.278862E+08 1.822577E+07 2.922214E+05 2.402962E+03 4.355866E-15 3.686389E+04 3.897276E+06 2.667273E+04 1.220146E+03 1.212712E+03 
5.776251E+05 6.979139E+07 1.236814E+07 2.492868E+05 1.405739E+03 2.470651E-15 5.291986E+04 1.069814E+06 2.667273E+04 1.116385E+03 1.112999E+03 
3.740304E+05 4.782720E+07 5.240380E+06 1.759875E+05 5.501649E+02 1.571889E-15 8.896776E+04 1.979741E+06 9.121815E+03 1.031247E+03 1.024848E+03 
6.748339E+05 1.245315E+08 2.369010E+07 4.911583E+05 4.578781E+03 4.564420E-15 3.244595E+04 5.370833E+06 2.667273E+04 1.306052E+03 1.293374E+03 
5.528601E+05 1.198041E+08 3.495798E+07 9.339618E+05 1.096255E+04 4.974543E-15 2.686428E+04 4.889974E+06 2.805445E+04 1.361868E+03 1.347389E+03 
1.375488E+14 0.000000E+00 2.049687E+19 5.498695E+18 2.451733E+17 1.261066E-03 0.000000E+00 0.000000E+00 0.000000E+00 1.027318E+03 2.814648E+02 
4.427443E+13 0.000000E+00 6.597567E+18 1.769929E+18 7.891680E+16 4.059139E-04 0.000000E+00 0.000000E+00 0.000000E+00 1.027318E+03 2.274180E+02 
2.127829E+12 0.000000E+00 3.170791E+17 8.506280E+16 3.792741E+15 1.950822E-05 0.000000E+00 0.000000E+00 0.000000E+00 1.027318E+03 2.374389E+02 
1.412184E+11 0.000000E+00 2.104370E+16 5.645392E+15 2.517142E+14 1.294709E-06 0.000000E+00 0.000000E+00 0.000000E+00 1.027318E+03 2.795551E+02 
1.254884E+10 0.000000E+00 1.874533E+15 4.923051E+14 2.239685E+13 1.147668E-07 0.000000E+00 0.000000E+00 0.000000E+00 1.027318E+03 2.190732E+02 
5.196477E+05 1.274494E+08 4.850450E+07 1.720838E+06 2.354487E+04 5.881940E-15 2.500078E+04 6.279210E+06 2.667273E+04 1.426412E+03 1.408608E+03 
4.260860E+07 1.241342E+11 4.929562E+12 1.048407E+12 4.993465E+10 2.914304E-10 8.831229E+06 2.252516E+05 2.415246E-42 1.027318E+03 1.934071E+02 


DAY            172          81         172         172         172
UT           29000       29000       75000       29000       29000
ALT            400         400        1000         100         400
LAT             60          60          60          60           0
LONG           -70         -70         -70         -70         -70
LST             16          16          16          16          16
F107A          150         150         150         150         150
F107           150         150         150         150         150


TINF       1250.54     1166.75     1239.89     1027.32     1212.40
TG         1241.42     1161.71     1239.89      206.89     1208.14
HE       6.665e+05   3.407e+06   1.124e+05   5.412e+07   1.851e+06
O        1.139e+08   1.586e+08   6.934e+04   1.919e+11   1.477e+08
N2       1.998e+07   1.391e+07   4.247e+01   6.116e+12   1.579e+07
O2       4.023e+05   3.263e+05   1.323e-01   1.225e+12   2.634e+05
AR       3.557e+03   1.560e+03   2.619e-05   6.023e+10   1.589e+03
H        3.475e+04   4.854e+04   2.017e+04   1.060e+07   5.816e+04
N        4.096e+06   4.381e+06   5.741e+03   2.616e+05   5.479e+06
ANM 0    2.667e+04   6.957e+03   2.374e+04   2.820e-42   1.264e+03
RHO      4.075e-15   5.002e-15   2.757e-18   3.584e-10   4.810e-15


DAY            172         172         172         172         172
UT           29000       29000       29000       29000       29000
ALT            400         400         400         400         400
LAT             60          60          60          60          60
LONG             0         -70         -70         -70         -70
LST             16           4          16          16          16
F107A          150         150          70         150         150
F107           150         150         150         180         150


TINF       1220.15     1116.39     1031.25     1306.05     1361.87
TG         1212.71     1113.00     1024.85     1293.37     1347.39
HE       8.673e+05   5.776e+05   3.740e+05   6.748e+05   5.529e+05
O        1.279e+08   6.979e+07   4.783e+07   1.245e+08   1.198e+08
N2       1.823e+07   1.237e+07   5.240e+06   2.369e+07   3.496e+07
O2       2.922e+05   2.493e+05   1.760e+05   4.912e+05   9.340e+05
AR       2.403e+03   1.406e+03   5.502e+02   4.579e+03   1.096e+04
H        3.686e+04   5.292e+04   8.897e+04   3.245e+04   2.686e+04
N        3.897e+06   1.070e+06   1.980e+06   5.371e+06   4.890e+06
ANM 0    2.667e+04   2.667e+04   9.122e+03   2.667e+04   2.805e+04
RHO      4.356e-15   2.471e-15   1.572e-15   4.564e-15   4.975e-15


DAY            172         172         172         172         172
UT           29000       29000       29000       29000       29000
ALT              0          10          30          50          70
LAT             60          60          60          60          60
LONG           -70         -70         -70         -70         -70
LST             16          16          16          16          16
F107A          150         150         150         150         150
F107           150         150         150         150         150


TINF       1027.32     1027.32     1027.32     1027.32     1027.32
TG          281.46      227.42      237.44      279.56      219.07
HE       1.375e+14   4.427e+13   2.128e+12   1.412e+11   1.255e+10
O        0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00
N2       2.050e+19   6.598e+18   3.171e+17   2.104e+16   1.875e+15
O2       5.499e+18   1.770e+18   8.506e+16   5.645e+15   4.923e+14
AR       2.452e+17   7.892e+16   3.793e+15   2.517e+14   2.240e+13
H        0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00
N        0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00
ANM 0    0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00
RHO      1.261e-03   4.059e-04   1.951e-05   1.295e-06   1.148e-07
"""


def test_msis():
    from ctypes import c_int, c_double, pointer, byref
    from MCEq.geometry.nrlmsise00 import (
        msis,
        nrlmsise_output,
        nrlmsise_input,
        nrlmsise_flags,
        ap_array,
    )

    output = [nrlmsise_output() for i in range(17)]
    inp = [nrlmsise_input() for i in range(17)]
    flags = nrlmsise_flags()
    aph = ap_array()

    # Inp values
    for i in range(7):
        aph.a[i] = c_double(100.0)

    flags.switches[0] = c_int(0)

    for i in range(1, 24):
        flags.switches[i] = c_int(1)

    for i in range(17):
        inp[i].doy = c_int(172)  # Day of year
        inp[i].year = c_int(0)  # No effect
        inp[i].sec = c_double(29000.0)
        inp[i].alt = c_double(400.0)
        inp[i].g_lat = c_double(60.0)
        inp[i].g_long = c_double(-70.0)
        inp[i].lst = c_double(16.0)
        inp[i].f107A = c_double(150.0)
        inp[i].f107 = c_double(150.0)
        inp[i].ap = c_double(4.0)

    inp[1].doy = c_int(81)
    inp[2].sec = c_double(75000.0)
    inp[2].alt = c_double(1000.0)
    inp[3].alt = c_double(100.0)
    inp[10].alt = c_double(0.0)
    inp[11].alt = c_double(10.0)
    inp[12].alt = c_double(30.0)
    inp[13].alt = c_double(50.0)
    inp[14].alt = c_double(70.0)
    inp[16].alt = c_double(100.0)
    inp[4].g_lat = c_double(0.0)
    inp[5].g_long = c_double(0.0)
    inp[6].lst = c_double(4.0)
    inp[7].f107A = c_double(70.0)
    inp[8].f107 = c_double(180.0)
    inp[9].ap = c_double(40.0)
    inp[15].ap_a = pointer(aph)
    inp[16].ap_a = pointer(aph)
    for i in range(15):
        #     msis.gtd7(byref(inp[i]), byref(flags), byref(output[i]))
        msis.gtd7_py(
            inp[i].year,
            inp[i].doy,
            inp[i].sec,
            inp[i].alt,
            inp[i].g_lat,
            inp[i].g_long,
            inp[i].lst,
            inp[i].f107A,
            inp[i].f107,
            inp[i].ap,
            inp[15].ap_a,
            byref(flags),
            byref(output[i]),
        )
    flags.switches[9] = -1
    for i in range(15, 17):
        msis.gtd7_py(
            inp[i].year,
            inp[i].doy,
            inp[i].sec,
            inp[i].alt,
            inp[i].g_lat,
            inp[i].g_long,
            inp[i].lst,
            inp[i].f107A,
            inp[i].f107,
            inp[i].ap,
            inp[15].ap_a,
            byref(flags),
            byref(output[i]),
        )
    #     msis.gtd7(byref(inp[i]), byref(flags), byref(output[i]))
    # output type 1
    outbuf = ""
    for i in range(17):
        for j in range(9):
            outbuf += "{0:E} ".format(output[i].d[j])
        outbuf += "{0:E} ".format(output[i].t[0])
        outbuf += "{0:E} \n".format(output[i].t[1])

    # output type 2
    for i in range(3):
        outbuf += "\n"
        outbuf += "\nDAY   "
        for j in range(5):
            outbuf += "         {0:3}".format(inp[i * 5 + j].doy.value)
        outbuf += "\nUT    "
        for j in range(5):
            outbuf += "       {0:5.0f}".format(inp[i * 5 + j].sec.value)
        outbuf += "\nALT   "
        for j in range(5):
            outbuf += "        {0:4.0f}".format(inp[i * 5 + j].alt.value)
        outbuf += "\nLAT   "
        for j in range(5):
            outbuf += "         {0:3.0f}".format(inp[i * 5 + j].g_lat.value)
        outbuf += "\nLONG  "
        for j in range(5):
            outbuf += "         {0:3.0f}".format(inp[i * 5 + j].g_long.value)
        outbuf += "\nLST   "
        for j in range(5):
            outbuf += "       {0:5.0f}".format(inp[i * 5 + j].lst.value)
        outbuf += "\nF107A "
        for j in range(5):
            outbuf += "         {0:3.0f}".format(inp[i * 5 + j].f107A.value)
        outbuf += "\nF107  "
        for j in range(5):
            outbuf += "         {0:3.0f}".format(inp[i * 5 + j].f107.value)
        outbuf += "\n\n"
        outbuf += "\nTINF  "
        for j in range(5):
            outbuf += "     {0:7.2f}".format(output[i * 5 + j].t[0])
        outbuf += "\nTG    "
        for j in range(5):
            outbuf += "     {0:7.2f}".format(output[i * 5 + j].t[1])
        outbuf += "\nHE    "
        for j in range(5):
            outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[0])
        outbuf += "\nO     "
        for j in range(5):
            outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[1])
        outbuf += "\nN2    "
        for j in range(5):
            outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[2])
        outbuf += "\nO2    "
        for j in range(5):
            outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[3])
        outbuf += "\nAR    "
        for j in range(5):
            outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[4])
        outbuf += "\nH     "
        for j in range(5):
            outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[6])
        outbuf += "\nN     "
        for j in range(5):
            outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[7])
        outbuf += "\nANM 0 "
        for j in range(5):
            outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[8])
        outbuf += "\nRHO   "
        for j in range(5):
            outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[5])
        outbuf += "\n"
    outbuf += "\n"

    assert outbuf.strip() == result_expected.strip()
