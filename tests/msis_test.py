from __future__ import print_function
from ctypes import (c_int, c_double, pointer, byref)
from MCEq.nrlmsise00.nrlmsise00 import (
    msis, nrlmsise_output, nrlmsise_input, nrlmsise_flags, ap_array)

output = [nrlmsise_output() for i in range(17)]
inp = [nrlmsise_input() for i in range(17)]
flags = nrlmsise_flags()
aph = ap_array()

# Inp values
for i in range(7):
    aph.a[i] = c_double(100.)

flags.switches[0] = c_int(0)

for i in range(1, 24):
    flags.switches[i] = c_int(1)

for i in range(17):
    inp[i].doy = c_int(172)  # Day of year
    inp[i].year = c_int(0)  # No effect
    inp[i].sec = c_double(29000.)
    inp[i].alt = c_double(400.)
    inp[i].g_lat = c_double(60.)
    inp[i].g_long = c_double(-70.)
    inp[i].lst = c_double(16.)
    inp[i].f107A = c_double(150.)
    inp[i].f107 = c_double(150.)
    inp[i].ap = c_double(4.)

inp[1].doy = c_int(81)
inp[2].sec = c_double(75000.)
inp[2].alt = c_double(1000.)
inp[3].alt = c_double(100.)
inp[10].alt = c_double(0.)
inp[11].alt = c_double(10.)
inp[12].alt = c_double(30.)
inp[13].alt = c_double(50.)
inp[14].alt = c_double(70.)
inp[16].alt = c_double(100.)
inp[4].g_lat = c_double(0.)
inp[5].g_long = c_double(0.)
inp[6].lst = c_double(4.)
inp[7].f107A = c_double(70.)
inp[8].f107 = c_double(180.)
inp[9].ap = c_double(40.)
inp[15].ap_a = pointer(aph)
inp[16].ap_a = pointer(aph)
for i in range(15):
    #     msis.gtd7(byref(inp[i]), byref(flags), byref(output[i]))
    msis.gtd7_py(inp[i].year, inp[i].doy, inp[i].sec, inp[i].alt, inp[i].g_lat,
                 inp[i].g_long, inp[i].lst, inp[i].f107A, inp[i].f107,
                 inp[i].ap, inp[15].ap_a, byref(flags), byref(output[i]))
flags.switches[9] = -1
for i in range(15, 17):
    msis.gtd7_py(inp[i].year, inp[i].doy, inp[i].sec, inp[i].alt, inp[i].g_lat,
                 inp[i].g_long, inp[i].lst, inp[i].f107A, inp[i].f107,
                 inp[i].ap, inp[15].ap_a, byref(flags), byref(output[i]))
#     msis.gtd7(byref(inp[i]), byref(flags), byref(output[i]))
# output type 1
outbuf = ""
for i in range(17):
    for j in range(9):
        outbuf += '{0:E} '.format(output[i].d[j])
    outbuf += '{0:E} '.format(output[i].t[0])
    outbuf += '{0:E} \n'.format(output[i].t[1])

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
print(outbuf)
