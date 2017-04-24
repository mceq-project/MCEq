#! /usr/bin/env python

from ctypes import cdll, Structure, c_long, c_int, \
                   c_double, pointer, byref, \
                   POINTER
import copy

msis = cdll.LoadLibrary("msis-00.so")


class NrlmsiseFlags(Structure):
    _fields_ = [("switches", c_int * 24),
                ("sw", c_double * 24),
                ("swc", c_double * 24)]
class ap_array(Structure):
    _fields_ = [("a", c_double * 7)]
    
class nrlmsise_input(Structure):
    _field_ = [("year", c_int),
               ("doy", c_int),
               ("sec", c_double),
               ("alt", c_double),
               ("g_lat", c_double),
               ("g_long", c_double),
               ("lst", c_double),
               ("f107A", c_double),
               ("f107", c_double),
               ("ap", c_double),
               ("ap_a", POINTER(ap_array))]
    
class NrlmsiseOutput(Structure):
    _fields_ = [("d", c_double * 9),
                ("t", c_double * 2)]
        
output = [NrlmsiseOutput() for i in xrange(17)]
input = [nrlmsise_input() for i in xrange(17)]
flags = NrlmsiseFlags()
aph = ap_array()

# Input values
for i in xrange(7):
    aph.a[i] = c_double(100.)

flags.switches[0] = c_int(0)

for i in range(1, 24):
    flags.switches[i] = c_int(1)
    
for i in xrange(17):
    input[i].doy = c_int(172)  # Day of year
    input[i].year = c_int(0)  # No effect
    input[i].sec = c_double(29000.)
    input[i].alt = c_double(400.)
    input[i].g_lat = c_double(60.)
    input[i].g_long = c_double(-70.)
    input[i].lst = c_double(16.)
    input[i].f107A = c_double(150.)
    input[i].f107 = c_double(150.)
    input[i].ap = c_double(4.)
     
input[1].doy = c_int(81)
input[2].sec = c_double(75000.)
input[2].alt = c_double(1000.)
input[3].alt = c_double(100.)
input[10].alt = c_double(0.)
input[11].alt = c_double(10.)
input[12].alt = c_double(30.)
input[13].alt = c_double(50.)
input[14].alt = c_double(70.)
input[16].alt = c_double(100.)
input[4].g_lat = c_double(0.)
input[5].g_long = c_double(0.)
input[6].lst = c_double(4.)
input[7].f107A = c_double(70.)
input[8].f107 = c_double(180.)
input[9].ap = c_double(40.)
input[15].ap_a = pointer(aph)
input[16].ap_a = pointer(aph)
for i in xrange(15):
#     msis.gtd7(byref(input[i]), byref(flags), byref(output[i]))
    msis.gtd7_py(input[i].year, input[i].doy, input[i].sec,
                input[i].alt, input[i].g_lat,
                input[i].g_long, input[i].lst,
                input[i].f107A, input[i].f107,
                input[i].ap, input[15].ap_a, byref(flags), byref(output[i]))
flags.switches[9] = -1
for i in range(15, 17):
    msis.gtd7_py(input[i].year, input[i].doy, input[i].sec,
                input[i].alt, input[i].g_lat,
                input[i].g_long, input[i].lst,
                input[i].f107A, input[i].f107,
                input[i].ap, input[15].ap_a, byref(flags), byref(output[i]))
#     msis.gtd7(byref(input[i]), byref(flags), byref(output[i]))
# output type 1
outbuf = ""
for i in xrange(17):
    for j in xrange(9):
        outbuf += '{0:E} '.format(output[i].d[j])
    outbuf += '{0:E} '.format(output[i].t[0])
    outbuf += '{0:E} \n'.format(output[i].t[1])

# output type 2
for i in xrange(3):
    outbuf += "\n"
    outbuf += "\nDAY   "
    for j in xrange(5):
        outbuf += "         {0:3}".format(input[i * 5 + j].doy.value)
    outbuf += "\nUT    "
    for j in xrange(5):
        outbuf += "       {0:5.0f}".format(input[i * 5 + j].sec.value)
    outbuf += "\nALT   "
    for j in xrange(5):
        outbuf += "        {0:4.0f}".format(input[i * 5 + j].alt.value)
    outbuf += "\nLAT   "
    for j in xrange(5):
        outbuf += "         {0:3.0f}".format(input[i * 5 + j].g_lat.value)
    outbuf += "\nLONG  "
    for j in xrange(5):
        outbuf += "         {0:3.0f}".format(input[i * 5 + j].g_long.value)
    outbuf += "\nLST   "
    for j in xrange(5):
        outbuf += "       {0:5.0f}".format(input[i * 5 + j].lst.value)
    outbuf += "\nF107A "
    for j in xrange(5):
        outbuf += "         {0:3.0f}".format(input[i * 5 + j].f107A.value)
    outbuf += "\nF107  "
    for j in xrange(5):
        outbuf += "         {0:3.0f}".format(input[i * 5 + j].f107.value)
    outbuf += "\n\n"
    outbuf += "\nTINF  "
    for j in xrange(5):
        outbuf += "     {0:7.2f}".format(output[i * 5 + j].t[0])
    outbuf += "\nTG    "
    for j in xrange(5):
        outbuf += "     {0:7.2f}".format(output[i * 5 + j].t[1])
    outbuf += "\nHE    "
    for j in xrange(5):
        outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[0])
    outbuf += "\nO     "
    for j in xrange(5):
        outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[1])
    outbuf += "\nN2    "
    for j in xrange(5):
        outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[2])
    outbuf += "\nO2    "
    for j in xrange(5):
        outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[3])
    outbuf += "\nAR    "
    for j in xrange(5):
        outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[4])
    outbuf += "\nH     "
    for j in xrange(5):
        outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[6])
    outbuf += "\nN     "
    for j in xrange(5):
        outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[7])
    outbuf += "\nANM 0 "
    for j in xrange(5):
        outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[8])
    outbuf += "\nRHO   "
    for j in xrange(5):
        outbuf += "   {0:1.3e}".format(output[i * 5 + j].d[5])
    outbuf += "\n"
outbuf += "\n"
print outbuf