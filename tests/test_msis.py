from __future__ import print_function


def test_msis():
    import pathlib
    from ctypes import byref, c_double, c_int, pointer
    from MCEq.geometry.nrlmsise00.nrlmsise00 import (
        ap_array,
        msis,
        nrlmsise_flags,
        nrlmsise_input,
        nrlmsise_output,
    )

    exp_file = pathlib.Path(__file__).parent / "msis_expected.txt"
    if not exp_file.exists():
        raise FileNotFoundError(
            f"Expected output file {exp_file} not found. "
            "Please run the test with the expected output file."
        )
    with open(exp_file, "r") as f:
        result_expected = f.read()

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
            outbuf += f"{output[i].d[j]:E} "
        outbuf += f"{output[i].t[0]:E} "
        outbuf += f"{output[i].t[1]:E} \n"

    # output type 2
    for i in range(3):
        outbuf += "\n"
        outbuf += "\nDAY   "
        for j in range(5):
            outbuf += f"         {inp[i * 5 + j].doy.value:3}"
        outbuf += "\nUT    "
        for j in range(5):
            outbuf += f"       {inp[i * 5 + j].sec.value:5.0f}"
        outbuf += "\nALT   "
        for j in range(5):
            outbuf += f"        {inp[i * 5 + j].alt.value:4.0f}"
        outbuf += "\nLAT   "
        for j in range(5):
            outbuf += f"         {inp[i * 5 + j].g_lat.value:3.0f}"
        outbuf += "\nLONG  "
        for j in range(5):
            outbuf += f"         {inp[i * 5 + j].g_long.value:3.0f}"
        outbuf += "\nLST   "
        for j in range(5):
            outbuf += f"       {inp[i * 5 + j].lst.value:5.0f}"
        outbuf += "\nF107A "
        for j in range(5):
            outbuf += f"         {inp[i * 5 + j].f107A.value:3.0f}"
        outbuf += "\nF107  "
        for j in range(5):
            outbuf += f"         {inp[i * 5 + j].f107.value:3.0f}"
        outbuf += "\n\n"
        outbuf += "\nTINF  "
        for j in range(5):
            outbuf += f"     {output[i * 5 + j].t[0]:7.2f}"
        outbuf += "\nTG    "
        for j in range(5):
            outbuf += f"     {output[i * 5 + j].t[1]:7.2f}"
        outbuf += "\nHE    "
        for j in range(5):
            outbuf += f"   {output[i * 5 + j].d[0]:1.3e}"
        outbuf += "\nO     "
        for j in range(5):
            outbuf += f"   {output[i * 5 + j].d[1]:1.3e}"
        outbuf += "\nN2    "
        for j in range(5):
            outbuf += f"   {output[i * 5 + j].d[2]:1.3e}"
        outbuf += "\nO2    "
        for j in range(5):
            outbuf += f"   {output[i * 5 + j].d[3]:1.3e}"
        outbuf += "\nAR    "
        for j in range(5):
            outbuf += f"   {output[i * 5 + j].d[4]:1.3e}"
        outbuf += "\nH     "
        for j in range(5):
            outbuf += f"   {output[i * 5 + j].d[6]:1.3e}"
        outbuf += "\nN     "
        for j in range(5):
            outbuf += f"   {output[i * 5 + j].d[7]:1.3e}"
        outbuf += "\nANM 0 "
        for j in range(5):
            outbuf += f"   {output[i * 5 + j].d[8]:1.3e}"
        outbuf += "\nRHO   "
        for j in range(5):
            outbuf += f"   {output[i * 5 + j].d[5]:1.3e}"
        outbuf += "\n"
    outbuf += "\n"

    # print(outbuf)

    assert outbuf.strip() == result_expected.strip()
