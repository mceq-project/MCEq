.PHONY: all clean realclean

PYMIC = ${HOME}/work/git/pyMIC

all: libmceq.so

libmceq.so: kernel_phi.c
	icc -I${PYMIC}/include -I$(MKLROOT)/include -fPIC -shared -mmic -g -O2 -L$(MKLROOT)/lib/mic -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -o $@ $<

clean:
	rm -f libmceq.so

realclean: clean
	rm -f *~
