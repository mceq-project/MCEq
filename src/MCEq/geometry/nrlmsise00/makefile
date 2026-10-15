CFLAGS = -Wall -O3 -g -fPIC
CC = gcc 
LIBTOOL = /opt/local/bin/glibtool

all: msis-00.so

msis-00.so :  nrlmsise-00.o nrlmsise-00_data.o
	$(CC) -shared -o msis-00.so nrlmsise-00.o nrlmsise-00_data.o 

test : nrlmsise-00.o nrlmsise-00_test.o nrlmsise-00_data.o
	$(CC) -o nrlmsise-test nrlmsise-00.o nrlmsise-00_test.o \
                     nrlmsise-00_data.o

.c.o:
	$(CC) -c $(CFLAGS) $<

clean   :
	rm -rf  nrlmsise-test *.o *.gch *.so
