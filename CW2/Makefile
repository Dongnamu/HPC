EXES=stencil

stencil: stencil.c
	mpicc -xHost -Ofast -std=c99 -Wall -o $@ $^

all: $(EXES)

.PHONY: clean all

clean:
	\rm -f $(EXES)
	\rm -f *.o
