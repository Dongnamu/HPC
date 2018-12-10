CC=mpicc

CFLAGS=-Wall

EXES=test

test: test.c
	$(CC) $(CFLAGS) -o $@ $^

all: $(EXES)

.PHONY: clean all

clean:
	\rm -f $(EXES)
	\rm -f *.o
