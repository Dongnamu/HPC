CC=mpicc

COMP=GNU
ifeq ($(COMP), GNU)
  CFLAGS=-Wall
endif

test: test.c
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean test

clean:
	\rm -f test
	\rm -f *.o
