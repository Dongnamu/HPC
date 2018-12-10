CC=mpicc

COMP=GNU
ifeq ($(COMP), GNU)
  CFLAGS=-Wall
endif

EXES = test

test: test.c
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean $(EXES)

clean:
	\rm -f $(EXES)
	\rm -f *.o
