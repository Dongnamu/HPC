CC=mpicc

COMP=GNU
ifeq ($(COMP), GNU)
  CFLAGS=-Wall
endif

test: test.c
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean all

clean:
	\rm -f $(EXES)
	\rm -f *.o
