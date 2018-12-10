CC=mpicc

COMP=GNU
ifeq ($(COMP), GNU)
  CFLAGS=-Wall
endif

EXES = test

test: test.c
	$(CC) $(CFLAGS) -o $@ $^

all: $(EXES)

.PHONY: clean all

clean:
	\rm $(EXES)
