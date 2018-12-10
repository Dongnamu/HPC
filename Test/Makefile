CC=mpicc

COMP=GNU
ifeq ($(COMP), GNU)
  CFLAGS=-Wall
endif

all: $(EXES)

test: test.c
	$(CC) $(CFLAGS) -o $@ $^
