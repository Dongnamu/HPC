CC=mpicc

COMP=GNU
ifeq ($(COMP), GNU)
  CFLAGS=-Wall
endif

EXES = test

all: $(EXES)

test: test.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	\rm $(EXES)
