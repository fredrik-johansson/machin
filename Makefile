CC ?= cc
CFLAGS ?= -O2 -Wall
# e.g. make FLINT=/usr/local  (headers in $(FLINT)/include/flint)
FLINT ?= /usr/local
CPPFLAGS += -I$(FLINT)/include -I$(FLINT)/include/flint
LDFLAGS += -L$(FLINT)/lib
LDLIBS = -lflint -lgmp -lm -lpthread

machin_set: machin_set.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

check: machin_set
	./machin_set 0 24 pell | tail -1
	./machin_set 1 22 pell | tail -1
	python3 scripts/check_machin_tab.py machin_tab.c

clean:
	rm -f machin_set
.PHONY: check clean
