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
	./machin_set 0 24 pell > check_24.py && tail -1 check_24.py
	./machin_set 1 22 pell > check_22.py && tail -1 check_22.py
	python3 -c "exec(open('check_24.py').read()); exec(open('check_22.py').read())"
	rm -f check_24.py check_22.py
	python3 scripts/check_machin_formulas.py machin_formulas.py

clean:
	rm -f machin_set
.PHONY: check clean
