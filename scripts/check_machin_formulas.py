#!/usr/bin/env python3
"""Verify formulas in machin_set's Python format (e.g. machin_formulas.py,
or the output of machin_set): every relation to 100 digits, P, X and mu.
Usage: check_machin_formulas.py [file ...]"""
import sys, re, math
from mpmath import mp, mpf, atanh, atan, log, inf
from sympy import isprime, prime

mp.dps = 130
env = {}
for path in (sys.argv[1:] or ["machin_formulas.py"]):
    exec(open(path).read(), env)

def gaussian_primes(n):
    out = [(1, 1)]
    p = 5
    while len(out) < n:
        if isprime(p):
            a = 1
            while 2 * a * a < p:
                b = math.isqrt(p - a * a)
                if b * b == p - a * a:
                    out.append((a, b)); break
                a += 1
        p += 4
    return out

count = 0
for name in sorted({k[:k.rindex('_')] for k in env if re.match(r'atanh?_\d+_[A-Za-z]+$', k)},
                   key=lambda s: (s.startswith('atan_'), int(s.split('_')[1]))):
    g = name.startswith('atan_')
    n = int(name.split('_')[1])
    P, X, den, C, mu = (env[name + s] for s in ('_P', '_X', '_den', '_C', '_mu'))
    assert len(P) == len(X) == len(C) == n and all(len(r) == n for r in C), name
    assert P == (gaussian_primes(n) if g else [prime(i + 1) for i in range(n)]), name
    assert X == sorted(X) and len(set(X)) == n, name
    f = atan if g else atanh
    y = [f(mpf(1) / x) for x in X]
    for i in range(n):
        target = atan(mpf(P[i][1]) / P[i][0]) if g else log(P[i])
        err = abs(sum(C[i][j] * y[j] for j in range(n)) / den - target)
        assert err < mpf(10) ** -100, (name, i, err)
    m = sum(1 / math.log10(x) for x in X) if min(X) > 1 else float('inf')
    assert (m == mu) if math.isinf(m) else abs(m - mu) < 6e-6, (name, m, mu)
    count += 1
print("%d formulas verified" % count)
