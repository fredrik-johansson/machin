#!/usr/bin/env python3
"""Verify machin_formulas.py: every formula to 100 digits, P, and mu.
Usage: check_machin_formulas.py [machin_formulas.py [machin_tab.c]]
With machin_tab.c, also checks that the tables there have the same X, den, C."""
import sys, re, math
from mpmath import mp, mpf, atanh, atan, log, inf
from sympy import isprime, prime

mp.dps = 130
path = sys.argv[1] if len(sys.argv) > 1 else "machin_formulas.py"
env = {}
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

tabs = {}
if len(sys.argv) > 2:
    src = open(sys.argv[2]).read()
    for name, n, xs, den, cbits, cdata in re.findall(
            r'static const ulong ((?:log_atanh|atan)_(\d+))_x\[\] = \{(.*?)\};\s*'
            r'static const ulong \1_den = (\d+);\s*#define \1_cbits (\d+)\s*'
            r'static const ulong \1_c\[\] = \{(.*?)\};', src, re.S):
        n, cbits = int(n), int(cbits)
        X = [int(a) + (int(b) << 64) for a, b in re.findall(r'X\((\d+),\s*(\d+)\)', xs)]
        words = [int(w, 16) for grp in re.findall(r'Z[28]\(([^)]*)\)', cdata) for w in grp.split(',')]
        big = sum(w << (32 * i) for i, w in enumerate(words))
        C = []
        for k in range(n * n):
            v = (big >> (k * cbits)) & ((1 << cbits) - 1)
            C.append(v - (1 << cbits) if v >> (cbits - 1) else v)
        key = ('atanh' if name.startswith('log') else 'atan') + '_%d' % n
        tabs[key] = (X, int(den), [C[i * n:(i + 1) * n] for i in range(n)])

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
    if name in tabs:
        assert tabs[name] == (X, den, C), name + ' differs from machin_tab.c'
    count += 1
print("%d formulas verified%s" % (count, (", %d compared with machin_tab.c" % len(tabs)) if tabs else ""))
