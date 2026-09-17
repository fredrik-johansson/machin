# decode machin_tab.c-format tables and verify every formula numerically
import re, sys
from mpmath import mp, mpf, atanh, atan, log
from sympy import prime, isprime
mp.dps = 120
src = open(sys.argv[1]).read()
names = sys.argv[2:] if len(sys.argv) > 2 else None

def gauss_primes(n):
    gp = [(1, 1)]; p = 5
    while len(gp) < n:
        if isprime(p):
            a = 1
            while True:
                b2 = p - a*a; b = int(b2**0.5 + 0.5)
                if b*b == b2: gp.append((a, b)); break
                a += 1
        p += 4
    return gp

tabs = re.findall(r'static const ulong ((?:log_atanh|atan)_(\d+))_x\[\] = \{(.*?)\};\s*'
                  r'static const s?ulong \1_den = (\d+);\s*#define \1_cbits (\d+)\s*'
                  r'static const ulong \1_c\[\] = \{(.*?)\};', src, re.S)
for name, n, xs, den, cbits, cdata in tabs:
    if names and name not in names: continue
    n, den, cbits = int(n), int(den), int(cbits)
    X = [int(lo) + (int(hi) << 64) for lo, hi in re.findall(r'X\((\d+),\s*(\d+)\)', xs)]
    words = []
    for grp in re.findall(r'Z[28]\(([^)]*)\)', cdata):
        words += [int(w, 16) for w in grp.split(',')]
    assert len(words) % 2 == 0
    nbits = n*n*cbits
    exp_words = 2*((nbits + 63)//64)
    big = sum(w << (32*i) for i, w in enumerate(words))
    C = []
    for k in range(n*n):
        v = (big >> (k*cbits)) & ((1 << cbits) - 1)
        if v >> (cbits-1): v -= 1 << cbits
        C.append(v)
    maxbits = max(abs(v).bit_length() for v in C)
    g = name.startswith('atan')
    y = [(atan if g else atanh)(mpf(1)/x) for x in X]
    if g:
        tg = [atan(mpf(b)/a) for a, b in gauss_primes(n)]
    else:
        tg = [log(prime(i+1)) for i in range(n)]
    err = max(abs(sum(C[i*n+j]*y[j] for j in range(n))/den - tg[i]) for i in range(n))
    print("%-14s n=%2d len(x)=%2d words=%d (expected %d) cbits=%d (maxbits+1=%d) extra bits zero=%s err=%s" % (
        name, n, len(X), len(words), exp_words, cbits, maxbits+1, (big >> nbits) == 0, mp.nstr(err, 3)))
