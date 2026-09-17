/*
    Copyright (C) 2026 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

/* Machin-type sets for the logarithms of the first NP primes or for
   the arguments of the first NP nonreal Gaussian primes pi_i = a_i + b_i i
   (1 + i, then the representatives with 0 < a < b, ordered by norm):

       log p_i = (1/den) sum_j C[i][j] atanh(1/x_j),        gaussian = 0
       atan(b_i/a_i) = (1/den) sum_j C[i][j] atan(1/x_j),   gaussian = 1

   Logarithms: integers x with x^2 - 1 smooth over the primes
   (Luca-Najman's smooth neighbours), found by a segmented sieve on
   the roots +-1 of x^2 = 1 modulo prime powers; each gives
   atanh(1/x) = (1/2) log((x+1)/(x-1)) = (1/2) sum_j e_j log p_j with
   e_j = v_j(x+1) - v_j(x-1).

   Gaussian primes: integers x with x^2 + 1 smooth over the norms
   N(pi_j), from a sieve on the Hensel-lifted roots of x^2 = -1 modulo
   prime powers; x + i = unit prod pi_j^(a_j) conj(pi_j)^(b_j) gives
   atan(1/x) = arg(x + i) = sum_j (a_j - b_j) arg(pi_j) + k pi/2, the
   (1+i) factor and the unit counted in units of arg(1+i) = pi/4 (the
   argument is only known modulo 2 pi from the factorization; the
   multiple is fixed numerically).

   The NP largest x whose rows are independent are kept (greedy by
   decreasing x, which also maximizes the smallest x of the set), the
   system is inverted with fmpq_mat, and the formula is printed as
   Python data (see fprint_python):

       atanh_<NP>_P = [2, 3, ...]    or   atan_<NP>_P = [(1, 1), (1, 2), ...]
       atanh_<NP>_X = [...]          the x_j, increasing
       atanh_<NP>_den = ...
       atanh_<NP>_C = [[...], ...]
       atanh_<NP>_mu = ...           sum_j 1/log10(x_j) (Lehmer's measure)

   preceded by progress lines and followed by a verification of every
   formula against arb_log_ui / arb_atan, both as Python comments, so
   that the whole output is valid Python. After the sieve, x is a
   two-limb value below 2^(2B-2) (B = FLINT_BITS).

   Three ways to find the x:

   1. Sieve (x <= XMAX < 2^(B-2)), multithreaded over its segments
      (flint_set_num_threads or FLINT_NUM_THREADS). The prime powers
      dividing a period P <= 2^20 are precomputed as a periodic pattern
      copied into each segment; the rest cost about
      XMAX sum_{p^k not | P} c/p^k cache-blocked updates (c = 1 for the
      logarithms, 2 for the arguments).

   2. ROUNDS rounds of combinations after the sieve (see chm_round)
      extend the set beyond XMAX, up to ZMAX (default and maximum
      2^(2B-2)): a pair x < y with (y - x) | x^2 -+ 1 yields
      z = x + (x^2 -+ 1)/(y - x). Not exhaustive above XMAX, but it
      reaches far larger x than sieving. With adaptive = 1, once NP
      independent x are known, later rounds only generate z above the
      smallest selected x (much faster; recommended).

   3. "pell" enumerates every smooth x <= ZMAX (default 2^(2B-2))
      exactly, by 2^NP Pell equations (see pell_enumerate); the result is
      the optimal set. About 1 microsecond per equation: NP = 24 in
      ~15 s, 30 in ~20 core-minutes, 32 in ~1.5 core-hours, 40 in ~370
      core-hours.

   With MACHIN_SET_EXTEND=K in the environment, each round also extends
   the K largest not yet extended elements x: every split x^2 -+ 1 = d e
   with d + e + 2x smooth gives the smooth x + d and x + e (see
   ext_round). Costly (tau(x^2 -+ 1)/2 tests per x, ~0.3 microseconds
   each; x with tau > MACHIN_SET_EXTEND_MAXTAU, default 2e7, are
   skipped) but it finds elements the rounds cannot reach. Only x >= zlo/2
   are extended, zlo being the smallest selected x, unless
   MACHIN_SET_EXTEND_FROM gives a lower bound (for the sparse Gaussian
   sets, small x can give large z = x + e as well).

   With MACHIN_SET_PY=file, the formula is also appended to file (to
   collect several formulas). MACHIN_SET_SOURCE=text replaces the
   description of how the x were found in the comment line before the
   formula (useful when regenerating a formula from a saved list).

   With MACHIN_SET_EXTEND_DONE=file, the x already extended are read
   from file and the file is updated after every extension, so that
   repeated or interrupted runs never extend an element twice (an
   element is only recorded once it has actually been extended, i.e.
   with tau within the limit then in force).

   Distributed or combined runs: with MACHIN_SET_SAVE=file in the
   environment, all x found are written to file (one per line);
   "pell ZMAX PART NPARTS" solves only the equations of jobs with
   index = PART mod NPARTS (4096 jobs in all); "load FILE" reads x (e.g. the
   concatenated parts, or Pell sets for fewer primes as seeds) and can
   be followed by rounds.

   Usage: machin_set gaussian NP XMAX [ROUNDS [ZMAX [adaptive]]]
          machin_set gaussian NP pell [ZMAX [PART NPARTS]]
          machin_set gaussian NP load FILE [ROUNDS [ZMAX [adaptive]]]
          e.g. machin_set 1 13 pell reproduces arb's 13-term set;
          machin_set 0 24 pell gives Luca-Najman's 24-term set
          (x from 134543112911873 to 19182937474703818751);
          machin_set 0 48 4e8 4 0 1 gives a 48-term set with all
          x > 6 * 10^19 (ZMAX = 0 means the maximum). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "flint.h"
#include "ulong_extras.h"
#include "fmpz.h"
#include "fmpq.h"
#include "fmpq_mat.h"
#include "fmpz_mat.h"
#include "mpn_extras.h"
#include "nmod.h"
#include "arb.h"
#include "thread_support.h"
#if FLINT_USES_PTHREAD
#include <pthread.h>
#endif

#define MAXP 128

/* the first n nonreal Gaussian primes in order of norm, as (a, b)
   with a + b i, a <= b: 1 + i (norm 2), then for each prime
   p = 1 mod 4 the pair a^2 + b^2 = p, which is unique with a < b
   (the convention of arb's atan tables and of fixed/atan_gauss.c) */
static void
gaussian_primes(slong * a, slong * b, slong n)
{
    slong k = 0;
    ulong p;

    if (n > 0)
    {
        a[0] = 1; b[0] = 1; k = 1;
    }
    for (p = 5; k < n; p += 4)
    {
        ulong u;
        if (!n_is_prime(p))
            continue;
        for (u = 1; 2 * u * u < p; u++)
        {
            ulong v = n_sqrt(p - u * u);
            if (v * v == p - u * u)
            {
                a[k] = (slong) u; b[k] = (slong) v; k++;
                break;
            }
        }
    }
}

typedef struct { ulong mod; ulong r; int lg16; } sieve_entry;

static int
cmp_entry(const void * x, const void * y)
{
    ulong a = ((const sieve_entry *) x)->mod, b = ((const sieve_entry *) y)->mod;
    return (a > b) - (a < b);
}

static int
cmp_ulong(const void * x, const void * y)
{
    ulong a = *(const ulong *) x, b = *(const ulong *) y;
    return (a > b) - (a < b);
}

/* 2-adic inverse of odd p */
static ulong
binvert(ulong p)
{
    ulong v = p;
    int i;
    for (i = 0; i < 6; i++)
        v *= 2 - p * v;
    return v;
}

/* exact smoothness tests on single / double limbs: for odd p with
   pinv = p^-1 mod 2^B and plim = floor((2^B - 1) / p), p | n iff
   n * pinv <= plim, and then n / p = n * pinv */
static int
smooth_ui(ulong n, slong NP, const ulong * pinv, const ulong * plim)
{
    slong j;
    n >>= flint_ctz(n);                 /* norms[0] = 2 */
    for (j = 1; j < NP && n != 1; j++)
        while (n * pinv[j] <= plim[j])
            n *= pinv[j];
    return n == 1;
}

/* is x^2 + 1 smooth over the Gaussian norms? (norms[0] = 2, the rest odd) */
static int
smooth_x2p1(ulong x, slong NP, const ulong * norms, const ulong * pinv,
    const ulong * plim)
{
    ulong hi, lo;
    slong j;
    umul_ppmm(hi, lo, x, x);
    add_ssaaaa(hi, lo, hi, lo, 0, 1);
    if ((lo & 1) == 0)                  /* x odd: 2 || x^2 + 1 */
    {
        lo = (lo >> 1) | (hi << (FLINT_BITS - 1));
        hi >>= 1;
    }
    for (j = 1; j < NP; j++)
    {
        ulong p = norms[j], inv = pinv[j], lim = plim[j];
        while (hi != 0)
        {
            /* (hi, lo) / p on two limbs: the low quotient limb is
               lo / p mod 2^B; divisible iff hi - high(ql * p) >= 0
               is divisible by p */
            ulong ql = lo * inv, h, l, t;
            umul_ppmm(h, l, ql, p);
            (void) l;
            if (hi < h)
                break;
            t = (hi - h) * inv;
            if (t > lim)
                break;
            hi = t; lo = ql;
        }
        if (hi != 0)
            continue;
        if (lo == 1)
            return 1;
        while (lo * inv <= lim)
            lo *= inv;
        if (lo == 1)
            return 1;
    }
    return hi == 0 && lo == 1;
}

/* The sieve runs over z: z = x - 1 for the logarithms (a log sieve of the
   integers themselves, x being kept when z = x - 1 and z + 2 = x + 1 are
   both smooth), z = x for the arguments (roots of x^2 = -1).
   lg16 = ceil(16 log2 p), so the accumulated value is never below the
   true 16 log2 of the smooth part: no smooth z is missed, and the
   rounding excess (< 1 per hit) is far below 16 log2 of any prime
   outside the set, so there are (essentially) no false candidates. */

#define SEG (1 << 22)       /* unit of work for the threads */
#define BLOCK (1 << 14)     /* cache block for the small moduli */
#define CHUNK 256           /* scan chunk: one threshold per chunk */
#define PATLIM (1 << 20)    /* max period of the precomputed pattern */

typedef struct
{
    const sieve_entry * ent;    /* scattered entries, sorted by modulus */
    slong nent, nsmall;         /* ent[0..nsmall) have mod < BLOCK */
    const unsigned short * pat; /* sum over the prime powers dividing P */
    ulong P;
    int gaussian;
    slong NP;
    const ulong * norms, * pinv, * plim;
    ulong zlo, zhi;             /* z range */
    ulong ** xs;
    slong * nx, * xalloc;
    slong nseg, done;
#if FLINT_USES_PTHREAD
    pthread_mutex_t mutex;
#endif
}
sieve_work_t;

static void
sieve_segment(slong segno, sieve_work_t * w)
{
    ulong z0 = w->zlo + (ulong) segno * SEG;
    ulong len, L, b, t, c, i;
    unsigned short * seg;
    ulong * pos;
    ulong * found = NULL;
    slong nfound = 0, falloc = 0;
    int gaussian = w->gaussian;

    if (z0 > w->zhi)
        return;
    len = FLINT_MIN((ulong) SEG, w->zhi - z0 + 1);
    L = len + (gaussian ? 0 : 2);       /* z + 2 is needed too */
    seg = flint_malloc(L * sizeof(unsigned short));
    pos = flint_malloc((w->nent + 1) * sizeof(ulong));

    /* small prime powers: copy the periodic pattern */
    {
        ulong o = z0 % w->P, done = 0;
        while (done < L)
        {
            ulong n = FLINT_MIN(w->P - o, L - done);
            memcpy(seg + done, w->pat + o, n * sizeof(unsigned short));
            done += n;
            o = 0;
        }
    }

    /* first hit of each remaining prime power in the segment */
    for (i = 0; i < (ulong) w->nent; i++)
    {
        ulong m = w->ent[i].mod, r = w->ent[i].r;
        pos[i] = (r >= z0) ? r - z0 : (m - (z0 - r) % m) % m;
    }

    /* moduli < BLOCK: cache-blocked */
    for (b = 0; b < L; b += BLOCK)
    {
        ulong bend = FLINT_MIN(b + BLOCK, L);
        for (i = 0; i < (ulong) w->nsmall; i++)
        {
            ulong q = pos[i], m = w->ent[i].mod;
            unsigned short lg = (unsigned short) w->ent[i].lg16;
            for ( ; q < bend; q += m)
                seg[q] += lg;
            pos[i] = q;
        }
    }
    /* moduli >= BLOCK: few hits each */
    for (i = w->nsmall; i < (ulong) w->nent; i++)
    {
        ulong q = pos[i], m = w->ent[i].mod;
        unsigned short lg = (unsigned short) w->ent[i].lg16;
        for ( ; q < L; q += m)
            seg[q] += lg;
    }

    /* scan: one (conservative) threshold per chunk, taken at its start;
       the chunk maximum is computed branch-free (vectorizable) */
    for (c = 0; c < len; c += CHUNK)
    {
        ulong cend = FLINT_MIN(c + CHUNK, len);
        double zc = (double) (z0 + c), lgv;
        unsigned mx = 0, thr;

        lgv = gaussian ? 16.0 * log2(zc * zc + 1.0) : 16.0 * log2(zc);
        lgv = floor(lgv - 1e-6);
        thr = (lgv > 0) ? (unsigned) lgv : 0;

        if (cend - c == CHUNK)      /* constant trip count: vectorizes at -O2 */
        {
            const unsigned short * u = seg + c;
            unsigned short m = 0;
            int k;
            if (gaussian)
                for (k = 0; k < CHUNK; k++)
                    m = FLINT_MAX(m, u[k]);
            else
                for (k = 0; k < CHUNK; k++)
                    m = FLINT_MAX(m, FLINT_MIN(u[k], u[k + 2]));
            mx = m;
        }
        else
        {
            for (t = c; t < cend; t++)
                mx = FLINT_MAX(mx, gaussian ? (unsigned) seg[t]
                    : (unsigned) FLINT_MIN(seg[t], seg[t + 2]));
        }
        if (mx < thr)
            continue;

        for (t = c; t < cend; t++)
        {
            ulong x;
            int ok;
            if (gaussian)
            {
                if (seg[t] < thr)
                    continue;
                x = z0 + t;
                ok = smooth_x2p1(x, w->NP, w->norms, w->pinv, w->plim);
            }
            else
            {
                if (seg[t] < thr || seg[t + 2] < thr)
                    continue;
                x = z0 + t + 1;
                ok = smooth_ui(x - 1, w->NP, w->pinv, w->plim)
                  && smooth_ui(x + 1, w->NP, w->pinv, w->plim);
            }
            if (ok)
            {
                if (nfound == falloc)
                {
                    falloc = 2 * falloc + 16;
                    found = flint_realloc(found, falloc * sizeof(ulong));
                }
                found[nfound++] = x;
            }
        }
    }
    flint_free(seg);
    flint_free(pos);

#if FLINT_USES_PTHREAD
    pthread_mutex_lock(&w->mutex);
#endif
    if (nfound > 0)
    {
        if (*w->nx + nfound > *w->xalloc)
        {
            *w->xalloc = 2 * (*w->nx + nfound) + 64;
            *w->xs = flint_realloc(*w->xs, *w->xalloc * sizeof(ulong));
        }
        memcpy(*w->xs + *w->nx, found, nfound * sizeof(ulong));
        *w->nx += nfound;
    }
    w->done++;
    if (w->done % 256 == 0 || w->done == w->nseg)
        fprintf(stderr, "  sieved %ld / %ld segments, %ld smooth so far\n",
            w->done, w->nseg, *w->nx);
#if FLINT_USES_PTHREAD
    pthread_mutex_unlock(&w->mutex);
#endif
    flint_free(found);
}

/* split the entries: the prime powers dividing P (chosen greedily by
   updates saved per bit of period) go into a periodic pattern */
static unsigned short *
build_pattern(ulong * Pout, sieve_entry * ent, slong * nent,
    const ulong * norms, slong NP)
{
    slong e[MAXP], j, i, k;
    ulong P = 1;
    unsigned short * pat;

    for (j = 0; j < NP; j++)
        e[j] = 0;
    while (1)
    {
        slong best = -1;
        double bestv = 0;
        for (j = 0; j < NP; j++)
        {
            ulong p = norms[j], q = 1;
            slong cnt = 0;
            if (P > PATLIM / p)
                continue;
            for (k = 0; k <= e[j]; k++)
                q *= p;
            for (i = 0; i < *nent; i++)
                cnt += (ent[i].mod == q);
            if (cnt > 0 && cnt / (double) q / log((double) p) > bestv)
            {
                bestv = cnt / (double) q / log((double) p);
                best = j;
            }
        }
        if (best < 0)
            break;
        e[best]++;
        P *= norms[best];
    }

    pat = flint_calloc(P, sizeof(unsigned short));
    for (i = k = 0; i < *nent; i++)
    {
        if (P % ent[i].mod == 0)
        {
            ulong t;
            for (t = ent[i].r % ent[i].mod; t < P; t += ent[i].mod)
                pat[t] += (unsigned short) ent[i].lg16;
        }
        else
            ent[k++] = ent[i];
    }
    *nent = k;
    *Pout = P;
    return pat;
}

/* ------------------------------------------------------------------ */
/* Two-limb x. Everything after the sieve works with x < 2^(2B-2)
   (B = FLINT_BITS), so that x^2 -+ 1 fits in four limbs. */

typedef struct { ulong lo, hi; } xval_t;

#define XV_MAXBITS (2 * FLINT_BITS - 2)

/* (hi, lo) = q d + r with hi < d */
static ulong
divrem_ll(ulong * r, ulong hi, ulong lo, ulong d)
{
    int s = flint_clz(d);
    ulong q, rr;
    if (s != 0)
    {
        hi = (hi << s) | (lo >> (FLINT_BITS - s));
        lo <<= s;
        d <<= s;
    }
    udiv_qrnnd(q, rr, hi, lo, d);
    *r = rr >> s;
    return q;
}

static xval_t xv_ui(ulong a) { xval_t r; r.lo = a; r.hi = 0; return r; }

static int
xv_cmp(xval_t a, xval_t b)
{
    if (a.hi != b.hi) return (a.hi < b.hi) ? -1 : 1;
    if (a.lo != b.lo) return (a.lo < b.lo) ? -1 : 1;
    return 0;
}

static int
cmp_xval(const void * a, const void * b)
{
    return xv_cmp(*(const xval_t *) a, *(const xval_t *) b);
}

static xval_t
xv_add(xval_t a, xval_t b)
{
    xval_t r;
    add_ssaaaa(r.hi, r.lo, a.hi, a.lo, b.hi, b.lo);
    return r;
}

static xval_t
xv_sub(xval_t a, xval_t b)
{
    xval_t r;
    sub_ddmmss(r.hi, r.lo, a.hi, a.lo, b.hi, b.lo);
    return r;
}

static double
xv_d(xval_t a)
{
    return ldexp((double) a.hi, FLINT_BITS) + (double) a.lo;
}

static void
xv_print(xval_t a)
{
    fmpz_t t;
    fmpz_init(t);
    fmpz_set_uiui(t, a.hi, a.lo);
    fmpz_print(t);
    fmpz_clear(t);
}

/* a * b if it is <= bound (both < 2^(2B)), else 0 */
static int
xv_mul_bounded(xval_t * r, xval_t a, xval_t b, xval_t bound)
{
    ulong p[4], u[2], v[2];
    u[0] = a.lo; u[1] = a.hi; v[0] = b.lo; v[1] = b.hi;
    if ((a.hi != 0 && b.hi != 0))
        return 0;
    mpn_mul_n(p, u, v, 2);
    if (p[2] != 0 || p[3] != 0)
        return 0;
    r->lo = p[0]; r->hi = p[1];
    return xv_cmp(*r, bound) <= 0;
}

/* a * p if it is <= bound, else 0 */
static int
xv_mul_ui_bounded(xval_t * r, xval_t a, ulong p, xval_t bound)
{
    ulong h1, l1, h2, l2;
    umul_ppmm(h1, l1, a.lo, p);
    umul_ppmm(h2, l2, a.hi, p);
    if (h2 != 0)
        return 0;
    r->lo = l1;
    r->hi = l2 + h1;
    if (r->hi < h1)
        return 0;
    return xv_cmp(*r, bound) <= 0;
}

static slong
mpn_normsize(const ulong * a, slong n)
{
    while (n > 0 && a[n - 1] == 0)
        n--;
    return n;
}

/* {a, n} = x^2 -+ 1 (n <= 4) */
static slong
xv_N(ulong * a, xval_t x, int gaussian)
{
    ulong u[2];
    u[0] = x.lo; u[1] = x.hi;
    mpn_sqr(a, u, 2);
    if (gaussian)
        mpn_add_1(a, a, 4, 1);
    else
        mpn_sub_1(a, a, 4, 1);
    return mpn_normsize(a, 4);
}

/* if the odd p divides {a, n}, set {q, n} = {a, n} / p and return 1
   (q and a must not overlap; q is garbage otherwise) */
static int
mpn_divexact_odd(ulong * q, const ulong * a, slong n, ulong p, ulong inv)
{
    ulong c = 0, s, t, h, l;
    slong i;
    for (i = 0; i < n; i++)
    {
        s = a[i];
        t = s - c;
        c = (s < c);
        q[i] = t * inv;
        umul_ppmm(h, l, q[i], p);
        (void) l;
        c += h;
    }
    return c == 0;
}

/* {t, n} (destroyed) smooth over the norms? if e != NULL, the exponents
   are added to e[] */
static int
smooth_mpn(int * e, ulong * t, slong n, slong NP, const ulong * norms,
    const ulong * pinv, const ulong * plim)
{
    ulong q[8];
    slong j;
    int c;

    n = mpn_normsize(t, n);
    if (n == 0)
        return 0;
    /* p = 2 */
    c = 0;
    while (t[0] == 0)
    {
        slong i;
        for (i = 0; i + 1 < n; i++)
            t[i] = t[i + 1];
        n--;
        c += FLINT_BITS;
    }
    if (flint_ctz(t[0]) != 0)
    {
        c += flint_ctz(t[0]);
        mpn_rshift(t, t, n, flint_ctz(t[0]));
    }
    n = mpn_normsize(t, n);
    if (e != NULL)
        e[0] += c;

    for (j = 1; j < NP; j++)
    {
        ulong p = norms[j], inv = pinv[j], lim = plim[j];
        if (n == 1 && t[0] == 1)
            return 1;
        while (n > 1)
        {
            if (!mpn_divexact_odd(q, t, n, p, inv))
                break;
            n = mpn_normsize(q, n);
            flint_mpn_copyi(t, q, n);
            if (e != NULL)
                e[j]++;
        }
        if (n == 1)
            while (t[0] * inv <= lim)
            {
                t[0] *= inv;
                if (e != NULL)
                    e[j]++;
            }
    }
    return n == 1 && t[0] == 1;
}

/* the factorization of N(x) = x^2 -+ 1: exponents e[], N in {nn, *nsz} */
static int
factor_N(int * e, ulong * nn, slong * nsz, xval_t x, int gaussian,
    slong NP, const ulong * norms, const ulong * pinv, const ulong * plim)
{
    ulong t[4];
    slong j;
    for (j = 0; j < NP; j++)
        e[j] = 0;
    *nsz = xv_N(nn, x, gaussian);
    flint_mpn_copyi(t, nn, *nsz);
    return smooth_mpn(e, t, *nsz, NP, norms, pinv, plim);
}

static int
smooth_xv(xval_t x, int gaussian, slong NP, const ulong * norms,
    const ulong * pinv, const ulong * plim)
{
    if (!gaussian)
    {
        ulong t[2];
        xval_t y = xv_sub(x, xv_ui(1));
        t[0] = y.lo; t[1] = y.hi;
        if (!smooth_mpn(NULL, t, 2, NP, norms, pinv, plim))
            return 0;
        y = xv_add(x, xv_ui(1));
        t[0] = y.lo; t[1] = y.hi;
        return smooth_mpn(NULL, t, 2, NP, norms, pinv, plim);
    }
    else
    {
        ulong t[4];
        slong n = xv_N(t, x, 1);
        return smooth_mpn(NULL, t, n, NP, norms, pinv, plim);
    }
}

/* the relation row c[0..NP) of x: e_j = v_j(x+1) - v_j(x-1) for the
   logarithms, the Gaussian exponents (with the quarter turns in c[0])
   for the arguments; returns 0 on a failed factorization */
static int
factor_row(slong * c, xval_t x, int gaussian, slong NP, const ulong * norms,
    const ulong * pinv, const ulong * plim, const slong * a_, const slong * b_)
{
    slong j;
    if (!gaussian)
    {
        int ep[MAXP], em[MAXP];
        ulong t[2];
        xval_t y;
        for (j = 0; j < NP; j++)
            ep[j] = em[j] = 0;
        y = xv_add(x, xv_ui(1));
        t[0] = y.lo; t[1] = y.hi;
        if (!smooth_mpn(ep, t, 2, NP, norms, pinv, plim))
            return 0;
        y = xv_sub(x, xv_ui(1));
        t[0] = y.lo; t[1] = y.hi;
        if (!smooth_mpn(em, t, 2, NP, norms, pinv, plim))
            return 0;
        for (j = 0; j < NP; j++)
            c[j] = ep[j] - em[j];
        return 1;
    }
    else
    {
        fmpz_t u, v, t1, t2;
        fmpz_init(u); fmpz_init(v); fmpz_init(t1); fmpz_init(t2);
        fmpz_set_uiui(u, x.hi, x.lo); fmpz_one(v);     /* u + v i = x + i */
        for (j = 0; j < NP; j++) c[j] = 0;
        /* (1 + i): (u + v i) / (1 + i) = ((u + v) + (v - u) i) / 2 */
        while (1)
        {
            fmpz_add(t1, u, v);
            if (!fmpz_is_even(t1)) break;
            fmpz_sub(t2, v, u);
            fmpz_fdiv_q_2exp(u, t1, 1);
            fmpz_fdiv_q_2exp(v, t2, 1);
            c[0]++;
        }
        for (j = 1; j < NP; j++)
        {
            ulong p = norms[j];
            slong a = a_[j], b = b_[j];
            /* divide by pi = a + b i: (u + v i)(a - b i) / p */
            while (1)
            {
                fmpz_mul_si(t1, u, a); fmpz_addmul_si(t1, v, b);
                fmpz_mul_si(t2, v, a); fmpz_submul_si(t2, u, b);
                if (fmpz_fdiv_ui(t1, p) != 0 || fmpz_fdiv_ui(t2, p) != 0) break;
                fmpz_divexact_ui(u, t1, p); fmpz_divexact_ui(v, t2, p);
                c[j]++;
            }
            /* divide by conj(pi) = a - b i: (u + v i)(a + b i) / p */
            while (1)
            {
                fmpz_mul_si(t1, u, a); fmpz_submul_si(t1, v, b);
                fmpz_mul_si(t2, v, a); fmpz_addmul_si(t2, u, b);
                if (fmpz_fdiv_ui(t1, p) != 0 || fmpz_fdiv_ui(t2, p) != 0) break;
                fmpz_divexact_ui(u, t1, p); fmpz_divexact_ui(v, t2, p);
                c[j]--;
            }
        }
        /* the remaining unit: 1, i, -1, -i -> 0, 2, 4, 6 quarter-pis */
        if (fmpz_is_one(u) && fmpz_is_zero(v)) ;
        else if (fmpz_is_zero(u) && fmpz_is_one(v)) c[0] += 2;
        else if (fmpz_equal_si(u, -1) && fmpz_is_zero(v)) c[0] += 4;
        else if (fmpz_is_zero(u) && fmpz_equal_si(v, -1)) c[0] += 6;
        else
        {
            fmpz_clear(u); fmpz_clear(v); fmpz_clear(t1); fmpz_clear(t2);
            return 0;
        }
        /* the argument is only known modulo 2 pi = 8 quarter turns:
           fix c[0] numerically (the angles are far from the ambiguity) */
        {
            double s = c[0] * (M_PI / 4), target = atan(1.0 / xv_d(x));
            slong kk;
            for (j = 1; j < NP; j++)
                s += c[j] * atan2((double) b_[j], (double) a_[j]);
            kk = (slong) floor((target - s) / (2 * M_PI) + 0.5);
            c[0] += 8 * kk;
        }
        fmpz_clear(u); fmpz_clear(v); fmpz_clear(t1); fmpz_clear(t2);
        return 1;
    }
}

/* ------------------------------------------------------------------ */
/* Combination rounds (Conrey-Holmstrom-McLaughlin / Stormer style).

   For smooth x < y with d = y - x dividing N(x) = x^2 - 1 (logarithms)
   or x^2 + 1 (arguments), the subtraction formula
       atanh(1/x) - atanh(1/y) = atanh(1/z),  z = (xy - 1)/(y - x),
       atan(1/x)  - atan(1/y)  = atan(1/z),   z = (xy + 1)/(y - x),
   gives the integer z = x + N(x)/d, and z is again smooth
   (N(y) = d (d + 2x + N(x)/d) and N(z) = e (e + 2x + d) with e = N(x)/d).
   Since xy -+ 1 = x^2 -+ 1 mod (y - x), the only test is d | N(x).

   Each round, for every x of the set, finds the y = x + d of the set with
   z in (zlo, zhi], either by enumerating the divisors d of N(x) in the
   admissible window (hash lookup of x + d) or, when the window holds
   fewer set elements than N(x) has divisors there, by testing
   N(x) mod (y - x) for each of them; whichever is cheaper is chosen on
   the fly. After the first round only pairs involving an element added
   in the previous round are examined. */

/* q = floor({a, an} / d), r = {a, an} mod d; returns 0 if q >= 2^(2B) */
static int
divrem_nx(xval_t * q, xval_t * r, const ulong * a, slong an, xval_t d)
{
    ulong qq[5], rr[2], dd[2];
    slong dn = (d.hi != 0) ? 2 : 1, i;
    if (an < dn)
    {
        q->lo = q->hi = 0;
        r->lo = (an > 0) ? a[0] : 0;
        r->hi = (an > 1) ? a[1] : 0;
        return 1;
    }
    dd[0] = d.lo; dd[1] = d.hi;
    mpn_tdiv_qr(qq, rr, 0, a, an, dd, dn);
    for (i = 2; i < an - dn + 1; i++)
        if (qq[i] != 0)
            return 0;
    q->lo = qq[0];
    q->hi = (an - dn + 1 > 1) ? qq[1] : 0;
    r->lo = rr[0];
    r->hi = (dn > 1) ? rr[1] : 0;
    return 1;
}

/* {a, an} mod d == 0 ? */
static int
divisible_nx(const ulong * a, slong an, xval_t d)
{
    xval_t q, r;
    if (d.hi == 0)
        return mpn_mod_1(a, an, d.lo) == 0;
    divrem_nx(&q, &r, a, an, d);
    return r.lo == 0 && r.hi == 0;
}

/* an open-addressing hash set of nonzero two-limb values */
typedef struct { xval_t * tab; ulong mask; int shift; } uset_t;

static ulong
xv_hash(xval_t v)
{
    return (v.lo ^ (v.hi * UWORD(0xD6E8FEB86659FD93))) * UWORD(0x9E3779B97F4A7C15);
}

static void
uset_init(uset_t * H, const xval_t * v, slong n)
{
    slong i;
    int k = 4;
    while ((WORD(1) << k) < 2 * n + 16)
        k++;
    H->mask = (UWORD(1) << k) - 1;
    H->shift = FLINT_BITS - k;
    H->tab = flint_calloc(H->mask + 1, sizeof(xval_t));
    for (i = 0; i < n; i++)
    {
        ulong h = xv_hash(v[i]) >> H->shift;
        while ((H->tab[h].lo | H->tab[h].hi) != 0 && xv_cmp(H->tab[h], v[i]) != 0)
            h = (h + 1) & H->mask;
        H->tab[h] = v[i];
    }
}

static int
uset_has(const uset_t * H, xval_t v)
{
    ulong h = xv_hash(v) >> H->shift;
    while ((H->tab[h].lo | H->tab[h].hi) != 0)
    {
        if (xv_cmp(H->tab[h], v) == 0)
            return 1;
        h = (h + 1) & H->mask;
    }
    return 0;
}

/* first index i with v[i] >= a */
static slong
lower_bound(const xval_t * v, slong n, xval_t a)
{
    slong lo = 0, hi = n;
    while (lo < hi)
    {
        slong mid = lo + (hi - lo) / 2;
        if (xv_cmp(v[mid], a) < 0) lo = mid + 1; else hi = mid;
    }
    return lo;
}

typedef struct
{
    const ulong * q;    /* primes dividing N, largest first */
    const int * ex;
    slong m;
    xval_t bound;       /* only divisors <= bound */
    xval_t * list;
    slong n, cap;
}
divgen_t;

/* all divisors <= bound; returns 0 if there are more than cap */
static int
divgen(divgen_t * G, xval_t d, slong k)
{
    int a;
    if (k == G->m)
    {
        if (G->n == G->cap)
            return 0;
        G->list[G->n++] = d;
        return 1;
    }
    for (a = 0; ; a++)
    {
        if (!divgen(G, d, k + 1))
            return 0;
        if (a == G->ex[k] || !xv_mul_ui_bounded(&d, d, G->q[k], G->bound))
            return 1;
    }
}

typedef struct
{
    const xval_t * S;           /* all elements, sorted */
    slong nS;
    const xval_t * T;           /* elements added in the last round, sorted */
    slong nT;
    uset_t HS, HT;
    const char * isnew;         /* isnew[i]: S[i] is in T */
    int first;                  /* first round: T = S */
    int gaussian;
    slong NP;
    const ulong * norms, * pinv, * plim;
    xval_t zlo, zhi;
    slong nblocks, bsize;
    xval_t ** out;
    slong * nout, * oalloc;
    slong pairs_tested, divs_tested, done;
#if FLINT_USES_PTHREAD
    pthread_mutex_t mutex;
#endif
}
chm_work_t;

#define PAIR_CUTOFF 16

static void
push_xv(xval_t ** v, slong * n, slong * alloc, xval_t x)
{
    if (*n == *alloc)
    {
        *alloc = 2 * *alloc + 64;
        *v = flint_realloc(*v, *alloc * sizeof(xval_t));
    }
    (*v)[(*n)++] = x;
}

static void
chm_block(slong blk, chm_work_t * w)
{
    slong i0 = blk * w->bsize, i1 = FLINT_MIN(i0 + w->bsize, w->nS), i;
    xval_t * found = NULL, * dl = NULL;
    slong nfound = 0, falloc = 0, dcap = 0;
    slong npairs = 0, ndivs = 0;
    xval_t Smax = w->S[w->nS - 1], one = xv_ui(1);
    int e[MAXP], ex[MAXP];
    ulong q[MAXP], nn[4];

    for (i = i0; i < i1; i++)
    {
        xval_t x = w->S[i], dlo, dhi, r, t, qq;
        const xval_t * V;
        const uset_t * H;
        slong nV, a, b, j, m, k, nsz;

        if (xv_cmp(x, w->zhi) >= 0 || xv_cmp(x, Smax) >= 0)
            continue;
        /* pair x with every later element if x is new, else only with
           the new ones */
        if (w->first || w->isnew[i])
            { V = w->S; nV = w->nS; H = &w->HS; }
        else
            { V = w->T; nV = w->nT; H = &w->HT; }
        if (nV == 0)
            continue;

        if (!factor_N(e, nn, &nsz, x, w->gaussian, w->NP, w->norms,
                w->pinv, w->plim))
        {
            printf("internal error: ");
            xv_print(x);
            printf(" is not smooth\n");
            flint_abort();
        }
        /* z = x + N/d <= zhi  <=>  d >= ceil(N / (zhi - x)) */
        if (!divrem_nx(&dlo, &r, nn, nsz, xv_sub(w->zhi, x)))
            continue;
        if (r.lo != 0 || r.hi != 0)
            dlo = xv_add(dlo, one);
        if (dlo.lo == 0 && dlo.hi == 0)
            dlo = one;
        /* z > zlo  <=>  N/d >= zlo - x + 1  <=>  d <= N / (zlo - x + 1) */
        dhi = xv_sub(Smax, x);
        if (xv_cmp(x, w->zlo) < 0
            && divrem_nx(&t, &r, nn, nsz, xv_add(xv_sub(w->zlo, x), one))
            && xv_cmp(t, dhi) < 0)
            dhi = t;
        if (xv_cmp(dlo, dhi) > 0)
            continue;

        a = lower_bound(V, nV, xv_add(x, dlo));
        b = lower_bound(V, nV, xv_add(xv_add(x, dhi), one));
        if (a >= b)
            continue;

        /* the primes of N, largest first */
        for (j = w->NP - 1, m = 0; j >= 0; j--)
            if (e[j] != 0)
            {
                q[m] = w->norms[j];
                ex[m] = e[j];
                m++;
            }

        k = -1;
        if (b - a > PAIR_CUTOFF)
        {
            divgen_t G;
            if (dcap < b - a)
            {
                dcap = b - a;
                dl = flint_realloc(dl, dcap * sizeof(xval_t));
            }
            G.q = q; G.ex = ex; G.m = m; G.bound = dhi;
            G.list = dl; G.n = 0; G.cap = b - a;
            if (divgen(&G, one, 0))
                k = G.n;
            ndivs += G.n;
        }

        if (k >= 0)
        {
            /* few divisors in the window: look up x + d */
            for (j = 0; j < k; j++)
            {
                xval_t d = dl[j];
                if (xv_cmp(d, dlo) < 0 || !uset_has(H, xv_add(x, d)))
                    continue;
                divrem_nx(&qq, &r, nn, nsz, d);
                push_xv(&found, &nfound, &falloc, xv_add(x, qq));
            }
        }
        else
        {
            /* few elements in the window: test them */
            for (j = a; j < b; j++)
            {
                xval_t d = xv_sub(V[j], x);
                if (!divisible_nx(nn, nsz, d))
                    continue;
                divrem_nx(&qq, &r, nn, nsz, d);
                push_xv(&found, &nfound, &falloc, xv_add(x, qq));
            }
            npairs += b - a;
        }
    }

#if FLINT_USES_PTHREAD
    pthread_mutex_lock(&w->mutex);
#endif
    for (i = 0; i < nfound; i++)
        push_xv(w->out, w->nout, w->oalloc, found[i]);
    w->pairs_tested += npairs;
    w->divs_tested += ndivs;
    w->done++;
#if FLINT_USES_PTHREAD
    pthread_mutex_unlock(&w->mutex);
#endif
    flint_free(found);
    flint_free(dl);
}

/* sorted, duplicate-free */
static slong
sort_unique(xval_t * v, slong n)
{
    slong i, k;
    qsort(v, n, sizeof(xval_t), cmp_xval);
    for (i = k = 0; i < n; i++)
        if (k == 0 || xv_cmp(v[i], v[k - 1]) != 0)
            v[k++] = v[i];
    return k;
}

/* merge the new elements of out (destroyed) into xs (sorted); the
   added ones replace *T */
static slong
merge_new(xval_t ** xs, slong * nx, slong * xalloc, xval_t ** T, slong * nT,
    xval_t * out, slong nout)
{
    xval_t * add;
    slong nadd, i, j, k;

    nout = sort_unique(out, nout);
    add = flint_malloc((nout + 1) * sizeof(xval_t));
    for (i = j = nadd = 0; i < nout; i++)
    {
        while (j < *nx && xv_cmp((*xs)[j], out[i]) < 0)
            j++;
        if (j < *nx && xv_cmp((*xs)[j], out[i]) == 0)
            continue;
        add[nadd++] = out[i];
    }

    if (*nx + nadd > *xalloc)
    {
        *xalloc = *nx + nadd;
        *xs = flint_realloc(*xs, *xalloc * sizeof(xval_t));
    }
    for (i = *nx - 1, j = nadd - 1, k = *nx + nadd - 1; j >= 0; k--)
    {
        if (i >= 0 && xv_cmp((*xs)[i], add[j]) > 0)
            (*xs)[k] = (*xs)[i--];
        else
            (*xs)[k] = add[j--];
    }
    *nx += nadd;

    flint_free(*T);
    *T = add;
    *nT = nadd;
    return nadd;
}

/* one round: new elements of (zlo, zhi] are merged into xs (kept sorted);
   *T / *nT holds the elements added by the previous round and is
   replaced by those added now. Returns the number added. */
static slong
chm_round(xval_t ** xs, slong * nx, slong * xalloc, xval_t ** T, slong * nT,
    int first, int gaussian, slong NP, const ulong * norms,
    const ulong * pinv, const ulong * plim, xval_t zlo, xval_t zhi)
{
    chm_work_t w;
    xval_t * out = NULL;
    slong nout = 0, oalloc = 0, nadd, i, j;
    char * isnew;

    isnew = flint_calloc(*nx + 1, 1);
    for (i = j = 0; i < *nx && j < *nT; )
    {
        int c = xv_cmp((*xs)[i], (*T)[j]);
        if (c < 0) i++;
        else if (c > 0) j++;
        else { isnew[i] = 1; i++; j++; }
    }

    w.S = *xs; w.nS = *nx; w.T = *T; w.nT = *nT;
    uset_init(&w.HS, w.S, w.nS);
    uset_init(&w.HT, w.T, first ? 0 : w.nT);
    w.isnew = isnew; w.first = first;
    w.gaussian = gaussian; w.NP = NP;
    w.norms = norms; w.pinv = pinv; w.plim = plim;
    w.zlo = zlo; w.zhi = zhi;
    w.bsize = 256;
    w.nblocks = (w.nS + w.bsize - 1) / w.bsize;
    w.out = &out; w.nout = &nout; w.oalloc = &oalloc;
    w.pairs_tested = w.divs_tested = w.done = 0;
#if FLINT_USES_PTHREAD
    pthread_mutex_init(&w.mutex, NULL);
#endif
    flint_parallel_do((do_func_t) chm_block, &w, w.nblocks, -1,
        FLINT_PARALLEL_STRIDED);
#if FLINT_USES_PTHREAD
    pthread_mutex_destroy(&w.mutex);
#endif
    fprintf(stderr, "  combination round: %ld pair tests, %ld divisors, "
        "%ld z found", w.pairs_tested, w.divs_tested, nout);

    nadd = merge_new(xs, nx, xalloc, T, nT, out, nout);
    fprintf(stderr, ", %ld new\n", nadd);

    flint_free(out);
    flint_free(isnew);
    flint_free(w.HS.tab);
    flint_free(w.HT.tab);
    return nadd;
}

/* Extension of single elements. For smooth x and any d e = N(x)
   = x^2 -+ 1, (x + d)^2 -+ 1 = d (d + e + 2x) and
   (x + e)^2 -+ 1 = e (d + e + 2x), so y = x + d and z = x + e are both
   smooth iff m = d + e + 2x is smooth; unlike the rounds, y need not be
   known. This costs tau(N(x))/2 smoothness tests per x, so it is only
   applied to the K largest elements not extended before
   (K = MACHIN_SET_EXTEND). */

typedef struct
{
    const xval_t * X;
    slong n;
    int gaussian;
    slong NP;
    const ulong * norms, * pinv, * plim;
    xval_t zlo, zhi;
    double maxtau;
    char * extended;            /* extended[i]: X[i] was extended */
    xval_t ** out;
    slong * nout, * oalloc;
    slong ntests;
#if FLINT_USES_PTHREAD
    pthread_mutex_t mutex;
#endif
}
ext_work_t;

typedef struct
{
    const ext_work_t * w;
    xval_t x, dmax;
    ulong N[4];
    slong Nn;
    const ulong * q;
    const int * ex;
    slong m;
    xval_t * found;
    slong nfound, falloc, ntests;
}
ext_local_t;

static void
ext_leaf(ext_local_t * L, xval_t d)
{
    const ext_work_t * w = L->w;
    ulong dd[2], e[4], rr[2], mm[5], x2[2];
    slong dn = d.hi ? 2 : 1, en, mn;
    L->ntests++;
    dd[0] = d.lo; dd[1] = d.hi;
    flint_mpn_zero(e, 4);
    mpn_tdiv_qr(e, rr, 0, L->N, L->Nn, dd, dn);
    en = mpn_normsize(e, L->Nn - dn + 1);
    /* m = d + e + 2x */
    flint_mpn_zero(mm, 5);
    flint_mpn_copyi(mm, e, en);
    mpn_add(mm, mm, 5, dd, dn);
    x2[0] = L->x.lo; x2[1] = L->x.hi;
    mpn_add(mm, mm, 5, x2, 2);
    mpn_add(mm, mm, 5, x2, 2);
    mn = mpn_normsize(mm, 5);
    if (mn > 4 || !smooth_mpn(NULL, mm, mn, w->NP, w->norms, w->pinv, w->plim))
        return;
    {
        xval_t y = xv_add(L->x, d);
        if (xv_cmp(y, w->zhi) <= 0)
            push_xv(&L->found, &L->nfound, &L->falloc, y);
        if (en <= 2)
        {
            xval_t z, ev;
            ev.lo = e[0]; ev.hi = (en > 1) ? e[1] : 0;
            z = xv_add(L->x, ev);
            if (xv_cmp(z, ev) >= 0 && xv_cmp(z, w->zhi) <= 0)
                push_xv(&L->found, &L->nfound, &L->falloc, z);
        }
    }
}

static void
ext_dfs(ext_local_t * L, xval_t d, slong k)
{
    int a;
    if (k == L->m)
    {
        ext_leaf(L, d);
        return;
    }
    for (a = 0; ; a++)
    {
        ext_dfs(L, d, k + 1);
        if (a == L->ex[k] || !xv_mul_ui_bounded(&d, d, L->q[k], L->dmax))
            return;
    }
}

static void
ext_one(slong i, ext_work_t * w)
{
    ext_local_t L;
    int e[MAXP], ex[MAXP];
    ulong q[MAXP], root[2], rem[4];
    slong j, m;

    L.w = w; L.x = w->X[i];
    L.found = NULL; L.nfound = L.falloc = 0; L.ntests = 0;
    w->extended[i] = 0;
    if (!factor_N(e, L.N, &L.Nn, L.x, w->gaussian, w->NP, w->norms, w->pinv, w->plim))
        return;
    for (j = w->NP - 1, m = 0; j >= 0; j--)
        if (e[j] != 0)
        {
            q[m] = w->norms[j];
            ex[m] = e[j];
            m++;
        }
    L.q = q; L.ex = ex; L.m = m;
    {
        double tau = 1;
        for (j = 0; j < m; j++)
            tau *= ex[j] + 1;
        if (tau > w->maxtau)
            return;
    }
    w->extended[i] = 1;     /* distinct i: no lock needed */
    /* d <= sqrt(N) (d > sqrt(N) gives the same pairs) */
    root[1] = 0;
    mpn_sqrtrem(root, rem, L.N, L.Nn);
    L.dmax.lo = root[0];
    L.dmax.hi = ((L.Nn + 1) / 2 > 1) ? root[1] : 0;
    ext_dfs(&L, xv_ui(1), 0);

#if FLINT_USES_PTHREAD
    pthread_mutex_lock(&w->mutex);
#endif
    for (j = 0; j < L.nfound; j++)
        push_xv(w->out, w->nout, w->oalloc, L.found[j]);
    w->ntests += L.ntests;
#if FLINT_USES_PTHREAD
    pthread_mutex_unlock(&w->mutex);
#endif
    flint_free(L.found);
}

/* extend the K largest elements of xs (in [zlo/2, zhi)) not in done;
   the newly found elements are merged and replace *T */
static slong
ext_round(xval_t ** xs, slong * nx, slong * xalloc, xval_t ** T, slong * nT,
    xval_t ** done, slong * ndone, slong * dalloc, slong K,
    int gaussian, slong NP, const ulong * norms,
    const ulong * pinv, const ulong * plim, xval_t zlo, xval_t zhi)
{
    ext_work_t w;
    xval_t * sel = flint_malloc((K + 1) * sizeof(xval_t)), * out = NULL, half;
    slong nsel = 0, nout = 0, oalloc = 0, i, nadd;
    uset_t H;

    half.lo = (zlo.lo >> 1) | (zlo.hi << (FLINT_BITS - 1));
    half.hi = zlo.hi >> 1;
    /* optionally extend smaller x too: z = x + e can reach far above zlo */
    if (getenv("MACHIN_SET_EXTEND_FROM") != NULL)
    {
        fmpz_t t;
        xval_t from;
        fmpz_init(t);
        fmpz_set_d(t, atof(getenv("MACHIN_SET_EXTEND_FROM")));
        if (fmpz_bits(t) <= XV_MAXBITS && fmpz_sgn(t) >= 0)
        {
            fmpz_get_uiui(&from.hi, &from.lo, t);
            if (xv_cmp(from, half) < 0)
                half = from;
        }
        fmpz_clear(t);
    }
    uset_init(&H, *done, *ndone);
    for (i = *nx - 1; i >= 0 && nsel < K; i--)
    {
        xval_t x = (*xs)[i];
        if (xv_cmp(x, half) < 0)
            break;
        if (xv_cmp(x, zhi) >= 0 || uset_has(&H, x))
            continue;
        sel[nsel++] = x;
    }
    flint_free(H.tab);

    w.X = sel; w.n = nsel; w.gaussian = gaussian; w.NP = NP;
    w.norms = norms; w.pinv = pinv; w.plim = plim;
    w.zlo = zlo; w.zhi = zhi;
    w.out = &out; w.nout = &nout; w.oalloc = &oalloc;
    w.ntests = 0;
    w.extended = flint_calloc(nsel + 1, 1);
    w.maxtau = getenv("MACHIN_SET_EXTEND_MAXTAU") ? atof(getenv("MACHIN_SET_EXTEND_MAXTAU")) : 2e7;
#if FLINT_USES_PTHREAD
    pthread_mutex_init(&w.mutex, NULL);
#endif
    flint_parallel_do((do_func_t) ext_one, &w, nsel, -1, FLINT_PARALLEL_STRIDED);
#if FLINT_USES_PTHREAD
    pthread_mutex_destroy(&w.mutex);
#endif
    /* only elements actually extended (tau within the limit) are done */
    {
        slong ne = 0;
        for (i = 0; i < nsel; i++)
            if (w.extended[i])
            {
                push_xv(done, ndone, dalloc, sel[i]);
                ne++;
            }
        fprintf(stderr, "  extension: %ld x (%ld within the tau limit), "
            "%ld smoothness tests, %ld found", nsel, ne, w.ntests, nout);
    }
    flint_free(w.extended);
    nadd = merge_new(xs, nx, xalloc, T, nT, out, nout);
    fprintf(stderr, ", %ld new\n", nadd);
    flint_free(out);
    flint_free(sel);
    return nadd;
}

/* greedy selection by decreasing x: sel[0..NP) (decreasing), rows in
   the same order; returns the rank reached. Candidates are first
   tested for independence modulo a word-size prime (exact when
   independent); only apparent dependencies are confirmed over Z. */
static slong
select_rows(slong * sel, slong * rows, const xval_t * xs, slong nx,
    int gaussian, slong NP, const ulong * norms, const ulong * pinv,
    const ulong * plim, const slong * a_, const slong * b_, slong * ntried)
{
    nmod_t mod;
    ulong p = n_nextprime(UWORD(1) << (FLINT_BITS - 2), 1);
    ulong * E = flint_malloc(NP * NP * sizeof(ulong));   /* echelon rows */
    slong * piv = flint_malloc(NP * sizeof(slong));
    ulong * v = flint_malloc(NP * sizeof(ulong));
    slong ns = 0, i, j, k;

    nmod_init(&mod, p);
    *ntried = 0;
    for (i = nx - 1; i >= 0 && ns < NP; i--)
    {
        slong * c = rows + ns * NP;
        int indep;
        (*ntried)++;
        if (!factor_row(c, xs[i], gaussian, NP, norms, pinv, plim, a_, b_))
        {
            printf("factorization failed for x = ");
            xv_print(xs[i]);
            printf("\n");
            flint_abort();
        }
        /* reduce modulo p against the echelon basis */
        for (j = 0; j < NP; j++)
            v[j] = (c[j] >= 0) ? (ulong) c[j] % p : p - ((ulong) (-c[j]) % p);
        for (k = 0; k < ns; k++)
        {
            ulong f = v[piv[k]];
            if (f != 0)
                for (j = 0; j < NP; j++)
                    v[j] = nmod_sub(v[j], nmod_mul(f, E[k * NP + j], mod), mod);
        }
        for (j = 0; j < NP && v[j] == 0; j++) ;
        indep = (j < NP);
        if (!indep)
        {
            /* confirm over Z (a dependency mod p is almost always real) */
            fmpz_mat_t M;
            fmpz_mat_init(M, ns + 1, NP);
            for (k = 0; k <= ns; k++)
                for (j = 0; j < NP; j++)
                    fmpz_set_si(fmpz_mat_entry(M, k, j), rows[k * NP + j]);
            if (fmpz_mat_rank(M) == ns + 1)
            {
                printf("independence mod p failed; use another prime\n");
                flint_abort();
            }
            fmpz_mat_clear(M);
            continue;
        }
        /* normalize v at pivot j and eliminate it from the basis */
        {
            ulong inv = n_invmod(v[j], p);
            slong t;
            for (t = 0; t < NP; t++)
                v[t] = nmod_mul(v[t], inv, mod);
            for (k = 0; k < ns; k++)
            {
                ulong f = E[k * NP + j];
                if (f != 0)
                    for (t = 0; t < NP; t++)
                        E[k * NP + t] = nmod_sub(E[k * NP + t],
                            nmod_mul(f, v[t], mod), mod);
            }
            memcpy(E + ns * NP, v, NP * sizeof(ulong));
            piv[ns] = j;
        }
        sel[ns++] = i;
    }
    flint_free(E); flint_free(piv); flint_free(v);
    return ns;
}

/* ------------------------------------------------------------------ */
/* Exhaustive enumeration by Pell equations (Lehmer's method).

   x^2 - 1 = D w^2 (logarithms) or x^2 + 1 = D w^2 (arguments) with D > 1
   squarefree, so D is one of the 2^NP - 1 products of the primes (norms)
   and x is a solution of x^2 - D w^2 = 1 resp. -1. For each D the
   continued fraction of sqrt(D) is expanded until the convergents exceed
   XB: the first Q_(k+1) = 1 gives the fundamental unit (norm (-1)^(k+1)),
   and every solution <= XB is a power of it. Every smooth x <= XB is
   found, each exactly once (D is the squarefree part of x^2 -+ 1).
   D <= XB^2 + 1 < 2^(4B-4); P_k, Q_k < 2 sqrt(D) < 2^(2B-1). While
   D < 2^(2B-2) (most D for the logarithms), P_k, Q_k and the partial
   quotients fit in one limb.

   The cost is 2^NP short expansions, each O(log XB / log D) steps: a few
   core-minutes for NP = 24, doubling with each further prime. */

typedef struct
{
    int gaussian;
    slong NP, K;
    const ulong * norms, * pinv, * plim;
    xval_t XB;
    ulong Bnd[4];               /* XB^2 + 1 */
    slong Bn;
    xval_t ** xs;
    slong * nx, * xalloc;
    slong nequations, done, njobs;
    const slong * masks;        /* the jobs of this part */
#if FLINT_USES_PTHREAD
    pthread_mutex_t mutex;
#endif
}
pell_work_t;

typedef struct
{
    const pell_work_t * w;
    xval_t * found;
    slong nfound, falloc, neq;
}
pell_local_t;

/* a * b + c <= bound ? (a, b, c two-limb) */
static int
xv_muladd_bounded(xval_t * r, xval_t a, xval_t b, xval_t c, xval_t bound)
{
    if (!xv_mul_bounded(r, a, b, bound))
        return 0;
    *r = xv_add(*r, c);
    return xv_cmp(*r, c) >= 0 && xv_cmp(*r, bound) <= 0;
}

/* all solutions <= XB from the fundamental solution p of norm (-1)^s */
static void
pell_powers(pell_local_t * L, xval_t p, int negative)
{
    const pell_work_t * w = L->w;
    xval_t XB = w->XB, x1, xp, x, t, two = xv_ui(2), one = xv_ui(1);
    xval_t lim = xv_ui(0);
    lim.hi = UWORD(1) << (FLINT_BITS - 1);  /* 2^(2B-1) */

    if (!w->gaussian)
    {
        if (!negative)
            x1 = p;
        else
        {
            /* x1 = 2 p^2 + 1 */
            if (!xv_mul_bounded(&x1, p, p, XB))
                return;
            x1 = xv_add(xv_add(x1, x1), one);
            if (xv_cmp(x1, XB) > 0)
                return;
        }
        /* x_n = 2 x1 x_(n-1) - x_(n-2), x_0 = 1 */
        xp = one;
        x = x1;
        while (1)
        {
            if (smooth_xv(x, 0, w->NP, w->norms, w->pinv, w->plim))
                push_xv(&L->found, &L->nfound, &L->falloc, x);
            if (!xv_mul_bounded(&t, xv_add(x1, x1), x, lim))
                return;
            t = xv_sub(t, xp);
            if (xv_cmp(t, XB) > 0)
                return;
            xp = x;
            x = t;
        }
    }
    else
    {
        xval_t X, twoX;
        if (!negative)          /* x^2 - D w^2 = -1 has no solution */
            return;
        x1 = p;
        if (xv_cmp(x1, two) >= 0
            && smooth_xv(x1, 1, w->NP, w->norms, w->pinv, w->plim))
            push_xv(&L->found, &L->nfound, &L->falloc, x1);
        /* the other solutions are the odd powers of the unit of norm -1:
           u_(n+1) = 2 X u_n - u_(n-1), X = 2 x1^2 + 1, u_(-1) = -x1 */
        if (!xv_mul_bounded(&X, x1, x1, XB))
            return;
        X = xv_add(xv_add(X, X), one);
        twoX = xv_add(X, X);
        if (!xv_muladd_bounded(&x, twoX, x1, x1, XB))     /* u_1 */
            return;
        xp = x1;
        while (1)
        {
            if (smooth_xv(x, 1, w->NP, w->norms, w->pinv, w->plim))
                push_xv(&L->found, &L->nfound, &L->falloc, x);
            if (!xv_mul_bounded(&t, twoX, x, lim))
                return;
            t = xv_sub(t, xp);
            if (xv_cmp(t, XB) > 0)
                return;
            xp = x;
            x = t;
        }
    }
}

/* D < 2^(2B-2): single-limb P, Q, a */
static void
pell_solve_small(pell_local_t * L, ulong Dhi, ulong Dlo)
{
    const pell_work_t * w = L->w;
    ulong a0, a, Pk, Qk, P1, Q1, h, l, rr;
    xval_t p, pm1, pm2, t;
    slong k;
    {
        ulong s[2], root[1], rem[2];
        s[0] = Dlo; s[1] = Dhi;
        if (mpn_sqrtrem(root, rem, s, Dhi ? 2 : 1) == 0)
            return;         /* square */
        a0 = root[0];
    }
    L->neq++;
    Pk = 0; Qk = 1; a = a0;
    pm1 = xv_ui(1); pm2 = xv_ui(0);
    for (k = 0; ; k++)
    {
        /* p = a pm1 + pm2 */
        if (!xv_mul_ui_bounded(&t, pm1, a, w->XB))
            return;
        p = xv_add(t, pm2);
        if (xv_cmp(p, t) < 0 || xv_cmp(p, w->XB) > 0)
            return;
        P1 = a * Qk - Pk;
        umul_ppmm(h, l, P1, P1);
        sub_ddmmss(h, l, Dhi, Dlo, h, l);
        Q1 = divrem_ll(&rr, h, l, Qk);
        if (Q1 == 1)
            break;
        a = (a0 + P1) / Q1;
        Pk = P1; Qk = Q1; pm2 = pm1; pm1 = p;
    }
    pell_powers(L, p, !(k & 1));
}

/* general D (up to four limbs): two-limb P, Q, a */
static void
pell_solve_large(pell_local_t * L, const ulong * D, slong Dn)
{
    const pell_work_t * w = L->w;
    ulong root[2], rem[4], prod[4], tmp[4], qq[4], rr[2], u[2], v[2];
    xval_t a0, a, Pk, Qk, P1, Q1, p, pm1, pm2;
    slong k, n, qn;

    for (n = 0; n < 2; n++) root[n] = 0;
    if (mpn_sqrtrem(root, rem, D, Dn) == 0)
        return;
    a0.lo = root[0]; a0.hi = ((Dn + 1) / 2 > 1) ? root[1] : 0;
    L->neq++;
    Pk = xv_ui(0); Qk = xv_ui(1); a = a0;
    pm1 = xv_ui(1); pm2 = xv_ui(0);
    for (k = 0; ; k++)
    {
        if (!xv_muladd_bounded(&p, a, pm1, pm2, w->XB))
            return;
        /* P1 = a Qk - Pk */
        u[0] = a.lo; u[1] = a.hi; v[0] = Qk.lo; v[1] = Qk.hi;
        mpn_mul_n(prod, u, v, 2);
        P1.lo = prod[0]; P1.hi = prod[1];
        P1 = xv_sub(P1, Pk);
        /* Q1 = (D - P1^2) / Qk */
        u[0] = P1.lo; u[1] = P1.hi;
        mpn_sqr(prod, u, 2);
        flint_mpn_zero(tmp, 4);
        flint_mpn_copyi(tmp, D, Dn);
        mpn_sub_n(tmp, tmp, prod, 4);
        n = mpn_normsize(tmp, 4);
        v[0] = Qk.lo; v[1] = Qk.hi;
        {
            slong dn = Qk.hi ? 2 : 1;
            mpn_tdiv_qr(qq, rr, 0, tmp, n, v, dn);
            qn = n - dn + 1;
            Q1.lo = qq[0]; Q1.hi = (qn > 1) ? qq[1] : 0;
        }
        if (Q1.hi == 0 && Q1.lo == 1)
            break;
        /* a = (a0 + P1) / Q1 */
        {
            xval_t s = xv_add(a0, P1);
            slong sn = s.hi ? 2 : 1, dn = Q1.hi ? 2 : 1;
            u[0] = s.lo; u[1] = s.hi; v[0] = Q1.lo; v[1] = Q1.hi;
            if (sn < dn)
                a = xv_ui(0);
            else
            {
                mpn_tdiv_qr(qq, rr, 0, u, sn, v, dn);
                a.lo = qq[0]; a.hi = (sn - dn + 1 > 1) ? qq[1] : 0;
            }
        }
        Pk = P1; Qk = Q1; pm2 = pm1; pm1 = p;
    }
    pell_powers(L, p, !(k & 1));
}

static void
pell_solve(pell_local_t * L, const ulong * D, slong Dn)
{
    if (Dn == 1 && D[0] <= 1)
        return;
    if (Dn <= 1 || (Dn == 2 && (D[1] >> (FLINT_BITS - 2)) == 0))
        pell_solve_small(L, Dn == 2 ? D[1] : 0, D[0]);
    else
        pell_solve_large(L, D, Dn);
}

static int
mpn_leq(const ulong * a, slong an, const ulong * b, slong bn)
{
    if (an != bn)
        return an < bn;
    return mpn_cmp(a, b, an) <= 0;
}

/* all squarefree D = {D, Dn} * (product of a subset of norms[0..j]) */
static void
pell_dfs(pell_local_t * L, slong j, const ulong * D, slong Dn)
{
    ulong E[5];
    slong En;
    if (j < 0)
    {
        pell_solve(L, D, Dn);
        return;
    }
    pell_dfs(L, j - 1, D, Dn);
    E[Dn] = mpn_mul_1(E, D, Dn, L->w->norms[j]);
    En = Dn + (E[Dn] != 0);
    if (En > 4 || !mpn_leq(E, En, L->w->Bnd, L->w->Bn))
        return;
    pell_dfs(L, j - 1, E, En);
}

/* job: a subset (bit mask) of the K largest norms; the rest by dfs */
static void
pell_job(slong idx, pell_work_t * w)
{
    slong mask = w->masks[idx];
    pell_local_t L;
    ulong D[5];
    slong i, Dn = 1, NP = w->NP, K = w->K;

    L.w = w; L.found = NULL; L.nfound = L.falloc = 0; L.neq = 0;
    D[0] = 1;
    for (i = 0; i < K; i++)
    {
        if (((mask >> i) & 1) == 0)
            continue;
        D[Dn] = mpn_mul_1(D, D, Dn, w->norms[NP - K + i]);
        Dn += (D[Dn] != 0);
        if (Dn > 4 || !mpn_leq(D, Dn, w->Bnd, w->Bn))
            goto done;
    }
    pell_dfs(&L, NP - K - 1, D, Dn);

done:
#if FLINT_USES_PTHREAD
    pthread_mutex_lock(&w->mutex);
#endif
    for (i = 0; i < L.nfound; i++)
        push_xv(w->xs, w->nx, w->xalloc, L.found[i]);
    w->nequations += L.neq;
    w->done++;
    if (w->done % 64 == 0 || w->done == w->njobs)
        fprintf(stderr, "  Pell: %ld / %ld jobs, %ld equations, %ld x\n",
            w->done, w->njobs, w->nequations, *w->nx);
#if FLINT_USES_PTHREAD
    pthread_mutex_unlock(&w->mutex);
#endif
    flint_free(L.found);
}

static slong
pell_enumerate(xval_t ** xs, slong * nx, slong * xalloc, int gaussian,
    slong NP, const ulong * norms, const ulong * pinv, const ulong * plim,
    xval_t XB, slong part, slong nparts)
{
    pell_work_t w;
    ulong u[2];
    slong * masks, i, nm;
    w.gaussian = gaussian; w.NP = NP;
    w.K = FLINT_MIN(NP, 12);
    w.norms = norms; w.pinv = pinv; w.plim = plim;
    w.XB = XB;
    u[0] = XB.lo; u[1] = XB.hi;
    mpn_sqr(w.Bnd, u, 2);
    mpn_add_1(w.Bnd, w.Bnd, 4, 1);
    w.Bn = mpn_normsize(w.Bnd, 4);
    w.xs = xs; w.nx = nx; w.xalloc = xalloc;
    w.nequations = 0; w.done = 0;
    /* only the jobs of this part are handed to flint_parallel_do, so
       that the (strided) distribution over threads stays balanced */
    masks = flint_malloc((WORD(1) << w.K) * sizeof(slong));
    for (i = nm = 0; i < (WORD(1) << w.K); i++)
        if (i % nparts == part)
            masks[nm++] = i;
    w.masks = masks;
    w.njobs = nm;
#if FLINT_USES_PTHREAD
    pthread_mutex_init(&w.mutex, NULL);
#endif
    flint_parallel_do((do_func_t) pell_job, &w, w.njobs, -1,
        FLINT_PARALLEL_STRIDED);
#if FLINT_USES_PTHREAD
    pthread_mutex_destroy(&w.mutex);
#endif
    flint_free(masks);
    *nx = sort_unique(*xs, *nx);
    return w.nequations;
}

/* x lists in decimal, one per line */
static void
read_xv_file(xval_t ** v, slong * n, slong * alloc, const char * path)
{
    FILE * f = fopen(path, "r");
    char buf[256];
    fmpz_t t;
    if (f == NULL)
        return;
    fmpz_init(t);
    while (fscanf(f, "%250s", buf) == 1)
    {
        xval_t x;
        if (fmpz_set_str(t, buf, 10) != 0 || fmpz_sgn(t) <= 0
            || fmpz_bits(t) > XV_MAXBITS)
            continue;
        fmpz_get_uiui(&x.hi, &x.lo, t);
        push_xv(v, n, alloc, x);
    }
    fmpz_clear(t);
    fclose(f);
}

static void
write_xv_file(const xval_t * v, slong n, const char * path)
{
    char tmp[1024];
    FILE * f;
    fmpz_t t;
    slong i;
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    f = fopen(tmp, "w");
    if (f == NULL)
        return;
    fmpz_init(t);
    for (i = 0; i < n; i++)
    {
        fmpz_set_uiui(t, v[i].hi, v[i].lo);
        fmpz_fprint(f, t);
        fprintf(f, "\n");
    }
    fmpz_clear(t);
    fclose(f);
    rename(tmp, path);      /* atomic: an interrupted run keeps the old file */
}

/* The same formula as Python data (also appended to the file
   MACHIN_SET_PY if set):
       atanh_<n>_P / atan_<n>_P   primes p_i, or Gaussian primes (a_i, b_i)
       ..._X, ..._den, ..._C, ..._mu
   with log(p_i) = (1/den) sum_j C[i][j] atanh(1/X[j]), resp.
   atan(b_i/a_i) = (1/den) sum_j C[i][j] atan(1/X[j]), and
   mu = sum_j 1/log10(X[j]) (Lehmer's measure). */
static void
fprint_python(FILE * f, int gaussian, slong NP, const xval_t * xs, const slong * sel,
    const fmpz_t den, const fmpz_mat_t C, const slong * a_, const slong * b_,
    const ulong * norms)
{
    char pre[32];
    slong i, j, ind;
    double mu = 0;
    fmpz_t t;

    snprintf(pre, sizeof(pre), "%s_%ld", gaussian ? "atan" : "atanh", NP);
    fmpz_init(t);

    fprintf(f, "%s_P = [", pre);
    for (i = 0; i < NP; i++)
    {
        if (gaussian)
            fprintf(f, "%s(%ld, %ld)", i ? ", " : "", a_[i], b_[i]);
        else
            fprintf(f, "%s%lu", i ? ", " : "", norms[i]);
    }
    fprintf(f, "]\n");

    fprintf(f, "%s_X = [", pre);
    for (j = 0; j < NP; j++)
    {
        fmpz_set_uiui(t, xs[sel[j]].hi, xs[sel[j]].lo);
        if (j) fprintf(f, ", ");
        fmpz_fprint(f, t);
        mu += 1.0 / log10(xv_d(xs[sel[j]]));
    }
    fprintf(f, "]\n");

    fprintf(f, "%s_den = ", pre);
    fmpz_fprint(f, den);
    fprintf(f, "\n");

    fprintf(f, "%s_C = [", pre);
    ind = (slong) strlen(pre) + 6;          /* align with "[[" */
    for (i = 0; i < NP; i++)
    {
        if (i)
            fprintf(f, ",\n%*s", (int) ind, "");
        fprintf(f, "[");
        for (j = 0; j < NP; j++)
        {
            if (j) fprintf(f, ", ");
            fmpz_fprint(f, fmpz_mat_entry(C, i, j));
        }
        fprintf(f, "]");
    }
    fprintf(f, "]\n");
    fprintf(f, "%s_mu = %.5f\n", pre, mu);
    fmpz_clear(t);
}

int
main(int argc, char ** argv)
{
    int gaussian;
    slong NP, i, j, k, ROUNDS;
    double XMAX, ZMAX;

    int adaptive = 0, pellmode = 0, loadmode = 0;
    slong part = 0, nparts = 1;
    const char * loadfile = NULL;

    if (argc < 4 || argc > 8)
    {
        printf("usage: machin_set gaussian NP XMAX [ROUNDS [ZMAX [adaptive]]]\n"
               "       machin_set gaussian NP pell [ZMAX [PART NPARTS]]\n"
               "       machin_set gaussian NP load FILE [ROUNDS [ZMAX [adaptive]]]\n");
        return 1;
    }
    if (getenv("FLINT_NUM_THREADS") != NULL && atoi(getenv("FLINT_NUM_THREADS")) > 0)
        flint_set_num_threads(atoi(getenv("FLINT_NUM_THREADS")));
    gaussian = atoi(argv[1]);
    NP = atol(argv[2]);
    if (strcmp(argv[3], "pell") == 0)
    {
        pellmode = 1;
        ZMAX = (argc > 4) ? atof(argv[4]) : ldexp(1.0, XV_MAXBITS);
        if (ZMAX <= 0 || ZMAX > ldexp(1.0, XV_MAXBITS))
            ZMAX = ldexp(1.0, XV_MAXBITS);
        XMAX = FLINT_MIN(ZMAX, 0.25 * (double) UWORD_MAX);
        if (argc > 6)
        {
            part = atol(argv[5]);
            nparts = atol(argv[6]);
            if (nparts < 1 || part < 0 || part >= nparts)
            {
                printf("need 0 <= PART < NPARTS\n");
                return 1;
            }
        }
        ROUNDS = 0;
        argc = 4;
    }
    else if (strcmp(argv[3], "load") == 0 && argc > 4)
    {
        /* x from a file (one decimal per line), then optional rounds */
        loadmode = 1;
        loadfile = argv[4];
        XMAX = 10;
        ROUNDS = (argc > 5) ? atol(argv[5]) : 0;
        ZMAX = (argc > 6) ? atof(argv[6]) : ldexp(1.0, XV_MAXBITS);
        adaptive = (argc > 7) ? atoi(argv[7]) : 0;
        argc = 4;
    }
    else
    {
        XMAX = atof(argv[3]);
        ROUNDS = (argc > 4) ? atol(argv[4]) : 0;
        ZMAX = (argc > 5) ? atof(argv[5]) : ldexp(1.0, XV_MAXBITS);
    }
    if (ZMAX <= 0 || ZMAX > ldexp(1.0, XV_MAXBITS))
        ZMAX = ldexp(1.0, XV_MAXBITS);
    if (ZMAX < XMAX)
        ZMAX = XMAX;
    if (argc > 6)
        adaptive = atoi(argv[6]);
    if (NP < 2 || NP > MAXP || XMAX < 10.0 || XMAX > 0.25 * (double) UWORD_MAX)
    {
        printf("NP must be between 2 and %d and XMAX between 10 and 2^%d\n",
            MAXP, FLINT_BITS - 2);
        return 1;
    }
    ulong norms[MAXP], pinv[MAXP], plim[MAXP];
    slong a_[MAXP], b_[MAXP];
    sieve_entry * ent = NULL;
    slong nent = 0, alloc = 0;
    xval_t * xs = NULL;
    ulong * xs1 = NULL;
    slong nx1 = 0, xalloc1 = 0;
    xval_t zmax_x;
    slong * rows = NULL;
    slong nx = 0, xalloc = 0;

    /* the primes (rational, or the norms a^2 + b^2 of the Gaussian ones) */
    if (gaussian)
    {
        gaussian_primes(a_, b_, NP);
        for (j = 0; j < NP; j++)
            norms[j] = (ulong) (a_[j] * a_[j] + b_[j] * b_[j]);
    }
    else
    {
        n_primes_t it;
        n_primes_init(it);
        for (j = 0; j < NP; j++)
            norms[j] = n_primes_next(it);
        n_primes_clear(it);
    }

    {
        fmpz_t t;
        fmpz_init(t);
        fmpz_set_d(t, ZMAX);
        if (fmpz_bits(t) > XV_MAXBITS)
        {
            fmpz_one(t);
            fmpz_mul_2exp(t, t, XV_MAXBITS);
            fmpz_sub_ui(t, t, 1);
        }
        fmpz_get_uiui(&zmax_x.hi, &zmax_x.lo, t);
        fmpz_clear(t);
    }
    for (j = 0; j < NP; j++)
    {
        pinv[j] = (norms[j] & 1) ? binvert(norms[j]) : 0;
        plim[j] = UWORD_MAX / norms[j];
    }

    if (pellmode)
    {
        slong neq = pell_enumerate(&xs, &nx, &xalloc, gaussian, NP, norms,
            pinv, plim, zmax_x, part, nparts);
        printf("# Pell equations: %ld solved, all x up to %.3g, %d threads",
            neq, ZMAX, flint_get_num_threads());
        if (nparts > 1)
            printf(", part %ld of 0..%ld", part, nparts - 1);
        printf("\n");
    }
    else if (loadmode)
    {
        FILE * f = fopen(loadfile, "r");
        char buf[256];
        fmpz_t t;
        slong bad = 0;
        if (f == NULL)
        {
            printf("cannot open %s\n", loadfile);
            return 1;
        }
        fmpz_init(t);
        while (fscanf(f, "%250s", buf) == 1)
        {
            xval_t x;
            if (fmpz_set_str(t, buf, 10) != 0 || fmpz_cmp_ui(t, 2) < 0
                || fmpz_bits(t) > XV_MAXBITS)
            {
                bad++;
                continue;
            }
            fmpz_get_uiui(&x.hi, &x.lo, t);
            if (!smooth_xv(x, gaussian, NP, norms, pinv, plim))
            {
                bad++;
                continue;
            }
            push_xv(&xs, &nx, &xalloc, x);
        }
        fclose(f);
        fmpz_clear(t);
        nx = sort_unique(xs, nx);
        printf("# loaded %ld smooth x from %s (%ld entries skipped)\n",
            nx, loadfile, bad);
    }
    else
    {
        /* sieve entries (logarithms): z = 0 mod p^k for z = x - 1 and
           z + 2 = x + 1, up to z + 2 = XMAX + 1 */
        if (!gaussian)
        {
            ulong zmax = (ulong) XMAX + 1;
            for (j = 0; j < NP; j++)
            {
                ulong p = norms[j], pk = p;
                while (1)
                {
                    if (nent == alloc)
                    {
                        alloc = 2 * alloc + 64;
                        ent = flint_realloc(ent, alloc * sizeof(sieve_entry));
                    }
                    ent[nent].mod = pk;
                    ent[nent].r = 0;
                    ent[nent].lg16 = (int) ceil(16.0 * log2((double) p));
                    nent++;
                    if (pk > zmax / p)
                        break;
                    pk *= p;
                }
            }
        }
        else
        {
            /* p = 2: odd x, once */
            alloc = 64;
            ent = flint_realloc(ent, alloc * sizeof(sieve_entry));
            ent[0].mod = 2; ent[0].r = 1; ent[0].lg16 = 16;
            nent = 1;
        }
        /* sieve entries: roots of x^2 = -1 mod p^k, Hensel-lifted, for
           odd p (p = 2 divides x^2 + 1 exactly once, for odd x) */
        for (j = 1; j < NP && gaussian; j++)
        {
            ulong p = norms[j], pk = p, r;
            int e;
            r = n_sqrtmod(p - 1, p);            /* r^2 = -1 mod p */
            for (e = 1; ; e++)
            {
                ulong roots[2] = { r, pk - r };
                int t;
                for (t = 0; t < 2; t++)
                {
                    if (nent == alloc)
                    {
                        alloc = 2 * alloc + 64;
                        ent = flint_realloc(ent, alloc * sizeof(sieve_entry));
                    }
                    ent[nent].mod = pk;
                    ent[nent].r = roots[t];
                    ent[nent].lg16 = (int) ceil(16.0 * log2((double) p));
                    nent++;
                }
                if ((double) pk * p > XMAX * XMAX + 2.0 || pk > UWORD_MAX / p)
                    break;
                /* lift r to mod p^(e+1): r' = r - p^e d with
                   d = ((r^2 + 1) / p^e) / (2 r) mod p */
                {
                    ulong pk1 = pk * p, d, t2, inv;
                    fmpz_t T;
                    fmpz_init(T);
                    fmpz_set_ui(T, r);
                    fmpz_mul(T, T, T);
                    fmpz_add_ui(T, T, 1);
                    fmpz_divexact_ui(T, T, pk);
                    t2 = fmpz_fdiv_ui(T, p);
                    inv = n_invmod((2 * r) % p, p);
                    d = n_mulmod2_preinv(t2, inv, p, n_preinvert_limb(p));
                    fmpz_set_ui(T, pk);
                    fmpz_mul_ui(T, T, d);
                    fmpz_neg(T, T);
                    fmpz_add_ui(T, T, r);
                    fmpz_mod_ui(T, T, pk1);
                    r = fmpz_get_ui(T);
                    pk = pk1;
                    fmpz_clear(T);
                }
            }
        }
        printf("# sieve: %ld prime-power roots, x up to %.3g, %d threads\n", nent, XMAX, flint_get_num_threads());
        fflush(stdout);

        /* segmented sieve, one thread per segment (flint_parallel_do
           over the segments; the smooth x found are appended under a
           lock and sorted afterwards) */
        {
            sieve_work_t work;
            ulong P;
            unsigned short * pat;
            slong nsmall, nseg;

            pat = build_pattern(&P, ent, &nent, norms, NP);
            qsort(ent, nent, sizeof(sieve_entry), cmp_entry);
            for (nsmall = 0; nsmall < nent && ent[nsmall].mod < BLOCK; nsmall++) ;
            printf("# pattern period %lu, %ld scattered entries (%ld small)\n", P, nent, nsmall);

            work.ent = ent; work.nent = nent; work.nsmall = nsmall;
            work.pat = pat; work.P = P;
            work.gaussian = gaussian;
            work.NP = NP; work.norms = norms; work.pinv = pinv; work.plim = plim;
            work.zlo = gaussian ? 2 : 1;
            work.zhi = gaussian ? (ulong) XMAX : (ulong) XMAX - 1;
            nseg = (slong) ((work.zhi - work.zlo) / SEG) + 1;
            work.xs = &xs1; work.nx = &nx1; work.xalloc = &xalloc1;
            work.nseg = nseg; work.done = 0;
    #if FLINT_USES_PTHREAD
            pthread_mutex_init(&work.mutex, NULL);
    #endif
            flint_parallel_do((do_func_t) sieve_segment, &work, nseg, -1,
                FLINT_PARALLEL_STRIDED);
    #if FLINT_USES_PTHREAD
            pthread_mutex_destroy(&work.mutex);
    #endif
            flint_free(pat);
            qsort(xs1, nx1, sizeof(ulong), cmp_ulong);
            xs = flint_malloc((nx1 + 1) * sizeof(xval_t));
            for (i = 0; i < nx1; i++)
                xs[i] = xv_ui(xs1[i]);
            nx = xalloc = nx1;
            flint_free(xs1);
        }
    }
    printf("# %ld smooth x found; largest: ", nx);
    for (i = nx - 1; i >= FLINT_MAX(nx - 5, 0); i--)
    {
        xv_print(xs[i]);
        printf(" ");
    }
    printf("\n");
    fflush(stdout);

    /* combination rounds */
    rows = flint_malloc((NP + 1) * NP * sizeof(slong));
    if (ROUNDS > 0)
    {
        xval_t * T = NULL, * edone = NULL;
        slong nedone = 0, edalloc = 0;
        slong extK = getenv("MACHIN_SET_EXTEND") ? atol(getenv("MACHIN_SET_EXTEND")) : 0;
        const char * donefile = getenv("MACHIN_SET_EXTEND_DONE");
        slong nT = 0, r, nadd, ns, ntried;
        slong * sel = flint_malloc(NP * sizeof(slong));
        xval_t zlo = xv_ui((ulong) XMAX), zhi = zmax_x;
        if (donefile != NULL)
        {
            read_xv_file(&edone, &nedone, &edalloc, donefile);
            nedone = sort_unique(edone, nedone);
            fprintf(stderr, "  %ld x already extended (%s)\n", nedone, donefile);
        }
        if (adaptive)
        {
            ns = select_rows(sel, rows, xs, nx, gaussian, NP, norms, pinv, plim, a_, b_, &ntried);
            if (ns == NP && xv_cmp(xs[sel[NP - 1]], zlo) > 0)
                zlo = xs[sel[NP - 1]];
        }
        for (r = 0; r < ROUNDS; r++)
        {
            nadd = chm_round(&xs, &nx, &xalloc, &T, &nT, r == 0, gaussian,
                NP, norms, pinv, plim, zlo, zhi);
            if (extK > 0)
            {
                /* the extension's finds are paired in the next round too */
                xval_t * T2 = NULL;
                slong nT2 = 0, nadd2, a, b, c;
                nadd2 = ext_round(&xs, &nx, &xalloc, &T2, &nT2, &edone,
                    &nedone, &edalloc, extK, gaussian, NP, norms, pinv,
                    plim, zlo, zhi);
                if (donefile != NULL)
                {
                    nedone = sort_unique(edone, nedone);
                    write_xv_file(edone, nedone, donefile);
                }
                /* T = T union T2 */
                T = flint_realloc(T, (nT + nT2 + 1) * sizeof(xval_t));
                for (a = 0; a < nT2; a++)
                    T[nT + a] = T2[a];
                c = sort_unique(T, nT + nT2);
                (void) b;
                nT = c;
                flint_free(T2);
                nadd += nadd2;
            }
            ns = select_rows(sel, rows, xs, nx, gaussian, NP, norms, pinv, plim, a_, b_, &ntried);
            printf("# round %ld: %ld new, %ld total; ", r + 1, nadd, nx);
            if (ns == NP)
            {
                printf("selected x from ");
                xv_print(xs[sel[NP - 1]]);
                printf(" to ");
                xv_print(xs[sel[0]]);
                printf(" (%ld tried)\n", ntried);
            }
            else
                printf("rank %ld\n", ns);
            fflush(stdout);
            if (nadd == 0)
                break;
            /* only z above the current smallest selected x can change
               the selection (this also stops seeding from below it) */
            if (adaptive && ns == NP && xv_cmp(xs[sel[NP - 1]], zlo) > 0)
                zlo = xs[sel[NP - 1]];
        }
        flint_free(sel);
        flint_free(T);
        flint_free(edone);
    }

    /* Gaussian factorization rows: c[j] = a_j - b_j, c[0] in units
       of pi/4 including the unit */
    /* greedy selection by decreasing x, factoring candidates on demand
       (only the rows tried are kept: rows[k] belongs to sel[k]) */
    if (getenv("MACHIN_SET_SAVE") != NULL)
    {
        FILE * f = fopen(getenv("MACHIN_SET_SAVE"), "w");
        if (f != NULL)
        {
            fmpz_t t;
            fmpz_init(t);
            for (i = 0; i < nx; i++)
            {
                fmpz_set_uiui(t, xs[i].hi, xs[i].lo);
                fmpz_fprint(f, t);
                fprintf(f, "\n");
            }
            fmpz_clear(t);
            fclose(f);
            printf("# %ld x saved to %s\n", nx, getenv("MACHIN_SET_SAVE"));
        }
    }

    {
        fmpz_mat_t M;
        slong * sel = flint_malloc(NP * sizeof(slong));
        slong ns, ntried;
        fmpz_mat_init(M, NP, NP);
        ns = select_rows(sel, rows, xs, nx, gaussian, NP, norms, pinv, plim, a_, b_, &ntried);
        if (ns < NP)
        {
            printf("# only rank %ld: not enough x for a complete set\n", ns);
            return 1;
        }
        /* rows in increasing x for the output */
        for (k = 0; k < ns / 2; k++)
        {
            slong t = sel[k]; sel[k] = sel[ns - 1 - k]; sel[ns - 1 - k] = t;
            for (j = 0; j < NP; j++)
            {
                slong u = rows[k * NP + j];
                rows[k * NP + j] = rows[(ns - 1 - k) * NP + j];
                rows[(ns - 1 - k) * NP + j] = u;
            }
        }

        for (k = 0; k < NP; k++)
            for (j = 0; j < NP; j++)
                fmpz_set_si(fmpz_mat_entry(M, k, j), rows[k * NP + j]);

        /* invert: arg = M^-1 atan-vector */
        {
            fmpq_mat_t Q, Qi;
            fmpz_mat_t C;
            fmpz_t den;
            fmpq_mat_init(Q, NP, NP);
            fmpq_mat_init(Qi, NP, NP);
            fmpz_mat_init(C, NP, NP);
            fmpz_init(den);
            fmpq_mat_set_fmpz_mat(Q, M);
            fmpq_mat_inv(Qi, Q);
            if (!gaussian)
            {
                fmpz_t two;
                fmpz_init_set_ui(two, 2);
                fmpq_mat_scalar_mul_fmpz(Qi, Qi, two);
                fmpz_clear(two);
            }
            fmpq_mat_get_fmpz_mat_matwise(C, den, Qi);

            {
                char descr[512];
                const char * src = getenv("MACHIN_SET_SOURCE");
                if (src != NULL)
                    snprintf(descr, sizeof(descr), "%s", src);
                else if (pellmode)
                    snprintf(descr, sizeof(descr), "all x up to %.3g by Pell equations", ZMAX);
                else if (loadmode)
                    snprintf(descr, sizeof(descr), "x loaded from %s, %ld combination rounds", loadfile, ROUNDS);
                else if (ROUNDS > 0)
                    snprintf(descr, sizeof(descr), "sieved up to %.3g, %ld combination rounds", XMAX, ROUNDS);
                else
                    snprintf(descr, sizeof(descr), "sieved up to %.3g", XMAX);
                printf("# %s_%ld: x^2 %s 1 smooth, %s\n", gaussian ? "atan" : "atanh", NP,
                    gaussian ? "+" : "-", descr);
                fprint_python(stdout, gaussian, NP, xs, sel, den, C, a_, b_, norms);
                if (getenv("MACHIN_SET_PY") != NULL)
                {
                    FILE * f = fopen(getenv("MACHIN_SET_PY"), "a");
                    if (f != NULL)
                    {
                        fprint_python(f, gaussian, NP, xs, sel, den, C, a_, b_, norms);
                        fprintf(f, "\n");
                        fclose(f);
                    }
                }
            }
            /* verify with arb */
            {
                arb_ptr y = _arb_vec_init(NP);
                arb_t s, r; fmpz_t p, q; int bad = 0;
                arb_init(s); arb_init(r); fmpz_init(p); fmpz_init(q);
                for (k = 0; k < NP; k++) { fmpz_one(p); fmpz_set_uiui(q, xs[sel[k]].hi, xs[sel[k]].lo); arb_atan_frac_bsplit(y + k, p, q, gaussian == 0, 300); }
                for (i = 0; i < NP; i++)
                {
                    fmpq_t fr;
                    arb_zero(s);
                    for (j = 0; j < NP; j++) arb_addmul_fmpz(s, y + j, fmpz_mat_entry(C, i, j), 300);
                    arb_div_fmpz(s, s, den, 300);
                    fmpq_init(fr);
                    if (gaussian)
                    {
                        fmpq_set_si(fr, b_[i], a_[i]);
                        arb_set_fmpq(r, fr, 300); arb_atan(r, r, 300);
                    }
                    else
                        arb_log_ui(r, norms[i], 300);
                    if (!arb_overlaps(s, r)) { printf("# verification FAILED for prime %ld\n", i); bad = 1; }
                    fmpq_clear(fr);
                }
                printf("# verification: %s\n", bad ? "FAILED" : "ok");
            }
            fmpq_mat_clear(Q); fmpq_mat_clear(Qi); fmpz_mat_clear(C); fmpz_clear(den);
        }
        fmpz_mat_clear(M);
    }
    return 0;
}
