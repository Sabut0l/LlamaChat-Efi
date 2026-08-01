/* =========================================================================
 * LlamaRuntime.c  --  Freestanding C-runtime shims (see LlamaRuntime.h).
 * ========================================================================= */
#include "LlamaRuntime.h"

/* =========================================================================
 * C-runtime intrinsics (memset/memcpy/memmove) mapped onto EDK II BaseMemoryLib.
 * This is the exact MSVC /alternatename technique used by the Cryptor project,
 * so structure/array initialization the compiler lowers to memset/memcpy works
 * without a real CRT. On MSVC we also emit _fltused because the module uses
 * floating point.
 * ========================================================================= */
void* my_memset(void* dest, int c, UINTN count)              { return SetMem(dest, count, (UINT8)c); }
void* my_memcpy(void* dest, const void* src, UINTN count)    { return CopyMem(dest, src, count); }
void* my_memmove(void* dest, const void* src, UINTN count)   { return CopyMem(dest, src, count); }

#ifdef _MSC_VER
int _fltused = 0;   /* MSVC marker: this object uses floating point */
#if defined(_M_IX86)
#pragma comment(linker, "/include:_my_memset")
#pragma comment(linker, "/include:_my_memcpy")
#pragma comment(linker, "/include:_my_memmove")
#pragma comment(linker, "/alternatename:_memset=_my_memset")
#pragma comment(linker, "/alternatename:_memcpy=_my_memcpy")
#pragma comment(linker, "/alternatename:_memmove=_my_memmove")
#pragma comment(linker, "/alternatename:memset=_my_memset")
#pragma comment(linker, "/alternatename:memcpy=_my_memcpy")
#pragma comment(linker, "/alternatename:memmove=_my_memmove")
#else
#pragma comment(linker, "/include:my_memset")
#pragma comment(linker, "/include:my_memcpy")
#pragma comment(linker, "/include:my_memmove")
#pragma comment(linker, "/alternatename:memset=my_memset")
#pragma comment(linker, "/alternatename:memcpy=my_memcpy")
#pragma comment(linker, "/alternatename:memmove=my_memmove")
#endif
#endif /* _MSC_VER */

/* =========================================================================
 * Memory helpers
 * ========================================================================= */
VOID* LmAlloc(UINTN Size) {
    if (Size == 0) Size = 1;
    return AllocatePool(Size);
}
VOID* LmCalloc(UINTN Count, UINTN Size) {
    UINTN Total = Count * Size;
    if (Total == 0) Total = 1;
    return AllocateZeroPool(Total);   /* calloc-like: zeroed */
}
VOID LmFree(VOID* Ptr) {
    if (Ptr != NULL) FreePool(Ptr);
}

/* =========================================================================
 * Software math -- there is no libm in UEFI. Double-precision cores with
 * range reduction + polynomial/series approximations; float wrappers on top.
 * Accuracy is more than sufficient for a small Llama-2 transformer.
 * ========================================================================= */
typedef union { double d; UINT64 u; } LmDU;

STATIC double LmLdexp(double x, INT64 n) {
    /* multiply x by 2^n by direct exponent construction, with clamping */
    while (n >  1023) { x *= 8.98846567431158e307;    n -= 1023; }   /* 2^1023 */
    while (n < -1022) { x *= 2.2250738585072014e-308; n += 1022; }   /* 2^-1022 */
    LmDU v;
    v.u = (UINT64)(n + 1023) << 52;
    return x * v.d;
}

double LmSqrt(double x) {
    if (x <= 0.0) return 0.0;
    /* initial guess by halving the exponent field */
    LmDU v; v.d = x;
    v.u = (v.u >> 1) + (0x3FF0000000000000ULL >> 1);
    double y = v.d;
    /* Newton-Raphson: y = (y + x/y)/2 */
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    return y;
}

double LmExp(double x) {
    if (x > 709.0)  return 1.0e308 * 1.0e308;  /* +inf */
    if (x < -745.0) return 0.0;
    const double LN2     = 0.69314718055994530942;
    const double INV_LN2 = 1.44269504088896340736;
    double kf = x * INV_LN2;
    INT64  k  = (INT64)(kf >= 0.0 ? kf + 0.5 : kf - 0.5);
    double r  = x - (double)k * LN2;           /* r in [-ln2/2, ln2/2] */
    /* exp(r) via Horner, coefficients 1/n! for n=7..0 */
    double p = 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    return LmLdexp(p, k);
}

double LmLog(double x) {
    if (x <= 0.0) return -1.0e308;
    LmDU v; v.d = x;
    INT64 e = (INT64)((v.u >> 52) & 0x7FF) - 1023;
    v.u = (v.u & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;  /* m in [1,2) */
    double m = v.d;
    if (m > 1.41421356237309515) { m *= 0.5; e += 1; }           /* center around 1 */
    /* log(m) = 2*(s + s^3/3 + s^5/5 + ...), s = (m-1)/(m+1) */
    double s  = (m - 1.0) / (m + 1.0);
    double s2 = s * s;
    double term = s;
    double sum  = s;
    term *= s2; sum += term / 3.0;
    term *= s2; sum += term / 5.0;
    term *= s2; sum += term / 7.0;
    term *= s2; sum += term / 9.0;
    term *= s2; sum += term / 11.0;
    term *= s2; sum += term / 13.0;
    term *= s2; sum += term / 15.0;
    return (double)e * 0.69314718055994530942 + 2.0 * sum;
}

double LmSin(double x) {
    const double TWO_PI = 6.28318530717958647692;
    /* reduce to [-pi, pi] */
    double kf = x / TWO_PI;
    INT64  k  = (INT64)(kf >= 0.0 ? kf + 0.5 : kf - 0.5);
    x = x - (double)k * TWO_PI;
    double x2 = x * x;
    double t  = x;
    double sum = x;
    t *= x2; sum -= t / 6.0;            /* x^3/3!  */
    t *= x2; sum += t / 120.0;          /* x^5/5!  */
    t *= x2; sum -= t / 5040.0;         /* x^7/7!  */
    t *= x2; sum += t / 362880.0;       /* x^9/9!  */
    t *= x2; sum -= t / 39916800.0;     /* x^11/11!*/
    t *= x2; sum += t / 6227020800.0;   /* x^13/13!*/
    return sum;
}

double LmCos(double x) {
    return LmSin(x + 1.57079632679489661923);
}

float LmSqrtf(float x)            { return (float)LmSqrt((double)x); }
float LmExpf(float x)             { return (float)LmExp((double)x); }
float LmSinf(float x)             { return (float)LmSin((double)x); }
float LmCosf(float x)             { return (float)LmCos((double)x); }
float LmPowf(float b, float e)    { return (float)LmExp((double)e * LmLog((double)b)); }
float LmFabsf(float x)            { return x < 0.0f ? -x : x; }

/* Быстрый одинарный exp для горячих циклов (softmax, silu). SSE2 на x64:
 * k = round(x/ln2), полином 6-й степени на r in [-ln2/2, ln2/2], 2^k через
 * поле экспоненты. Отн. ошибка ~2e-7 на [-87,88] -- для fp32-активаций
 * трансформера заведомо достаточно. В 5-10 раз быстрее double-ядра выше. */
#if defined(MDE_CPU_X64) || defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
float LmFastExpf(float x) {
    if (x > 88.0f)  return 3.402823466e38f;
    if (x < -87.0f) return 0.0f;
    __m128 vx = _mm_set_ss(x);
    int   k  = _mm_cvtss_si32(_mm_mul_ss(vx, _mm_set_ss(1.4426950408889634f)));
    float r  = x - (float)k * 0.6931471805599453f;
    __m128 vr = _mm_set_ss(r);
    __m128 p = _mm_set_ss(1.3888888889e-3f);
    p = _mm_add_ss(_mm_mul_ss(p, vr), _mm_set_ss(8.3333333333e-3f));
    p = _mm_add_ss(_mm_mul_ss(p, vr), _mm_set_ss(4.1666666667e-2f));
    p = _mm_add_ss(_mm_mul_ss(p, vr), _mm_set_ss(1.6666666667e-1f));
    p = _mm_add_ss(_mm_mul_ss(p, vr), _mm_set_ss(5.0000000000e-1f));
    p = _mm_add_ss(_mm_mul_ss(p, vr), _mm_set_ss(1.0000000000e+0f));
    p = _mm_add_ss(_mm_mul_ss(p, vr), _mm_set_ss(1.0000000000e+0f));
    __m128i two_k = _mm_slli_epi32(_mm_add_epi32(_mm_set1_epi32(k), _mm_set1_epi32(127)), 23);
    return _mm_cvtss_f32(_mm_mul_ss(p, _mm_castsi128_ps(two_k)));
}
#else
float LmFastExpf(float x) { return LmExpf(x); }
#endif

/* =========================================================================
 * qsort / bsearch (self-contained; element size <= 64 bytes)
 * ========================================================================= */
STATIC VOID LmSwap(UINT8* a, UINT8* b, UINTN size) {
    UINT8 tmp[64];
    while (size > 0) {
        UINTN chunk = size > sizeof(tmp) ? sizeof(tmp) : size;
        CopyMem(tmp, a, chunk);
        CopyMem(a, b, chunk);
        CopyMem(b, tmp, chunk);
        a += chunk; b += chunk; size -= chunk;
    }
}

STATIC VOID LmQsortRange(UINT8* base, INTN lo, INTN hi, UINTN size, LM_CMP cmp) {
    while (lo < hi) {
        /* median-of-three pivot into base[hi] */
        INTN mid = lo + (hi - lo) / 2;
        if (cmp(base + mid * size, base + lo * size) < 0) LmSwap(base + mid * size, base + lo * size, size);
        if (cmp(base + hi  * size, base + lo * size) < 0) LmSwap(base + hi  * size, base + lo * size, size);
        if (cmp(base + hi  * size, base + mid * size) < 0) LmSwap(base + hi  * size, base + mid * size, size);
        UINT8* pivot = base + mid * size;
        LmSwap(pivot, base + hi * size, size);
        pivot = base + hi * size;
        INTN store = lo;
        for (INTN i = lo; i < hi; i++) {
            if (cmp(base + i * size, pivot) < 0) {
                LmSwap(base + i * size, base + store * size, size);
                store++;
            }
        }
        LmSwap(base + store * size, base + hi * size, size);
        /* recurse into the smaller side, loop on the larger (bounded stack) */
        if (store - lo < hi - store) {
            LmQsortRange(base, lo, store - 1, size, cmp);
            lo = store + 1;
        } else {
            LmQsortRange(base, store + 1, hi, size, cmp);
            hi = store - 1;
        }
    }
}

VOID LmQsort(VOID* Base, UINTN Num, UINTN Size, LM_CMP Cmp) {
    if (Num < 2 || Size == 0 || Size > 64) return;
    LmQsortRange((UINT8*)Base, 0, (INTN)Num - 1, Size, Cmp);
}

VOID* LmBsearch(CONST VOID* Key, CONST VOID* Base, UINTN Num, UINTN Size, LM_CMP Cmp) {
    INTN lo = 0, hi = (INTN)Num - 1;
    CONST UINT8* b = (CONST UINT8*)Base;
    while (lo <= hi) {
        INTN mid = lo + (hi - lo) / 2;
        int c = Cmp(Key, b + mid * Size);
        if (c == 0) return (VOID*)(b + mid * Size);
        if (c < 0)  hi = mid - 1;
        else        lo = mid + 1;
    }
    return NULL;
}

/* =========================================================================
 * ASCII number parsing
 * ========================================================================= */
STATIC BOOLEAN LmIsSpace(CHAR8 c) { return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }

INTN LmAtoi(CONST CHAR8* s) {
    if (s == NULL) return 0;
    while (*s && LmIsSpace(*s)) s++;
    INTN sign = 1;
    if (*s == '-') { sign = -1; s++; } else if (*s == '+') s++;
    INTN v = 0;
    while (*s >= '0' && *s <= '9') { v = v * 10 + (*s - '0'); s++; }
    return sign * v;
}

float LmAtof(CONST CHAR8* s) {
    if (s == NULL) return 0.0f;
    while (*s && LmIsSpace(*s)) s++;
    double sign = 1.0;
    if (*s == '-') { sign = -1.0; s++; } else if (*s == '+') s++;
    double v = 0.0;
    while (*s >= '0' && *s <= '9') { v = v * 10.0 + (*s - '0'); s++; }
    if (*s == '.') {
        s++;
        double scale = 0.1;
        while (*s >= '0' && *s <= '9') { v += (*s - '0') * scale; scale *= 0.1; s++; }
    }
    return (float)(sign * v);
}
