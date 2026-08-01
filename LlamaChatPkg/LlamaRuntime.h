/* =========================================================================
 * LlamaRuntime.h  --  Freestanding C-runtime shims for the EFI LLM chat.
 *
 *  - C runtime intrinsics (memset/memcpy/memmove) mapped onto the EDK II
 *    BaseMemoryLib primitives (SetMem/CopyMem) -- same MSVC /alternatename
 *    trick used by the Cryptor project.
 *  - A tiny software math library (no libm in UEFI): exp, log, sqrt, sin,
 *    cos, pow, fabs (double core + float wrappers).
 *  - qsort / bsearch used by the tokenizer and the top-p sampler.
 *  - A couple of ASCII string->number helpers.
 * ========================================================================= */
#ifndef LLAMA_RUNTIME_H
#define LLAMA_RUNTIME_H

#include <Uefi.h>
#include <Library/BaseMemoryLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseLib.h>

/* ---- memory helpers (calloc/malloc/free equivalents) ------------------- */
VOID*  LmAlloc(UINTN Size);
VOID*  LmCalloc(UINTN Count, UINTN Size);
VOID   LmFree(VOID* Ptr);

/* ---- software floating point math (float wrappers) --------------------- */
float  LmSqrtf(float x);
float  LmExpf(float x);
float  LmSinf(float x);
float  LmCosf(float x);
float  LmPowf(float base, float exp);
float  LmFabsf(float x);
float  LmFastExpf(float x);   /* SSE2 exp (~2e-7): горячие циклы softmax/silu */

/* double cores, exposed in case they are handy */
double LmSqrt(double x);
double LmExp(double x);
double LmLog(double x);
double LmSin(double x);
double LmCos(double x);

/* ---- sort / search ----------------------------------------------------- */
typedef int (*LM_CMP)(CONST VOID* a, CONST VOID* b);
VOID   LmQsort(VOID* Base, UINTN Num, UINTN Size, LM_CMP Cmp);
VOID*  LmBsearch(CONST VOID* Key, CONST VOID* Base, UINTN Num, UINTN Size, LM_CMP Cmp);

/* ---- ASCII number parsing ---------------------------------------------- */
float  LmAtof(CONST CHAR8* s);
INTN   LmAtoi(CONST CHAR8* s);

#endif /* LLAMA_RUNTIME_H */
