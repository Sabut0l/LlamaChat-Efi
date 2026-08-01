//Copyright (c) 2026 Sabut0l
/* =========================================================================
 * LlamaMp.h  --  Multi-core matmul using EFI MP Services (Application
 * Processors). Persistent worker pool: StartupThisAP is called once per AP
 * at MmInit; each AP then spins on a shared descriptor and executes a
 * row-slice of the current matmul (matmul is the hot loop of the
 * transformer, so per-call/per-token StartupThisAP overhead is
 * unacceptable -- firmware only signals AP completion from a periodic
 * status-check timer, ~100 ms per wait on many platforms).
 *
 * Hardened for real hardware:
 *   - SSE is explicitly enabled on every AP (CR0.MP, ~EM, ~TS;
 *     CR4.OSFXSR, OSXMMEXCPT) and verified by an FP self-test before the
 *     worker goes online;
 *   - all waits are bounded: workers that fail to start are excluded at
 *     init, workers that stop acking are retired by a watchdog and their
 *     slice is recomputed on the BSP. Worst case the app degrades to
 *     single core -- it never freezes.
 *
 * AP tasks run at TPL_HIGH_LEVEL: they touch memory and floats only -- no
 * Boot Services, no protocols.
 * ========================================================================= */
#ifndef LLAMA_MP_H
#define LLAMA_MP_H

#include <Uefi.h>

#define MM_MAX_WORKERS 64

/* Locate MP Services and launch the persistent AP worker pool.
 * Falls back to single-core (BSP only) when MP Services is unavailable,
 * only one processor is enabled, or no AP passes the online handshake.
 * Always returns EFI_SUCCESS (never fatal). */
EFI_STATUS MmInit(VOID);

/* Stop the AP worker pool and release its events (bounded wait; events of
 * stuck workers are leaked on purpose). Returning from the AP procedure
 * parks the cores in the firmware -- use between turns to idle the CPUs. */
VOID MmShutdown(VOID);

/* (Re)start the pool when it is stopped and MP is available. No-op when the
 * pool is already running, MP is unavailable/broken, or MmInit never
 * succeeded (nomp.txt, <=1 CPU). Safe to call on every turn. */
VOID MmEnsureStarted(VOID);

/* Number of workers actually in use right now (1 == BSP only). */
UINTN MmWorkerCount(VOID);

/* W (d,n) @ x (n,) -> xout (d,). Parallelized across the worker pool by
 * splitting the d output rows into contiguous ranges. */
VOID MatMul(float* xout, float* x, float* w, INT32 n, INT32 d);

/* int8 Q8_0 variant (runq.c format): W and x are group-quantized (int8
 * values + one fp32 scale per group of gs values). Same row-range
 * parallelization and watchdog behaviour as MatMul. n and the row stride
 * must be multiples of gs. */
VOID MatMulQ8(float* xout, CONST INT8* xq, CONST float* xs,
              CONST INT8* wq, CONST float* ws, INT32 n, INT32 d, INT32 gs);

#endif /* LLAMA_MP_H */
