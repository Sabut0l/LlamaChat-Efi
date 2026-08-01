/* =========================================================================
 * LlamaMp.c  --  Persistent MP Services worker pool for parallel matmul,
 * hardened for real hardware.
 *
 * History of the two failure modes this file fixes:
 *
 *  (1) v1 (persistent pool) froze hard on real hardware: APs never came
 *      online (typically because SSE/FPU state is NOT initialized on APs by
 *      every firmware: CR0.EM/CR0.TS set or CR4.OSFXSR clear -> the first
 *      float instruction raises #UD/#NM and the AP dead-loops in the
 *      firmware exception handler). The BSP then spun FOREVER in
 *      MatMul() waiting for Ack. There were no timeouts anywhere.
 *
 *  (2) v2 (per-token batch pool) was extremely slow / effectively
 *      single-threaded: it called StartupThisAP for every worker on EVERY
 *      token, and MmEndBatch() waited for the AP-finished event, which the
 *      firmware only signals from its periodic AP-status-check timer
 *      (~100 ms with default PcdCpuApStatusCheckIntervalInMicroSeconds).
 *      That added hundreds of ms of idle wait per token. It also had a
 *      startup race: ApPool() checked "i >= gBatchWorkers" while the BSP
 *      incremented gBatchWorkers only AFTER StartupThisAP returned, so a
 *      fast-starting AP could exit immediately.
 *
 * The fix combines both worlds:
 *   - persistent worker pool (v1): StartupThisAP is called ONCE per AP,
 *     matmul dispatch is a lock-free Request/Ack generation handshake;
 *   - explicit SSE enable on each AP (v2's PrepareFp) + an FP self-test
 *     before the worker reports itself online;
 *   - bounded waits everywhere: MmInit waits for the online handshake with
 *     a timeout and excludes workers that never started; MatMul has a
 *     watchdog -- if a worker stops acking, the BSP recomputes its slice
 *     itself, marks the worker dead and keeps going. The app degrades to
 *     fewer cores (worst case: BSP only) instead of freezing.
 *
 * AP tasks run at TPL_HIGH_LEVEL: memory and floats only -- no Boot
 * Services, no protocols. Debug output is raw port I/O under a spin lock.
 * ========================================================================= */
#include "LlamaMp.h"
#include "LlamaDebug.h"

#include <Library/UefiBootServicesTableLib.h>
#include <Library/UefiLib.h>
#include <Library/BaseLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/SynchronizationLib.h>
#include <Pi/PiMultiPhase.h>
#include <Protocol/MpService.h>

/* SSE2 есть у любого x64-CPU. Вычислительные ядра ниже получают SIMD-версии;
 * скалярные фолбэки оставлены для не-x86 сборок. SSE включается на каждом AP
 * явно (PrepareFp), так что воркерам XMM-регистры доступны. */
#if defined(MDE_CPU_X64) || defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#define MM_HAVE_SSE2 1
#else
#define MM_HAVE_SSE2 0
#endif

#if defined(_MSC_VER)
#define MM_NOINLINE __declspec(noinline)
#else
#define MM_NOINLINE __attribute__((noinline))
#endif

/* Watchdog limits. TSC-based; calibrated against gBS->Stall in MmInit. */
#define MM_ONLINE_TIMEOUT_MS   1000ULL   /* AP must report online within this  */
#define MM_ACK_TIMEOUT_MS      2000ULL   /* max wait for one matmul slice      */
#define MM_EXIT_TIMEOUT_MS     1000ULL   /* max wait for AP exit in shutdown   */

/* Escape-hatch watchdog for a StartupThisAP call that never returns
 * (buggy old firmware, e.g. Kabini-era Aptio 4: the firmware spins forever
 * trying to wake an AP it cannot actually start). 3 s in 100 ns units. */
#define MM_STARTUP_WDT_100NS   30000000ULL

/* One row-slice job handed to a worker. Request/Ack form a generation
 * handshake: BSP publishes params then bumps Request; the AP runs the slice
 * and sets Ack = Request. MemoryFence brackets the publish/consume.
 * Pad keeps neighbouring workers' hot flags on different cache lines. */
typedef struct {
    volatile INTN    Request;
    volatile INTN    Ack;
    volatile BOOLEAN Online;    /* set by AP after PrepareFp + FP self-test */
    volatile BOOLEAN Dead;      /* set by BSP: excluded from dispatch       */
    volatile float   FpProbe;   /* AP FP self-test result (must become 7.0) */
    float* xout;
    float* x;
    float* w;
    INT32  n;
    INT32  rowStart;
    INT32  rowEnd;
    /* int8 Q8_0 job variant: when Quant is TRUE the worker uses Xq/Xs/Wq/Ws/Gs
     * instead of x/w (quantized matmul, runq.c format). */
    volatile BOOLEAN Quant;
    CONST INT8*  Xq;
    CONST float* Xs;
    CONST INT8*  Wq;
    CONST float* Ws;
    INT32  Gs;
    UINT8  Pad[64];
} MM_WORKER;

STATIC EFI_MP_SERVICES_PROTOCOL* gMp = NULL;
STATIC UINTN            gNumWorkers = 1;      /* slots used incl. BSP slot 0 */
STATIC BOOLEAN          gUseAp = FALSE;
STATIC UINTN            gProcIds[MM_MAX_WORKERS];
STATIC EFI_EVENT        gApEvents[MM_MAX_WORKERS];
STATIC MM_WORKER        gWorkers[MM_MAX_WORKERS];
STATIC volatile BOOLEAN gMmShutdown = FALSE;
STATIC UINT64           gTscPerMs = 3000000;  /* recalibrated in MmInit */

/* StartupThisAP hang detection (see MmStartupWatchdog below). */
STATIC BASE_LIBRARY_JUMP_BUFFER gStartupJump;
STATIC volatile BOOLEAN         gStartupInFlight = FALSE;
STATIC BOOLEAN                  gMpBroken = FALSE;
STATIC BOOLEAN                  gMpAvailable = FALSE; /* хоть один MmInit успешен -> перезапуски разрешены */
STATIC BOOLEAN                  gMpQuiet     = FALSE; /* глушить консольный вывод при перезапусках пула */
#define MpPrint(...)  do { if (!gMpQuiet) Print(__VA_ARGS__); } while (0)

/* Timer callback armed around every StartupThisAP call. StartupThisAP is a
 * blocking firmware call: if the firmware itself hangs inside it (spinning
 * forever on an AP that never wakes up), no flag or TSC deadline on the BSP
 * can help -- the BSP never gets control back. But timer INTERRUPTS still
 * fire while the firmware spins at low TPL, so this callback runs, detects
 * that the call is still "in flight" after 3 s, and long-jumps out of the
 * stuck call back into MmInit. MP is then declared broken and the app
 * continues single-core instead of freezing. */
STATIC VOID EFIAPI MmStartupWatchdog(EFI_EVENT Event, VOID* Context) {
    if (gStartupInFlight) {
        gStartupInFlight = FALSE;
        LongJump(&gStartupJump, 1);   /* never returns */
    }
}

UINTN MmWorkerCount(VOID) {
    UINTN live = 1;
    for (UINTN i = 1; i < gNumWorkers; i++) {
        if (gWorkers[i].Online && !gWorkers[i].Dead) live++;
    }
    return live;
}

/* Compute output rows [rs, re) of W (d,n) @ x (n,). AP-safe (math only).
 * noinline: keeps MmApLoop's own prologue free of XMM spills so the AP
 * reaches PrepareFp() before executing any SSE instruction. */
#if MM_HAVE_SSE2
/* 4 float за такт, четыре независимых аккумулятора (развязка по latency),
 * хвост -- скалярно. Строки матрицы не выровнены -> loadu. */
STATIC MM_NOINLINE VOID MmComputeRange(float* xout, float* x, float* w, INT32 n, INT32 rs, INT32 re) {
    for (INT32 i = rs; i < re; i++) {
        float* wr = w + (UINTN)i * (UINTN)n;
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        __m128 acc2 = _mm_setzero_ps();
        __m128 acc3 = _mm_setzero_ps();
        INT32 j = 0;
        for (; j + 16 <= n; j += 16) {
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(wr + j),      _mm_loadu_ps(x + j)));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(wr + j + 4),  _mm_loadu_ps(x + j + 4)));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(wr + j + 8),  _mm_loadu_ps(x + j + 8)));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(wr + j + 12), _mm_loadu_ps(x + j + 12)));
        }
        for (; j + 4 <= n; j += 4) {
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(wr + j), _mm_loadu_ps(x + j)));
        }
        acc0 = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
        __m128 shuf = _mm_shuffle_ps(acc0, acc0, _MM_SHUFFLE(1, 0, 3, 2));
        __m128 sums = _mm_add_ps(acc0, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float val = _mm_cvtss_f32(sums);
        for (; j < n; j++) val += wr[j] * x[j];
        xout[i] = val;
    }
}
#else
STATIC MM_NOINLINE VOID MmComputeRange(float* xout, float* x, float* w, INT32 n, INT32 rs, INT32 re) {
    for (INT32 i = rs; i < re; i++) {
        float  val = 0.0f;
        float* wr  = w + (UINTN)i * (UINTN)n;
        for (INT32 j = 0; j < n; j++) {
            val += wr[j] * x[j];
        }
        xout[i] = val;
    }
}
#endif

/* int8 Q8_0 variant of MmComputeRange (port of runq.c matmul): dot products
 * accumulate in int32 inside each quantization group, then get scaled by the
 * two fp32 group scales. AP-safe (math only), noinline for the same reason
 * as MmComputeRange. */
#if MM_HAVE_SSE2
/* Q8_0 в int16-SIMD: int8 расширяется до int16 со знаком, произведения
 * складываются _mm_madd_epi16 попарно в int32. gs кратен 16 по контракту
 * формата (64) -> 16 значений за итерацию. Точность: |q|<=127, 16*127*127
 * много меньше 2^31. В 6-8 раз быстрее скалярного цикла. */
STATIC MM_NOINLINE VOID MmComputeRangeQ8(float* xout, CONST INT8* xq, CONST float* xs,
                                         CONST INT8* wq, CONST float* ws,
                                         INT32 n, INT32 gs, INT32 rs, INT32 re) {
    INT32 gs16 = gs / 16;
    for (INT32 i = rs; i < re; i++) {
        float val = 0.0f;
        UINTN in = (UINTN)i * (UINTN)n;
        for (INT32 j = 0; j <= n - gs; j += gs) {
            CONST INT8* xg = xq + j;
            CONST INT8* wg = wq + in + (UINTN)j;
            __m128i ivec = _mm_setzero_si128();
            for (INT32 k = 0; k < gs16; k++) {
                __m128i x8 = _mm_loadu_si128((CONST __m128i*)(xg + k * 16));
                __m128i w8 = _mm_loadu_si128((CONST __m128i*)(wg + k * 16));
                __m128i xlo = _mm_srai_epi16(_mm_unpacklo_epi8(x8, x8), 8);
                __m128i xhi = _mm_srai_epi16(_mm_unpackhi_epi8(x8, x8), 8);
                __m128i wlo = _mm_srai_epi16(_mm_unpacklo_epi8(w8, w8), 8);
                __m128i whi = _mm_srai_epi16(_mm_unpackhi_epi8(w8, w8), 8);
                ivec = _mm_add_epi32(ivec, _mm_madd_epi16(xlo, wlo));
                ivec = _mm_add_epi32(ivec, _mm_madd_epi16(xhi, whi));
            }
            __m128i sh64 = _mm_shuffle_epi32(ivec, _MM_SHUFFLE(1, 0, 3, 2));
            __m128i su64 = _mm_add_epi32(ivec, sh64);
            __m128i sh32 = _mm_shuffle_epi32(su64, _MM_SHUFFLE(2, 3, 0, 1));
            __m128i su32 = _mm_add_epi32(su64, sh32);
            INT32 ival = _mm_cvtsi128_si32(su32);
            val += (float)ival * ws[(in + (UINTN)j) / (UINTN)gs] * xs[j / gs];
        }
        xout[i] = val;
    }
}
#else
STATIC MM_NOINLINE VOID MmComputeRangeQ8(float* xout, CONST INT8* xq, CONST float* xs,
                                         CONST INT8* wq, CONST float* ws,
                                         INT32 n, INT32 gs, INT32 rs, INT32 re) {
    for (INT32 i = rs; i < re; i++) {
        float val = 0.0f;
        UINTN in = (UINTN)i * (UINTN)n;
        for (INT32 j = 0; j <= n - gs; j += gs) {
            INT32 ival = 0;
            CONST INT8* xg = xq + j;
            CONST INT8* wg = wq + in + (UINTN)j;
            for (INT32 k = 0; k < gs; k++) {
                ival += (INT32)xg[k] * (INT32)wg[k];
            }
            val += (float)ival * ws[(in + (UINTN)j) / (UINTN)gs] * xs[j / gs];
        }
        xout[i] = val;
    }
}
#endif

/* Enable SSE on the calling processor. Firmware is NOT required to hand APs
 * over with usable FP state (this is exactly what killed v1 on real HW):
 *   CR0.MP=1, CR0.EM=0, CR0.TS=0; CR4.OSFXSR=1, CR4.OSXMMEXCPT=1.
 * Integer-only code: must not itself touch XMM registers. */
STATIC VOID PrepareFp(VOID) {
#if defined(MDE_CPU_X64) || defined(MDE_CPU_IA32)
    UINTN c0 = AsmReadCr0();
    c0 |= BIT1;                              /* MP */
    c0 &= ~((UINTN)BIT2 | (UINTN)BIT3);      /* clear EM, TS */
    AsmWriteCr0(c0);
    UINTN c4 = AsmReadCr4();
    c4 |= (UINTN)BIT9 | (UINTN)BIT10;        /* OSFXSR, OSXMMEXCPT */
    AsmWriteCr4(c4);
    MemoryFence();
#endif
}

/* AP entry -- NO UEFI services, only port I/O for debug (TPL_HIGH_LEVEL).
 * Persistent: loops for the lifetime of the pool. The slot index is fixed
 * at StartupThisAP time, so there is no counter race (v2 bug). */
STATIC VOID EFIAPI MmApLoop(VOID* Arg) {
    UINTN idx = (UINTN)Arg;
    if (idx == 0 || idx >= MM_MAX_WORKERS) return;
    MM_WORKER* w = &gWorkers[idx];

    PrepareFp();

    /* FP self-test: proves SSE actually works on this AP before we accept
     * jobs. If this faults, the AP never goes online and MmInit's bounded
     * wait excludes it instead of the app hanging later in MatMul. */
    w->FpProbe = 2.0f;
    w->FpProbe = w->FpProbe * 3.0f + 1.0f;   /* -> 7.0 exactly */
    MemoryFence();
    w->Online = TRUE;

    DBG_AP(idx, "worker online");

    for (;;) {
        while (w->Ack == w->Request) {
            if (gMmShutdown) {
                DBG_AP(idx, "worker shutdown");
                return;
            }
            CpuPause();
        }
        MemoryFence();                 /* consume freshly published params */
        if (w->Quant) {
            MmComputeRangeQ8(w->xout, w->Xq, w->Xs, w->Wq, w->Ws, w->n, w->Gs, w->rowStart, w->rowEnd);
        } else {
            MmComputeRange(w->xout, w->x, w->w, w->n, w->rowStart, w->rowEnd);
        }
        MemoryFence();                 /* publish results before acking     */
        w->Ack = w->Request;
    }
}

EFI_STATUS MmInit(VOID) {
    DBG("MmInit: start");
    gUseAp = FALSE;
    gNumWorkers = 1;
    gMmShutdown = FALSE;
    ZeroMem((VOID*)gWorkers, sizeof(gWorkers));
    ZeroMem(gApEvents, sizeof(gApEvents));
    gProcIds[0] = 0;

    /* Calibrate TSC so the watchdog timeouts below mean real milliseconds. */
    UINT64 t0 = AsmReadTsc();
    gBS->Stall(20 * 1000);                       /* 20 ms */
    UINT64 ticks = AsmReadTsc() - t0;
    if (ticks > 20) gTscPerMs = ticks / 20;
    DBG_DEC("MmInit: TSC ticks per ms", gTscPerMs);

    /* Console progress on every firmware call: these calls are blocking and
     * CANNOT be timed out from a single-core app. If the app freezes on
     * real hardware, the last printed line names the guilty call. */
    MpPrint(L"  MP: locating MP Services protocol\r\n");
    EFI_STATUS s = gBS->LocateProtocol(&gEfiMpServiceProtocolGuid, NULL, (VOID**)&gMp);
    if (EFI_ERROR(s) || gMp == NULL) {
        MpPrint(L"  MP: not available -- single core\r\n");
        DBG("MmInit: MP Services unavailable -- single core");
        return EFI_SUCCESS;
    }

    MpPrint(L"  MP: querying processor count\r\n");
    UINTN total = 0, enabled = 0;
    s = gMp->GetNumberOfProcessors(gMp, &total, &enabled);
    if (EFI_ERROR(s) || enabled <= 1) {
        MpPrint(L"  MP: <=1 enabled processor -- single core\r\n");
        DBG("MmInit: <=1 enabled processor -- single core");
        return EFI_SUCCESS;
    }
    MpPrint(L"  MP: %d total / %d enabled processors\r\n", (UINT32)total, (UINT32)enabled);
    DBG_DEC("MmInit: total processors", total);
    DBG_DEC("MmInit: enabled processors", enabled);

    UINTN bsp = 0;
    s = gMp->WhoAmI(gMp, &bsp);
    if (EFI_ERROR(s)) {
        DBG_HEX("MmInit: WhoAmI FAILED", s);
        return EFI_SUCCESS;
    }
    DBG_DEC("MmInit: BSP index", bsp);
    gProcIds[0] = bsp;

    /* Launch every healthy enabled AP into its persistent loop
     * (non-blocking: WaitEvent given, infinite timeout). */
    for (UINTN proc = 0; proc < total && gNumWorkers < MM_MAX_WORKERS; proc++) {
        if (proc == bsp) continue;
        EFI_PROCESSOR_INFORMATION info;
        MpPrint(L"  MP: querying cpu %d info\r\n", (UINT32)proc);
        if (EFI_ERROR(gMp->GetProcessorInfo(gMp, proc, &info))) continue;
        if ((info.StatusFlag & PROCESSOR_ENABLED_BIT) == 0) continue;
        if ((info.StatusFlag & PROCESSOR_HEALTH_STATUS_BIT) == 0) continue;

        UINTN slot = gNumWorkers;
        EFI_EVENT ev = NULL;
        s = gBS->CreateEvent(0, TPL_CALLBACK, NULL, NULL, &ev);
        if (EFI_ERROR(s)) {
            DBG_HEX("MmInit: CreateEvent FAILED", s);
            continue;
        }

        /* Arm the anti-hang watchdog around the blocking firmware call. */
        EFI_EVENT wdt = NULL;
        if (EFI_ERROR(gBS->CreateEvent(EVT_TIMER | EVT_NOTIFY_SIGNAL, TPL_NOTIFY,
                                       MmStartupWatchdog, NULL, &wdt))) {
            wdt = NULL;   /* no watchdog possible; call unprotected */
        }

        if (SetJump(&gStartupJump) != 0) {
            /* We long-jumped out of a StartupThisAP call that hung inside
             * the firmware. We arrive here at TPL_NOTIFY (from the timer
             * callback); drop back to application level first. */
            gBS->RestoreTPL(TPL_APPLICATION);
            if (wdt != NULL) {
                gBS->SetTimer(wdt, TimerCancel, 0);
                gBS->CloseEvent(wdt);
            }
            /* Deliberately LEAK ev: the firmware's half-executed MP state
             * may still reference it. */
            gMpBroken = TRUE;
            Print(L"  MP: StartupThisAP HUNG for cpu %d (firmware bug) -- multicore OFF\r\n", (UINT32)proc);
            DBG_DEC("MmInit: StartupThisAP HANG detected on cpu", proc);
            break;   /* trust nothing: launch no further workers */
        }

        MpPrint(L"  MP: starting worker on cpu %d\r\n", (UINT32)proc);
        if (wdt != NULL) {
            gBS->SetTimer(wdt, TimerRelative, MM_STARTUP_WDT_100NS);
        }
        gStartupInFlight = TRUE;
        s = gMp->StartupThisAP(gMp, MmApLoop, proc, ev, 0, (VOID*)slot, NULL);
        gStartupInFlight = FALSE;
        if (wdt != NULL) {
            gBS->SetTimer(wdt, TimerCancel, 0);
            gBS->CloseEvent(wdt);
        }

        if (EFI_ERROR(s)) {
            MpPrint(L"  MP: StartupThisAP failed (0x%x) -- cpu %d skipped\r\n", (UINT32)s, (UINT32)proc);
            DBG_HEX("MmInit: StartupThisAP FAILED", s);
            gBS->CloseEvent(ev);
            continue;
        }
        gApEvents[slot] = ev;
        gProcIds[slot]  = proc;
        gNumWorkers++;
    }
    MpPrint(L"  MP: waiting for workers to come online\r\n");

    /* KEY FIX vs v1: wait for the online handshake with a BOUNDED timeout.
     * Workers that never come online (AP didn't start, or SSE faulted) are
     * excluded here instead of hanging the first MatMul forever. */
    UINT64 deadline = AsmReadTsc() + MM_ONLINE_TIMEOUT_MS * gTscPerMs;
    for (;;) {
        UINTN online = 0;
        for (UINTN i = 1; i < gNumWorkers; i++) {
            if (gWorkers[i].Online) online++;
        }
        if (online == gNumWorkers - 1) break;
        if (AsmReadTsc() > deadline) break;
        CpuPause();
    }

    UINTN live = 0;
    for (UINTN i = 1; i < gNumWorkers; i++) {
        if (!gWorkers[i].Online) {
            gWorkers[i].Dead = TRUE;
            DBG_DEC("MmInit: worker never came online -- EXCLUDED, slot", i);
        } else if (gWorkers[i].FpProbe != 7.0f) {
            gWorkers[i].Dead = TRUE;
            DBG_DEC("MmInit: worker FP self-test FAILED -- EXCLUDED, slot", i);
        } else {
            live++;
        }
    }

    gUseAp = (live > 0);
    gMpAvailable = gUseAp;
    MpPrint(L"  MP: %d AP worker(s) online\r\n", (UINT32)live);
    gMpQuiet = TRUE;   /* перезапуски пула больше не пишут в консоль */
    DBG_DEC("MmInit: live AP workers", live);
    DBG_DEC("MmInit: gUseAp", (UINTN)gUseAp);
    return EFI_SUCCESS;
}

VOID MmShutdown(VOID) {
    if (gNumWorkers <= 1) return;
    DBG("MmShutdown: stopping worker pool");
    gMmShutdown = TRUE;
    MemoryFence();

    if (gMpBroken) {
        /* We long-jumped out of a hung StartupThisAP earlier: the firmware
         * MP driver state is unreliable. Tell workers to exit via the flag
         * above, but do NOT touch any MP events -- leak them all. */
        DBG("MmShutdown: MP broken -- leaking all AP events");
        for (UINTN i = 1; i < gNumWorkers; i++) gApEvents[i] = NULL;
        gUseAp = FALSE;
        gNumWorkers = 1;
        gMpAvailable = FALSE;   /* сломанный MP не перезапускаем */
        return;
    }

    UINT64 deadline = AsmReadTsc() + MM_EXIT_TIMEOUT_MS * gTscPerMs;
    for (UINTN i = 1; i < gNumWorkers; i++) {
        if (gApEvents[i] == NULL) continue;
        /* Spin at TPL_APPLICATION so the firmware's AP-status timer can run
         * and signal the finished event -- but never wait forever. */
        BOOLEAN finished = FALSE;
        while (AsmReadTsc() <= deadline) {
            if (gBS->CheckEvent(gApEvents[i]) != EFI_NOT_READY) {
                finished = TRUE;
                break;
            }
            CpuPause();
        }
        if (finished) {
            gBS->CloseEvent(gApEvents[i]);
        } else {
            /* Stuck/dead AP: deliberately LEAK the event. Closing it could
             * make firmware later signal freed memory. One-shot at exit. */
            DBG_DEC("MmShutdown: worker did not exit -- leaking event, slot", i);
        }
        gApEvents[i] = NULL;
    }

    gUseAp = FALSE;
    gNumWorkers = 1;
    DBG("MmShutdown: done");
}

/* Разбудить пул AP перед тяжёлым участком (генерация ответа). No-op, если
 * пул жив, MP недоступен/сломан или ни один MmInit не был успешен. */
VOID MmEnsureStarted(VOID) {
    if (gMpAvailable && !gUseAp && !gMpBroken) {
        MmInit();
    }
}

VOID MatMul(float* xout, float* x, float* w, INT32 n, INT32 d) {
    UINTN liveIdx[MM_MAX_WORKERS];
    INT32 nLive = 0;

    if (gUseAp) {
        for (UINTN i = 1; i < gNumWorkers; i++) {
            if (gWorkers[i].Online && !gWorkers[i].Dead) {
                liveIdx[nLive++] = i;
            }
        }
    }

    INT32 nw = nLive + 1;   /* + BSP */
    if (nLive == 0 || d < nw) {
        /* single-core path (also used for tiny d where splitting is pointless) */
        MmComputeRange(xout, x, w, n, 0, d);
        return;
    }

    INT32 base = d / nw;
    INT32 rem  = d % nw;

    /* Publish params + dispatch live AP workers. BSP is share index 0. */
    INT32 bspEnd = base + (rem > 0 ? 1 : 0);
    INT32 cursor = bspEnd;
    for (INT32 k = 0; k < nLive; k++) {
        INT32 cnt = base + ((k + 1) < rem ? 1 : 0);
        MM_WORKER* wk = &gWorkers[liveIdx[k]];
        wk->Quant = FALSE;
        wk->xout = xout; wk->x = x; wk->w = w; wk->n = n;
        wk->rowStart = cursor;
        wk->rowEnd   = cursor + cnt;
        cursor += cnt;
        MemoryFence();      /* publish params before signalling */
        wk->Request++;      /* wake the AP */
    }

    /* BSP executes its own slice while the APs run. */
    MmComputeRange(xout, x, w, n, 0, bspEnd);

    /* KEY FIX vs v1: wait for acks with a WATCHDOG. If a worker dies
     * mid-flight (machine-specific AP failure), the BSP recomputes that
     * slice itself and permanently retires the worker -- the app slows
     * down instead of freezing. A retired AP never writes again: it is
     * either dead-looping in the firmware exception handler or was never
     * running at all, so late buffer corruption is not a concern. */
    UINT64 deadline = AsmReadTsc() + MM_ACK_TIMEOUT_MS * gTscPerMs;
    BOOLEAN anyDied = FALSE;
    for (INT32 k = 0; k < nLive; k++) {
        MM_WORKER* wk = &gWorkers[liveIdx[k]];
        while (wk->Ack != wk->Request) {
            if (AsmReadTsc() > deadline) {
                wk->Dead = TRUE;
                anyDied  = TRUE;
                DBG_DEC("MatMul: WATCHDOG -- worker retired, slot", liveIdx[k]);
                MmComputeRange(xout, x, w, n, wk->rowStart, wk->rowEnd);
                break;
            }
            CpuPause();
        }
    }
    MemoryFence();

    if (anyDied) {
        UINTN still = 0;
        for (UINTN i = 1; i < gNumWorkers; i++) {
            if (gWorkers[i].Online && !gWorkers[i].Dead) still++;
        }
        if (still == 0) {
            gUseAp = FALSE;
            DBG("MatMul: all AP workers retired -- single core from now on");
        }
    }
}

/* int8 Q8_0 matmul across the worker pool. Mirrors MatMul: same row split,
 * same bounded-wait watchdog with BSP recompute of a dead worker's slice. */
VOID MatMulQ8(float* xout, CONST INT8* xq, CONST float* xs,
              CONST INT8* wq, CONST float* ws, INT32 n, INT32 d, INT32 gs) {
    UINTN liveIdx[MM_MAX_WORKERS];
    INT32 nLive = 0;

    if (gUseAp) {
        for (UINTN i = 1; i < gNumWorkers; i++) {
            if (gWorkers[i].Online && !gWorkers[i].Dead) {
                liveIdx[nLive++] = i;
            }
        }
    }

    INT32 nw = nLive + 1;   /* + BSP */
    if (nLive == 0 || d < nw) {
        /* single-core path (also used for tiny d where splitting is pointless) */
        MmComputeRangeQ8(xout, xq, xs, wq, ws, n, gs, 0, d);
        return;
    }

    INT32 base = d / nw;
    INT32 rem  = d % nw;

    /* Publish params + dispatch live AP workers. BSP is share index 0. */
    INT32 bspEnd = base + (rem > 0 ? 1 : 0);
    INT32 cursor = bspEnd;
    for (INT32 k = 0; k < nLive; k++) {
        INT32 cnt = base + ((k + 1) < rem ? 1 : 0);
        MM_WORKER* wk = &gWorkers[liveIdx[k]];
        wk->Quant = TRUE;
        wk->xout = xout; wk->n = n;
        wk->Xq = xq; wk->Xs = xs; wk->Wq = wq; wk->Ws = ws; wk->Gs = gs;
        wk->rowStart = cursor;
        wk->rowEnd   = cursor + cnt;
        cursor += cnt;
        MemoryFence();      /* publish params before signalling */
        wk->Request++;      /* wake the AP */
    }

    /* BSP executes its own slice while the APs run. */
    MmComputeRangeQ8(xout, xq, xs, wq, ws, n, gs, 0, bspEnd);

    /* Watchdog: recompute a dead worker's slice on the BSP and retire it
     * (see MatMul above for the full rationale). */
    UINT64 deadline = AsmReadTsc() + MM_ACK_TIMEOUT_MS * gTscPerMs;
    BOOLEAN anyDied = FALSE;
    for (INT32 k = 0; k < nLive; k++) {
        MM_WORKER* wk = &gWorkers[liveIdx[k]];
        while (wk->Ack != wk->Request) {
            if (AsmReadTsc() > deadline) {
                wk->Dead = TRUE;
                anyDied  = TRUE;
                DBG_DEC("MatMulQ8: WATCHDOG -- worker retired, slot", liveIdx[k]);
                MmComputeRangeQ8(xout, xq, xs, wq, ws, n, gs, wk->rowStart, wk->rowEnd);
                break;
            }
            CpuPause();
        }
    }
    MemoryFence();

    if (anyDied) {
        UINTN still = 0;
        for (UINTN i = 1; i < gNumWorkers; i++) {
            if (gWorkers[i].Online && !gWorkers[i].Dead) still++;
        }
        if (still == 0) {
            gUseAp = FALSE;
            DBG("MatMulQ8: all AP workers retired -- single core from now on");
        }
    }
}
