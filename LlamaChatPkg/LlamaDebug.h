/* =========================================================================
 * LlamaDebug.h  --  Debugcon plus optional on-disk debug output.
 *
 * Ported directly from the Cryptor project's MP-safe serial debug facility.
 * APs print at TPL_HIGH_LEVEL concurrently with the BSP; without a spin lock
 * their characters would interleave on port 0x402. Each full log line is made
 * atomic by taking gDbgLock (both BSP and AP take it).
 *
 * QEMU: -debugcon stdio -global isa-debugcon.iobase=0x402
 * ========================================================================= */
#ifndef LLAMA_DEBUG_H
#define LLAMA_DEBUG_H

#include <Uefi.h>
#include <Library/BaseLib.h>
#include <Library/IoLib.h>
#include <Library/SynchronizationLib.h>
#include <Protocol/SimpleFileSystem.h>

/* One-time initialization: arms the spin lock and captures the start TSC so
 * that DbgPrintTimestamp() can emit a rough [<ms>] prefix. Call once on the
 * BSP before any AP is dispatched. */
VOID  DbgInit(VOID);

/* Enable a second debug sink on the boot volume. The file is flushed after
 * every complete line, so the last line should survive a hard hang/reset.
 * File output is BSP-only: close it before dispatching AP workers. */
EFI_STATUS DbgInitFile(EFI_FILE_PROTOCOL* Root, CONST CHAR16* Name);
VOID       DbgCloseFile(VOID);

VOID  DbgLock(VOID);
VOID  DbgUnlock(VOID);
VOID  DbgPortChar(CHAR8 C);
VOID  DbgStr(CONST CHAR8* S);
VOID  DbgHex64(UINT64 V);
VOID  DbgDec(UINT64 V);
VOID  DbgPrintTimestamp(VOID);

/* Convenience macros -- safe from AP context (port I/O only, no UEFI calls).
 * Each macro takes the lock so an entire line is emitted atomically. */
#define DBG(s)          do { DbgLock(); DbgPrintTimestamp(); DbgStr("[DBG] " s "\n"); DbgUnlock(); } while(0)
#define DBGF(s)         DbgStr(s)   /* raw fragment, caller already holds lock */
#define DBG_HEX(lbl,v)  do { DbgLock(); DbgPrintTimestamp(); DbgStr("[DBG] " lbl ": "); DbgHex64((UINT64)(v)); DbgPortChar('\n'); DbgUnlock(); } while(0)
#define DBG_DEC(lbl,v)  do { DbgLock(); DbgPrintTimestamp(); DbgStr("[DBG] " lbl ": "); DbgDec((UINT64)(v)); DbgPortChar('\n'); DbgUnlock(); } while(0)
#define DBG_AP(idx,s)   do { DbgLock(); DbgPrintTimestamp(); DbgStr("[AP"); DbgDec(idx); DbgStr("] " s "\n"); DbgUnlock(); } while(0)

#endif /* LLAMA_DEBUG_H */
