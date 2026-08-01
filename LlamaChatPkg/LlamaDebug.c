/* =========================================================================
 * LlamaDebug.c  --  Debugcon (port 0x402) and file debug output.
 * Ported from the Cryptor project (1-Cryptor.c debug block).
 * ========================================================================= */
#include "LlamaDebug.h"

#include <Guid/FileInfo.h>
#include <Library/BaseMemoryLib.h>
#include <Library/MemoryAllocationLib.h>

STATIC SPIN_LOCK gDbgLock;
STATIC BOOLEAN   gDbgLockReady = FALSE;
STATIC UINT64    gStartTsc = 0;
STATIC EFI_FILE_PROTOCOL* gDbgFile = NULL;
STATIC CHAR8     gDbgFileBuffer[512];
STATIC UINTN     gDbgFileUsed = 0;

STATIC VOID DbgFlushFile(VOID) {
    if (gDbgFile == NULL || gDbgFileUsed == 0) return;
    UINTN Size = gDbgFileUsed;
    EFI_STATUS Status = gDbgFile->Write(gDbgFile, &Size, gDbgFileBuffer);
    if (!EFI_ERROR(Status)) gDbgFile->Flush(gDbgFile);
    gDbgFileUsed = 0;
}

VOID DbgLock(VOID)   { if (gDbgLockReady) AcquireSpinLock(&gDbgLock); }
VOID DbgUnlock(VOID) { if (gDbgLockReady) ReleaseSpinLock(&gDbgLock); }

VOID DbgPortChar(CHAR8 C) {
    IoWrite8(0x402, (UINT8)C);
    if (gDbgFile != NULL) {
        gDbgFileBuffer[gDbgFileUsed++] = C;
        if (C == '\n' || gDbgFileUsed == sizeof(gDbgFileBuffer)) {
            DbgFlushFile();
        }
    }
}

VOID DbgStr(CONST CHAR8* S) {
    while (S && *S) DbgPortChar(*S++);
}

/* Print a 64-bit value in hex (0x...) via the debug port. */
VOID DbgHex64(UINT64 V) {
    CHAR8 Buf[19];
    CHAR8 Hex[] = "0123456789ABCDEF";
    Buf[0] = '0';
    Buf[1] = 'x';
    for (INT32 i = 17; i >= 2; i--) {   /* 16 nibbles: [17..2] */
        Buf[i] = Hex[V & 0xF];
        V >>= 4;
    }
    Buf[18] = '\0';
    DbgStr(Buf);
}

/* Print an unsigned decimal value via the debug port. */
VOID DbgDec(UINT64 V) {
    CHAR8 Buf[22];
    UINTN Pos = 20;
    Buf[21] = '\0';
    if (V == 0) { DbgPortChar('0'); return; }
    while (V > 0 && Pos > 0) {
        Buf[Pos--] = (CHAR8)('0' + (V % 10));
        V /= 10;
    }
    DbgStr(&Buf[Pos + 1]);
}

/* Rough elapsed-time prefix [<ms>], assuming a ~1 GHz invariant TSC (QEMU). */
VOID DbgPrintTimestamp(VOID) {
    if (gStartTsc == 0) return;
    UINT64 CurrentTsc = AsmReadTsc();
    UINT64 ElapsedTsc = CurrentTsc - gStartTsc;
    UINT32 ElapsedMs = (UINT32)(ElapsedTsc / 1000000ULL);
    DbgPortChar('[');
    DbgDec(ElapsedMs);
    DbgStr(" ms] ");
}

VOID DbgInit(VOID) {
    InitializeSpinLock(&gDbgLock);
    gDbgLockReady = TRUE;
    gStartTsc = AsmReadTsc();
}

EFI_STATUS DbgInitFile(EFI_FILE_PROTOCOL* Root, CONST CHAR16* Name) {
    if (Root == NULL || Name == NULL) return EFI_INVALID_PARAMETER;
    if (gDbgFile != NULL) DbgCloseFile();

    EFI_STATUS Status = Root->Open(
        Root, &gDbgFile, (CHAR16*)Name,
        EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE, 0);
    if (EFI_ERROR(Status)) {
        gDbgFile = NULL;
        return Status;
    }

    /* Truncate an older log so stale lines cannot be mistaken for this run. */
    UINTN InfoSize = 0;
    Status = gDbgFile->GetInfo(gDbgFile, &gEfiFileInfoGuid, &InfoSize, NULL);
    if (Status == EFI_BUFFER_TOO_SMALL) {
        EFI_FILE_INFO* Info = AllocatePool(InfoSize);
        if (Info != NULL) {
            Status = gDbgFile->GetInfo(gDbgFile, &gEfiFileInfoGuid, &InfoSize, Info);
            if (!EFI_ERROR(Status)) {
                Info->FileSize = 0;
                Status = gDbgFile->SetInfo(gDbgFile, &gEfiFileInfoGuid, InfoSize, Info);
            }
            FreePool(Info);
        }
    }
    gDbgFile->SetPosition(gDbgFile, 0);
    gDbgFileUsed = 0;
    return EFI_SUCCESS;
}

VOID DbgCloseFile(VOID) {
    if (gDbgFile == NULL) return;
    DbgFlushFile();
    gDbgFile->Close(gDbgFile);
    gDbgFile = NULL;
    gDbgFileUsed = 0;
}
