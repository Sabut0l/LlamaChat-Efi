//Copyright (c) 2026 Sabut0l
/* =========================================================================
 * LlamaCmd.c  --  Перехват <CMD>-команд модели и определение конфига ПК.
 * См. LlamaCmd.h. Алгоритм фильтра проверен юнит-тестом на любых разбивках
 * ответа на токены (маркер может прийти по одному байту).
 * ========================================================================= */
#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/DevicePathLib.h>
#include <Library/BaseLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/PrintLib.h>
#include <Pi/PiMultiPhase.h>
#include <Protocol/MpService.h>
#include <Protocol/PciIo.h>
#include <Protocol/BlockIo.h>
#include <Guid/FileInfo.h>

#include "LlamaCmd.h"
#include "LlamaRuntime.h"
#include "LlamaUi.h"
#include "LlamaDebug.h"

STATIC EFI_FILE_PROTOCOL* gCmdRoot = NULL;
STATIC EFI_HANDLE         gCmdDevice = NULL;
STATIC BOOLEAN            gLaunchReq = FALSE;

VOID CmdInit(EFI_FILE_PROTOCOL* Root, EFI_HANDLE DeviceHandle) {
    gCmdRoot = Root;
    gCmdDevice = DeviceHandle;
}

/* ------------------------------------------------------------------------
 * Вывод на консоль (как ChatOut/PrintPiece в LlamaChat.c): UI-путь умеет
 * UTF-8/кириллицу, чистая консоль -- только печатные ASCII + перевод строки.
 * ------------------------------------------------------------------------ */
STATIC VOID CmdConsole(CONST CHAR8* Text) {
    if (Text == NULL || Text[0] == '\0') return;
    if (UiIsReady()) { UiWriteUtf8(Text); return; }
    UINTN i;
    for (i = 0; Text[i] != '\0'; i++) {
        UINT8 b = (UINT8)Text[i];
        if (b == '\n') { AsciiPrint("\r\n"); continue; }
        if ((b >= 0x20 && b < 0x7F) || b == '\t') {
            CHAR8 Tmp[2];
            Tmp[0] = (CHAR8)b; Tmp[1] = '\0';
            AsciiPrint("%a", Tmp);
        }
        /* байты UTF-8 >= 0x80 в чистой консоли не отобразить -- пропускаем */
    }
}

/* ------------------------------------------------------------------------
 * Потоковый фильтр маркеров <CMD>имя</CMD>
 * ------------------------------------------------------------------------ */
#define CMD_OPEN      "<CMD>"
#define CMD_OPEN_LEN  5
#define CMD_CLOSE     "</CMD>"
#define CMD_CLOSE_LEN 6
#define CMD_MAX_SPAN  40   /* «<CMD>» без «</CMD>» длиннее этого -- не маркер */

VOID CmdFilterInit(CmdFilter* Filter, CHAR8* AnswerBuf, UINTN AnswerCap) {
    Filter->HoldLen    = 0;
    Filter->Cmd[0]     = '\0';
    Filter->HaveCmd    = 0;
    Filter->Answer     = AnswerBuf;
    Filter->AnswerCap  = AnswerCap;
    Filter->AnswerLen  = 0;
    Filter->PrintedLen = 0;
    if (AnswerCap > 0) AnswerBuf[0] = '\0';
}

STATIC VOID CmdEmitByte(CmdFilter* f, CHAR8 b) {
    if (f->AnswerLen + 1 < f->AnswerCap) {
        f->Answer[f->AnswerLen++] = b;
        f->Answer[f->AnswerLen]   = '\0';
    }
}

/* 1 = префикс присутствует целиком, 0 = не совпал, 2 = совпал частично (данные кончились) */
STATIC INTN CmdStartsWith(CONST CHAR8* s, UINTN slen, CONST CHAR8* pfx) {
    UINTN i;
    for (i = 0; pfx[i] != '\0'; i++) {
        if (i >= slen) return 2;
        if (s[i] != pfx[i]) return 0;
    }
    return 1;
}

STATIC VOID CmdProcess(CmdFilter* f) {
    UINTN i = 0;
    while (i < f->HoldLen) {
        if (f->Hold[i] != '<') { CmdEmitByte(f, f->Hold[i]); i++; continue; }
        INTN r = CmdStartsWith(f->Hold + i, f->HoldLen - i, CMD_OPEN);
        if (r == 0) { CmdEmitByte(f, '<'); i++; continue; }
        if (r == 2) break;                 /* частичный «<CMD» -- ждём продолжения */
        /* есть «<CMD>»: ищем закрывающий «</CMD>» */
        UINTN j = i + CMD_OPEN_LEN;
        INTN  found = -1;
        while (j + CMD_CLOSE_LEN <= f->HoldLen) {
            if (CompareMem(f->Hold + j, CMD_CLOSE, CMD_CLOSE_LEN) == 0) { found = (INTN)j; break; }
            j++;
        }
        if (found < 0) {
            if (f->HoldLen - i > CMD_MAX_SPAN) { CmdEmitByte(f, '<'); i++; continue; }
            break;                          /* закрывающий тег ещё может прийти */
        }
        UINTN nlen = (UINTN)found - (i + CMD_OPEN_LEN);
        if (nlen > 0 && nlen < CMD_NAME_CAP) {
            CopyMem(f->Cmd, f->Hold + i + CMD_OPEN_LEN, nlen);
            f->Cmd[nlen] = '\0';
            f->HaveCmd = 1;
        }
        i = (UINTN)found + CMD_CLOSE_LEN;   /* маркер целиком пропускается */
    }
    if (i > 0) {
        CopyMem(f->Hold, f->Hold + i, f->HoldLen - i);
        f->HoldLen -= i;
    }
}

/* Допечатать на консоль ту часть Answer, которая ещё не выводилась. */
STATIC VOID CmdFlushConsole(CmdFilter* f) {
    if (f->AnswerLen > f->PrintedLen) {
        CmdConsole(f->Answer + f->PrintedLen);   /* суффикс NUL-terminated */
        f->PrintedLen = f->AnswerLen;
    }
}

VOID CmdFilterFeed(CmdFilter* Filter, CONST CHAR8* Piece) {
    if (Piece == NULL) return;
    UINTN k;
    for (k = 0; Piece[k] != '\0'; k++) {
        if (Filter->HoldLen + 1 >= CMD_HOLD_CAP) {
            /* защита от переполнения: сбрасываем накопленное как обычный текст */
            UINTN x;
            for (x = 0; x < Filter->HoldLen; x++) CmdEmitByte(Filter, Filter->Hold[x]);
            Filter->HoldLen = 0;
        }
        Filter->Hold[Filter->HoldLen++] = Piece[k];
        CmdProcess(Filter);
    }
    CmdFlushConsole(Filter);
}

VOID CmdFilterFlush(CmdFilter* Filter) {
    UINTN x;
    for (x = 0; x < Filter->HoldLen; x++) CmdEmitByte(Filter, Filter->Hold[x]);
    Filter->HoldLen = 0;
    CmdFlushConsole(Filter);
}

/* ------------------------------------------------------------------------
 * Пароль и секретный файл
 * ------------------------------------------------------------------------ */
BOOLEAN CmdCheckPassword(CONST CHAR8* Input) {
    return (BOOLEAN)(AsciiStrCmp(Input, CHAT_PASSWORD) == 0);
}

VOID CmdOpenSecretFile(VOID) {
    if (gCmdRoot == NULL) {
        CmdConsole("[secret.txt: загрузочный том недоступен]\r\n");
        return;
    }
    EFI_FILE_PROTOCOL* File = NULL;
    EFI_STATUS s = gCmdRoot->Open(gCmdRoot, &File, SECRET_FILE, EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(s)) {
        CmdConsole("[secret.txt не найден: положите его рядом с LlamaChat.efi]\r\n");
        return;
    }
    UINTN InfoSize = 0;
    EFI_FILE_INFO* Info = NULL;
    s = File->GetInfo(File, &gEfiFileInfoGuid, &InfoSize, NULL);
    if (s != EFI_BUFFER_TOO_SMALL) { File->Close(File); return; }
    Info = (EFI_FILE_INFO*)LmAlloc(InfoSize);
    if (Info == NULL) { File->Close(File); return; }
    s = File->GetInfo(File, &gEfiFileInfoGuid, &InfoSize, Info);
    if (EFI_ERROR(s)) { LmFree(Info); File->Close(File); return; }
    UINTN Size = (UINTN)Info->FileSize;
    LmFree(Info);

    CHAR8* Buf = (CHAR8*)LmAlloc(Size + 1);
    if (Buf == NULL) { File->Close(File); return; }
    UINTN Total = 0;
    while (Total < Size) {
        UINTN Chunk = Size - Total;
        s = File->Read(File, &Chunk, (UINT8*)Buf + Total);
        if (EFI_ERROR(s) || Chunk == 0) break;
        Total += Chunk;
    }
    Buf[Total] = '\0';
    File->Close(File);

    CmdConsole("----- secret.txt -----\r\n");
    CmdConsole(Buf);
    CmdConsole("\r\n----------------------\r\n");
    LmFree(Buf);
    DBG("CmdOpenSecretFile: printed");
}

/* ------------------------------------------------------------------------
 * Запуск другого EFI-приложения после верного пароля (<CMD>openfile</CMD>).
 * Вызывается из UefiMain ПОСЛЕ teardown: модель и KV-кэш освобождены,
 * MP-воркеры остановлены -- дочернему приложению доступна вся память.
 * Если дочернее приложение завершится БЕЗ ExitBootServices, управление
 * вернётся сюда (и дальше в shell). Фолбэк при отсутствии LAUNCH_FILE --
 * печать SECRET_FILE, как было в v11. */
BOOLEAN CmdLaunchRequested(VOID) { return gLaunchReq; }
VOID    CmdRequestLaunch(VOID)   { gLaunchReq = TRUE; }

EFI_STATUS CmdLaunchApp(VOID) {
    if (gCmdRoot == NULL || gCmdDevice == NULL) {
        DBG("CmdLaunchApp: no root/device");
        return EFI_INVALID_PARAMETER;
    }
    EFI_FILE_PROTOCOL* File = NULL;
    EFI_STATUS s = gCmdRoot->Open(gCmdRoot, &File, LAUNCH_FILE, EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(s)) {
        DBG("CmdLaunchApp: LAUNCH_FILE not found -- secret.txt fallback");
        CmdOpenSecretFile();   /* файла нет -- старое поведение */
        return EFI_NOT_FOUND;
    }
    File->Close(File);

    EFI_DEVICE_PATH_PROTOCOL* Dp = FileDevicePath(gCmdDevice, LAUNCH_FILE);
    if (Dp == NULL) {
        DBG("CmdLaunchApp: FileDevicePath FAILED");
        CmdOpenSecretFile();
        return EFI_OUT_OF_RESOURCES;
    }
    EFI_HANDLE Child = NULL;
    s = gBS->LoadImage(FALSE, gImageHandle, Dp, NULL, 0, &Child);
    FreePool(Dp);
    if (EFI_ERROR(s)) {
        DBG_HEX("CmdLaunchApp: LoadImage FAILED", s);
        CmdOpenSecretFile();
        return s;
    }
    UINTN ExitSize = 0;
    CHAR16* ExitData = NULL;
    s = gBS->StartImage(Child, &ExitSize, &ExitData);
    DBG_HEX("CmdLaunchApp: StartImage returned", s);
    if (ExitData != NULL) {
        Print(L"%s\r\n", ExitData);
        FreePool(ExitData);
    }
    return s;
}

/* ------------------------------------------------------------------------
 * Определение конфигурации ПК
 * ------------------------------------------------------------------------ */
/* Убирает из CPUID-бренда служебные хвосты: "(R)", "(TM)", "CPU",
 * "@ x.xxGHz", "x-Core Processor", "with ...", двойные пробелы, хвостовые
 * пробелы/дефисы. В обучающих строках бренды короткие: "AMD Ryzen 5 3600",
 * "Intel Core i5-10400F" -- ровно это и должна увидеть модель. */
STATIC VOID CmdSanitizeBrand(CHAR8* s) {
    CHAR8* cut;
    if ((cut = AsciiStrStr(s, "(R)")) != NULL)  { CopyMem(cut, cut + 3, AsciiStrLen(cut + 3) + 1); }
    if ((cut = AsciiStrStr(s, "(TM)")) != NULL) { CopyMem(cut, cut + 4, AsciiStrLen(cut + 4) + 1); }
    if ((cut = AsciiStrStr(s, "(tm)")) != NULL) { CopyMem(cut, cut + 4, AsciiStrLen(cut + 4) + 1); }
    if ((cut = AsciiStrStr(s, "CPU")) != NULL)  { CopyMem(cut, cut + 3, AsciiStrLen(cut + 3) + 1); }
    if ((cut = AsciiStrStr(s, " @")) != NULL)   { *cut = '\0'; }
    if ((cut = AsciiStrStr(s, "-Core Processor")) != NULL) { *cut = '\0'; }
    if ((cut = AsciiStrStr(s, " Processor")) != NULL && AsciiStrStr(s, "Ryzen") == NULL) { *cut = '\0'; }
    if ((cut = AsciiStrStr(s, " with ")) != NULL) { *cut = '\0'; }
    /* схлопываем двойные пробелы */
    for (CHAR8* r = s; *r; ) {
        if (r[0] == ' ' && r[1] == ' ') { CopyMem(r, r + 1, AsciiStrLen(r + 1) + 1); continue; }
        r++;
    }
    UINTN n = AsciiStrLen(s);
    while (n > 0 && (s[n-1] == ' ' || s[n-1] == '-')) s[--n] = '\0';
}

STATIC VOID CmdCpuBrand(CHAR8* Out, UINTN Cap) {
    UINT32 MaxExt = 0, Ebx = 0, Ecx = 0, Edx = 0;
    AsmCpuid(0x80000000, &MaxExt, &Ebx, &Ecx, &Edx);
    if (MaxExt >= 0x80000004) {
        UINT32 Regs[12];
        CHAR8  Raw[49];
        AsmCpuid(0x80000002, &Regs[0], &Regs[1], &Regs[2],  &Regs[3]);
        AsmCpuid(0x80000003, &Regs[4], &Regs[5], &Regs[6],  &Regs[7]);
        AsmCpuid(0x80000004, &Regs[8], &Regs[9], &Regs[10], &Regs[11]);
        CopyMem(Raw, Regs, 48);
        Raw[48] = '\0';
        CHAR8* p = Raw;
        while (*p == ' ') p++;              /* Intel добивает строку пробелами слева */
        /* чистим суффиксы CPUID, которых нет в обучающих строках */
        CmdSanitizeBrand(p);
        AsciiSPrint(Out, Cap, "%a", p);
        return;
    }
    AsciiSPrint(Out, Cap, "%a", "неизвестный процессор");
}

/* Частота в десятых долях ГГц: TSC калибруется по gBS->Stall (50 мс).
 * На современных CPU TSC инвариантен и равен номинальной (базовой) частоте. */
STATIC UINT32 CmdCpuGhz10(VOID) {
    UINT64 T0 = AsmReadTsc();
    gBS->Stall(50 * 1000);
    UINT64 Dt = AsmReadTsc() - T0;
    UINT64 Hz = MultU64x32(Dt, 20);         /* * (1000 мс / 50 мс) */
    UINT32 Ghz10 = (UINT32)DivU64x32(Hz + 50000000ULL, 100000000U);
    if (Ghz10 == 0) Ghz10 = 10;
    return Ghz10;
}

STATIC UINT32 CmdCoreCount(VOID) {
    EFI_MP_SERVICES_PROTOCOL* Mp = NULL;
    UINTN Total = 0, Enabled = 0;
    if (!EFI_ERROR(gBS->LocateProtocol(&gEfiMpServiceProtocolGuid, NULL, (VOID**)&Mp)) && Mp != NULL) {
        if (!EFI_ERROR(Mp->GetNumberOfProcessors(Mp, &Total, &Enabled)) && Enabled > 0) {
            return (UINT32)Enabled;
        }
    }
    return 1;
}

STATIC UINT32 CmdRamGb(VOID) {
    UINTN  MapSize = 0, MapKey = 0, DescSize = 0;
    UINT32 DescVer = 0;
    EFI_STATUS s = gBS->GetMemoryMap(&MapSize, NULL, &MapKey, &DescSize, &DescVer);
    if (s != EFI_BUFFER_TOO_SMALL || DescSize == 0) return 0;
    MapSize += DescSize * 8;                /* запас: карта могла подрасти */
    EFI_MEMORY_DESCRIPTOR* Map = (EFI_MEMORY_DESCRIPTOR*)LmAlloc(MapSize);
    if (Map == NULL) return 0;
    s = gBS->GetMemoryMap(&MapSize, Map, &MapKey, &DescSize, &DescVer);
    if (EFI_ERROR(s)) { LmFree(Map); return 0; }

    UINT64 Pages = 0;
    UINT8* p   = (UINT8*)Map;
    UINT8* End = p + MapSize;
    for (; p + DescSize <= End; p += DescSize) {
        EFI_MEMORY_DESCRIPTOR* d = (EFI_MEMORY_DESCRIPTOR*)p;
        switch (d->Type) {
            case EfiLoaderCode:
            case EfiLoaderData:
            case EfiBootServicesCode:
            case EfiBootServicesData:
            case EfiRuntimeServicesCode:
            case EfiRuntimeServicesData:
            case EfiConventionalMemory:
            case EfiACPIReclaimMemory:
            case EfiACPIMemoryNVS:
                Pages += d->NumberOfPages;
                break;
            default:
                break;
        }
    }
    LmFree(Map);

    UINT64 Gb = RShiftU64(MultU64x32(Pages, 4096) + (1ULL << 29), 30); /* округление до ГБ */
    /* прошивка резервирует часть памяти -- прижимаем к ближайшему типовому объёму планок */
    STATIC CONST UINT32 Sizes[] = {1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512};
    UINT32 Best = (UINT32)Gb;
    UINT64 BestDiff = (UINT64)-1;
    UINTN  i;
    for (i = 0; i < sizeof(Sizes)/sizeof(Sizes[0]); i++) {
        UINT64 Diff = (Sizes[i] > Gb) ? (Sizes[i] - Gb) : (Gb - Sizes[i]);
        if (Diff < BestDiff) { BestDiff = Diff; Best = Sizes[i]; }
    }
    return Best;
}

/* Имена 1-в-1 из build_sysinfo() (prepare_data.py): модель обучена на них,
 * поэтому подставляем текстовое имя, а не "видеокарта NVIDIA [10DE:1C81]". */
typedef struct { UINT16 Vid; UINT16 Did; CONST CHAR8* Name; } GPU_NAME;
STATIC CONST GPU_NAME gGpuNames[] = {
    {0x10DE,0x0A65,"GeForce GT 210"}, {0x10DE,0x128B,"GeForce GT 710"},
    {0x1002,0x68F9,"Radeon HD 5450"},
    {0x10DE,0x1380,"GeForce GTX 750 Ti"}, {0x10DE,0x1C81,"GeForce GTX 1050"},
    {0x1002,0x699F,"Radeon RX 550"}, {0x10DE,0x1401,"GeForce GTX 960"},
    {0x10DE,0x21C4,"GeForce GTX 1660 Super"}, {0x10DE,0x2503,"GeForce RTX 3060"},
    {0x1002,0x73FF,"Radeon RX 6600"}, {0x10DE,0x1F02,"GeForce RTX 2070"},
    {0x10DE,0x2782,"GeForce RTX 4070 Ti"}, {0x10DE,0x2204,"GeForce RTX 3090"},
    {0x1002,0x744C,"Radeon RX 7900 XTX"}, {0x10DE,0x2704,"GeForce RTX 4080"},
    {0x10DE,0x2684,"GeForce RTX 4090"}, {0x10DE,0x2B85,"GeForce RTX 5090"},
};

STATIC VOID CmdGpuName(CHAR8* Out, UINTN Cap) {
    EFI_HANDLE* Handles = NULL;
    UINTN Count = 0, i;
    AsciiSPrint(Out, Cap, "%a", "видеокарта не определена");
    EFI_STATUS s = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &Count, &Handles);
    if (EFI_ERROR(s) || Handles == NULL) return;
    for (i = 0; i < Count; i++) {
        EFI_PCI_IO_PROTOCOL* Pci = NULL;
        if (EFI_ERROR(gBS->HandleProtocol(Handles[i], &gEfiPciIoProtocolGuid, (VOID**)&Pci))) continue;
        UINT32 ClassReg = 0;
        if (EFI_ERROR(Pci->Pci.Read(Pci, EfiPciIoWidthUint32, 0x08, 1, &ClassReg))) continue;
        if (((ClassReg >> 24) & 0xFF) != 0x03) continue;   /* 0x03 = display controller */
        UINT16 Vid = 0, Did = 0;
        Pci->Pci.Read(Pci, EfiPciIoWidthUint16, 0x00, 1, &Vid);
        Pci->Pci.Read(Pci, EfiPciIoWidthUint16, 0x02, 1, &Did);
        CONST CHAR8* Name = NULL;
        for (UINTN k = 0; k < sizeof(gGpuNames)/sizeof(gGpuNames[0]); k++) {
            if (gGpuNames[k].Vid == Vid && gGpuNames[k].Did == Did) { Name = gGpuNames[k].Name; break; }
        }
        if (Name == NULL) {
            /* встроенные GPU: имена тоже есть в обучающих данных */
            if (Vid == 0x8086)      Name = "встроенная Intel GMA";
            else if (Vid == 0x1002) Name = "встроенная Vega 3";
            else if (Vid == 0x10DE) Name = "видеокарта NVIDIA";
            else if (Vid == 0x15AD) Name = "VMware SVGA";
            else if (Vid == 0x1AF4) Name = "QEMU virtio";
            else                    Name = "видеокарта неизвестная";
        }
        AsciiSPrint(Out, Cap, "%a", Name);
        break;   /* берём первый дисплей-контроллер */
    }
    gBS->FreePool(Handles);
}

/* Диск в формате обучающих строк: "HDD 500 ГБ", "SSD 1 ТБ", "NVMe 2 ТБ".
 * NVMe определяем по PCI-классу 01/08/02 (mass storage / NVM / NVMe). */
STATIC VOID CmdDiskStr(CHAR8* Out, UINTN Cap) {
    EFI_HANDLE* Handles = NULL;
    UINTN  Count = 0, i;
    UINT64 BestBytes = 0;
    AsciiSPrint(Out, Cap, "%a", "");
    EFI_STATUS s = gBS->LocateHandleBuffer(ByProtocol, &gEfiBlockIoProtocolGuid, NULL, &Count, &Handles);
    if (!EFI_ERROR(s) && Handles != NULL) {
        for (i = 0; i < Count; i++) {
            EFI_BLOCK_IO_PROTOCOL* Bio = NULL;
            if (EFI_ERROR(gBS->HandleProtocol(Handles[i], &gEfiBlockIoProtocolGuid, (VOID**)&Bio))) continue;
            if (Bio->Media == NULL) continue;
            if (Bio->Media->LogicalPartition) continue;
            if (!Bio->Media->MediaPresent) continue;
            UINT64 Bytes = MultU64x32(Bio->Media->LastBlock + 1, Bio->Media->BlockSize);
            if (Bytes > BestBytes) BestBytes = Bytes;
        }
        gBS->FreePool(Handles);
    }
    if (BestBytes == 0) return;
    UINT32 Gb = (UINT32)RShiftU64(BestBytes + (1ULL << 29), 30);
    BOOLEAN Nvme = FALSE;
    Handles = NULL; Count = 0;
    s = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &Count, &Handles);
    if (!EFI_ERROR(s) && Handles != NULL) {
        for (i = 0; i < Count && !Nvme; i++) {
            EFI_PCI_IO_PROTOCOL* Pci = NULL;
            if (EFI_ERROR(gBS->HandleProtocol(Handles[i], &gEfiPciIoProtocolGuid, (VOID**)&Pci))) continue;
            UINT32 ClassReg = 0;
            if (EFI_ERROR(Pci->Pci.Read(Pci, EfiPciIoWidthUint32, 0x08, 1, &ClassReg))) continue;
            if (((ClassReg >> 24) & 0xFF) == 0x01 && ((ClassReg >> 16) & 0xFF) == 0x08) Nvme = TRUE;
        }
        gBS->FreePool(Handles);
    }
    AsciiSPrint(Out, Cap, "%a %d ГБ", Nvme ? "NVMe" : (Gb >= 100 ? "SSD" : "HDD"), Gb);
}

/* «1 ядро / 2 ядра / 5 ядер» -- как в обучающих данных build_sysinfo() */
STATIC CONST CHAR8* CmdCoresWord(UINT32 n) {
    UINT32 n100 = n % 100;
    UINT32 n10  = n % 10;
    if (n100 >= 11 && n100 <= 14) return "ядер";
    if (n10 == 1)                 return "ядро";
    if (n10 >= 2 && n10 <= 4)     return "ядра";
    return "ядер";
}

VOID CmdGatherConfig(CHAR8* Out, UINTN Cap) {
    CHAR8  Brand[64];
    CHAR8  Gpu[96];
    CHAR8  Disk[40];
    UINT32 Ghz10, Cores, Ram;

    CmdCpuBrand(Brand, sizeof(Brand));
    Ghz10 = CmdCpuGhz10();
    Cores = CmdCoreCount();
    Ram   = CmdRamGb();
    CmdGpuName(Gpu, sizeof(Gpu));
    CmdDiskStr(Disk, sizeof(Disk));

    if (Disk[0] != '\0') {
        AsciiSPrint(Out, Cap,
            "(конфиг: %a, %d.%d ГГц, %d %a, %d ГБ ОЗУ, %a, %a)",
            Brand, Ghz10 / 10, Ghz10 % 10, Cores, CmdCoresWord(Cores), Ram, Gpu, Disk);
    } else {
        AsciiSPrint(Out, Cap,
            "(конфиг: %a, %d.%d ГГц, %d %a, %d ГБ ОЗУ, %a)",
            Brand, Ghz10 / 10, Ghz10 % 10, Cores, CmdCoresWord(Cores), Ram, Gpu);
    }
    DBG("CmdGatherConfig: done");
}
