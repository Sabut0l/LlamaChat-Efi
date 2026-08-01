#ifndef LLAMA_UI_H
#define LLAMA_UI_H

#include <Uefi.h>

/* EFI Shell console with UTF-8 and bitmap Cyrillic overlay. */
EFI_STATUS UiInit(VOID);
BOOLEAN    UiIsReady(VOID);
VOID       UiClear(VOID);
VOID       UiWriteUtf8(CONST CHAR8* Text);
VOID       UiWriteCodepoint(UINT32 Codepoint);
VOID       UiReadLineUtf8(CHAR8* Buffer, UINTN BufferSize);

#endif
