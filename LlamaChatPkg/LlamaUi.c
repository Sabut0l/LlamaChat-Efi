//Copyright (c) 2026 Sabut0l
/* =========================================================================
 * LlamaUi.c -- EFI Shell text console with UTF-8/Cyrillic conversion.
 *
 * ASCII and control characters remain handled by SimpleTextOutput. Cyrillic
 * glyphs are drawn into the current EFI Shell character cell through GOP
 * using an 8x19 bitmap font with OVMF (GraphicsConsoleDxe) metrics.
 *
 * Cell layout matches MdeModulePkg GraphicsConsoleDxe exactly:
 *   - glyph cell is EFI_GLYPH_WIDTH x EFI_GLYPH_HEIGHT = 8 x 19 pixels;
 *   - the text area is CENTERED on the screen: the firmware draws text at
 *     DeltaX = (HRes - Columns*8) / 2, DeltaY = (VRes - Rows*19) / 2.
 * The old code assumed cell = HRes/Columns x VRes/Rows with origin (0,0),
 * which is wrong whenever the shell text mode does not exactly tile the
 * framebuffer (e.g. 80x25 on 800x600: real origin is (80,62), not (0,0)),
 * so overlay glyphs drifted away from the firmware's own text cells.
 * ========================================================================= */
#include "LlamaUi.h"
#include "LlamaCyrFont.h"

#include <Library/BaseMemoryLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Protocol/SimpleTextInEx.h>
#include <Protocol/GraphicsOutput.h>

STATIC EFI_SIMPLE_TEXT_INPUT_EX_PROTOCOL* gInputEx = NULL;
STATIC EFI_GRAPHICS_OUTPUT_PROTOCOL* gGop = NULL;
STATIC BOOLEAN gReady = FALSE;
STATIC BOOLEAN gRussian = FALSE;
STATIC BOOLEAN gAltLatch = FALSE;
STATIC UINT32 gUtf8Codepoint = 0;
STATIC UINT8 gUtf8Need = 0;
STATIC UINTN gTextColumns = 0;
STATIC UINTN gTextRows = 0;
STATIC UINTN gOriginX = 0;      /* DeltaX of the centered text area */
STATIC UINTN gOriginY = 0;      /* DeltaY of the centered text area */
STATIC BOOLEAN gGlyphOk = FALSE;

/* OVMF / GraphicsConsoleDxe glyph cell (EFI_GLYPH_WIDTH/HEIGHT). */
#define UI_GLYPH_W 8
#define UI_GLYPH_H 19

STATIC CONST EFI_GRAPHICS_OUTPUT_BLT_PIXEL gConsolePalette[16] = {
    {0x00,0x00,0x00,0}, {0x98,0x00,0x00,0}, {0x00,0x98,0x00,0}, {0x98,0x98,0x00,0},
    {0x00,0x00,0x98,0}, {0x98,0x00,0x98,0}, {0x00,0x98,0x98,0}, {0x98,0x98,0x98,0},
    {0x30,0x30,0x30,0}, {0xFF,0x00,0x00,0}, {0x00,0xFF,0x00,0}, {0xFF,0xFF,0x00,0},
    {0x00,0x00,0xFF,0}, {0xFF,0x00,0xFF,0}, {0x00,0xFF,0xFF,0}, {0xFF,0xFF,0xFF,0}
};

STATIC CONST UINT8* FindCyrillicGlyph(UINT32 Cp) {
    UINTN Lo = 0, Hi = LLAMA_CYR_FONT_COUNT;
    while (Lo < Hi) {
        UINTN Mid = Lo + (Hi - Lo) / 2;
        if (gLlamaCyrFont[Mid].Codepoint == Cp) return gLlamaCyrFont[Mid].Rows;
        if (gLlamaCyrFont[Mid].Codepoint < Cp) Lo = Mid + 1; else Hi = Mid;
    }
    return NULL;
}

STATIC BOOLEAN DrawCyrillicGlyph(UINT32 Cp) {
    CONST UINT8* Glyph = FindCyrillicGlyph(Cp);
    if (Glyph == NULL || !gGlyphOk || gGop == NULL || gST->ConOut == NULL)
        return FALSE;

    UINTN Col = (UINTN)gST->ConOut->Mode->CursorColumn;
    UINTN Row = (UINTN)gST->ConOut->Mode->CursorRow;
    if (Col >= gTextColumns || Row >= gTextRows) return FALSE;

    UINTN Attr = (UINTN)gST->ConOut->Mode->Attribute;
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL Fg = gConsolePalette[Attr & 0x0F];
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL Bg = gConsolePalette[(Attr >> 4) & 0x0F];

    /* Hide the firmware cursor while we draw. GraphicsConsoleDxe erases its
     * cursor by INVERTING the cell pixels: if we Blt a glyph while the
     * cursor underscore is shown, the console later "removes" the cursor by
     * inverting our pixels, leaving a permanent stripe under the glyph. */
    BOOLEAN CursorWasVisible = gST->ConOut->Mode->CursorVisible;
    if (CursorWasVisible) {
        gST->ConOut->EnableCursor(gST->ConOut, FALSE);
    }

    EFI_GRAPHICS_OUTPUT_BLT_PIXEL Pixels[UI_GLYPH_W * UI_GLYPH_H];
    for (UINTN Y = 0; Y < UI_GLYPH_H; Y++) {
        for (UINTN X = 0; X < UI_GLYPH_W; X++) {
            /* Bit 7 is the leftmost pixel, 1:1, no scaling -- exactly like
             * GraphicsConsoleDxe draws its own EFI_NARROW_GLYPH data. */
            Pixels[Y * UI_GLYPH_W + X] =
                (Glyph[Y] & (0x80u >> X)) ? Fg : Bg;
        }
    }
    gGop->Blt(gGop, Pixels, EfiBltBufferToVideo, 0, 0,
              gOriginX + Col * UI_GLYPH_W, gOriginY + Row * UI_GLYPH_H,
              UI_GLYPH_W, UI_GLYPH_H,
              UI_GLYPH_W * sizeof(Pixels[0]));

    if (Col + 1 < gTextColumns) {
        gST->ConOut->SetCursorPosition(gST->ConOut, Col + 1, Row);
    } else if (Row + 1 < gTextRows) {
        /* fall through to the wrap handling below */
        gST->ConOut->SetCursorPosition(gST->ConOut, 0, Row + 1);
    } else {
        CHAR16 NewLine[] = L"\r\n";
        gST->ConOut->OutputString(gST->ConOut, NewLine);
    }

    /* Re-enable the cursor only after the cursor position has moved, so the
     * firmware draws it in the NEXT cell and never inverts our glyph. */
    if (CursorWasVisible) {
        gST->ConOut->EnableCursor(gST->ConOut, TRUE);
    }
    return TRUE;
}

BOOLEAN UiIsReady(VOID) { return gReady; }

EFI_STATUS UiInit(VOID) {
    /* InputEx is optional. F2 still works through the basic input protocol. */
    gBS->HandleProtocol(
        gST->ConsoleInHandle,
        &gEfiSimpleTextInputExProtocolGuid,
        (VOID**)&gInputEx);
    gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID**)&gGop);
    if (gST->ConOut != NULL && gST->ConOut->Mode != NULL && gGop != NULL) {
        UINTN Columns = 0, Rows = 0;
        EFI_STATUS S = gST->ConOut->QueryMode(
            gST->ConOut, (UINTN)gST->ConOut->Mode->Mode, &Columns, &Rows);
        if (!EFI_ERROR(S) && Columns != 0 && Rows != 0) {
            gTextColumns = Columns;
            gTextRows = Rows;
            /* GraphicsConsoleDxe metrics: fixed 8x19 cell, text area
             * centered on the framebuffer. */
            UINTN HRes = gGop->Mode->Info->HorizontalResolution;
            UINTN VRes = gGop->Mode->Info->VerticalResolution;
            if (Columns * UI_GLYPH_W <= HRes && Rows * UI_GLYPH_H <= VRes) {
                gOriginX = (HRes - Columns * UI_GLYPH_W) / 2;
                gOriginY = (VRes - Rows * UI_GLYPH_H) / 2;
                gGlyphOk = TRUE;
            }
        }
    }
    gReady = TRUE;
    if (gST->ConOut != NULL) gST->ConOut->EnableCursor(gST->ConOut, TRUE);
    return EFI_SUCCESS;
}

VOID UiClear(VOID) {
    if (!gReady || gST->ConOut == NULL) return;
    gST->ConOut->ClearScreen(gST->ConOut);
}

VOID UiWriteCodepoint(UINT32 Cp) {
    if (!gReady || gST->ConOut == NULL) return;
    if (Cp == '\r') return;
    if (Cp == '\n') {
        CHAR16 NewLine[] = L"\r\n";
        gST->ConOut->OutputString(gST->ConOut, NewLine);
        return;
    }
    if (Cp == '\t') {
        CHAR16 Tab[] = L"    ";
        gST->ConOut->OutputString(gST->ConOut, Tab);
        return;
    }
    if (Cp < 0x20) return;

    if ((Cp == 0x0401 || (Cp >= 0x0410 && Cp <= 0x044F) || Cp == 0x0451) &&
        DrawCyrillicGlyph(Cp)) return;

    CHAR16 Out[3];
    if (Cp <= 0xFFFF) {
        Out[0] = (CHAR16)Cp;
        Out[1] = 0;
    } else if (Cp <= 0x10FFFF) {
        Cp -= 0x10000;
        Out[0] = (CHAR16)(0xD800 + (Cp >> 10));
        Out[1] = (CHAR16)(0xDC00 + (Cp & 0x3FF));
        Out[2] = 0;
    } else {
        Out[0] = L'?';
        Out[1] = 0;
    }
    gST->ConOut->OutputString(gST->ConOut, Out);
}

VOID UiWriteUtf8(CONST CHAR8* Text) {
    if (!gReady || Text == NULL) return;
    while (*Text) {
        UINT8 B = (UINT8)*Text++;
        if (gUtf8Need == 0) {
            if (B < 0x80) UiWriteCodepoint(B);
            else if ((B & 0xE0) == 0xC0) { gUtf8Codepoint = B & 0x1F; gUtf8Need = 1; }
            else if ((B & 0xF0) == 0xE0) { gUtf8Codepoint = B & 0x0F; gUtf8Need = 2; }
            else if ((B & 0xF8) == 0xF0) { gUtf8Codepoint = B & 0x07; gUtf8Need = 3; }
            else UiWriteCodepoint('?');
        } else if ((B & 0xC0) == 0x80) {
            gUtf8Codepoint = (gUtf8Codepoint << 6) | (B & 0x3F);
            if (--gUtf8Need == 0) UiWriteCodepoint(gUtf8Codepoint);
        } else {
            gUtf8Need = 0;
            gUtf8Codepoint = 0;
            UiWriteCodepoint('?');
            Text--;
        }
    }
}

STATIC VOID Backspace(VOID) {
    if (!gReady || gST->ConOut == NULL) return;
    CHAR16 Erase[] = L"\b \b";
    gST->ConOut->OutputString(gST->ConOut, Erase);
}

STATIC VOID ShowLayout(VOID) {
    if (!gReady || gST->ConOut == NULL) return;
    gST->ConOut->OutputString(gST->ConOut, gRussian ? L" [RU] " : L" [EN] ");
}

STATIC UINT32 RussianKey(CHAR16 Ch, BOOLEAN Upper) {
    CHAR8 C8 = (Ch <= 0x7F) ? (CHAR8)Ch : 0;
    if (C8 >= 'A' && C8 <= 'Z') C8 = (CHAR8)(C8 + ('a' - 'A'));
    CONST CHAR8* Keys = "`qwertyuiop[]asdfghjkl;'zxcvbnm,.";
    STATIC CONST UINT16 Lower[] = {
        0x0451,0x0439,0x0446,0x0443,0x043A,0x0435,0x043D,0x0433,0x0448,0x0449,0x0437,0x0445,0x044A,
        0x0444,0x044B,0x0432,0x0430,0x043F,0x0440,0x043E,0x043B,0x0434,0x0436,0x044D,
        0x044F,0x0447,0x0441,0x043C,0x0438,0x0442,0x044C,0x0431,0x044E
    };
    for (UINTN I = 0; Keys[I]; I++) if (C8 == Keys[I] ||
        (Keys[I]=='`' && C8=='~') || (Keys[I]=='[' && C8=='{') || (Keys[I]==']' && C8=='}') ||
        (Keys[I]==';' && C8==':') || (Keys[I]=='\'' && C8=='"') ||
        (Keys[I]==',' && C8=='<') || (Keys[I]=='.' && C8=='>')) {
        UINT32 Cp = Lower[I];
        if (Upper) Cp = (Cp == 0x0451) ? 0x0401 : Cp - 0x20;
        return Cp;
    }
    return Ch;
}

STATIC UINTN EncodeUtf8(UINT32 Cp, CHAR8 Out[4]) {
    if (Cp < 0x80) { Out[0]=(CHAR8)Cp; return 1; }
    if (Cp < 0x800) {
        Out[0]=(CHAR8)(0xC0|(Cp>>6)); Out[1]=(CHAR8)(0x80|(Cp&0x3F)); return 2;
    }
    Out[0]=(CHAR8)(0xE0|(Cp>>12));
    Out[1]=(CHAR8)(0x80|((Cp>>6)&0x3F));
    Out[2]=(CHAR8)(0x80|(Cp&0x3F));
    return 3;
}

STATIC VOID ToggleLayout(VOID) {
    gRussian = !gRussian;
    ShowLayout();
}

VOID UiReadLineUtf8(CHAR8* Buffer, UINTN BufferSize) {
    UINTN Len = 0;
    if (Buffer == NULL || BufferSize == 0) return;
    Buffer[0] = 0;

    for (;;) {
        EFI_INPUT_KEY Key;
        EFI_KEY_STATE State;
        SetMem(&State, sizeof(State), 0);

        if (gInputEx != NULL) {
            UINTN Index;
            gBS->WaitForEvent(1, &gInputEx->WaitForKeyEx, &Index);
            EFI_KEY_DATA Data;
            if (EFI_ERROR(gInputEx->ReadKeyStrokeEx(gInputEx, &Data))) continue;
            Key = Data.Key;
            State = Data.KeyState;
        } else {
            UINTN Index;
            gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &Index);
            if (EFI_ERROR(gST->ConIn->ReadKeyStroke(gST->ConIn, &Key))) continue;
        }

        BOOLEAN Shift = (State.KeyShiftState &
            (EFI_LEFT_SHIFT_PRESSED | EFI_RIGHT_SHIFT_PRESSED)) != 0;
        BOOLEAN Alt = (State.KeyShiftState &
            (EFI_LEFT_ALT_PRESSED | EFI_RIGHT_ALT_PRESSED)) != 0;

        if (Key.ScanCode == SCAN_F2) { ToggleLayout(); gAltLatch = Alt; continue; }
        if (Alt && !gAltLatch) { ToggleLayout(); gAltLatch = TRUE; continue; }
        if (!Alt) gAltLatch = FALSE;
        if (Alt) continue;

        if (Key.UnicodeChar == CHAR_CARRIAGE_RETURN) {
            UiWriteCodepoint('\n');
            break;
        }
        if (Key.UnicodeChar == CHAR_BACKSPACE) {
            if (Len > 0) {
                do { Len--; } while (Len > 0 && (((UINT8)Buffer[Len] & 0xC0) == 0x80));
                Buffer[Len] = 0;
                Backspace();
            }
            continue;
        }
        if (Key.UnicodeChar < 0x20) continue;

        BOOLEAN Caps = (State.KeyToggleState & EFI_CAPS_LOCK_ACTIVE) != 0;
        UINT32 Cp = gRussian ? RussianKey(Key.UnicodeChar, Shift ^ Caps) : Key.UnicodeChar;
        CHAR8 Enc[4];
        UINTN N = EncodeUtf8(Cp, Enc);
        if (Len + N >= BufferSize) continue;
        CopyMem(Buffer + Len, Enc, N);
        Len += N;
        Buffer[Len] = 0;
        UiWriteCodepoint(Cp);
    }
}
