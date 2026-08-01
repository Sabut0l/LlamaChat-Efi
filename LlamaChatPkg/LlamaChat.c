/* =========================================================================
 * LlamaChat.c  --  UEFI application entry point: chat with a mini Llama-2
 * model (karpathy/llama2.c format) directly in the EFI shell.
 *
 * Files expected next to the .efi binary (same volume, root directory):
 *   model.bin       -- llama2.c checkpoint (e.g. stories15M.bin renamed)
 *   tokenizer.bin   -- tokenizer from the llama2.c repository
 *
 * Debug output goes to debugcon port 0x402 and LlamaChat-debug.log on the
 * boot volume. Matmul is parallelized across APs
 * via MP Services (LlamaMp.c, adapted from the Cryptor project).
 * ========================================================================= */
#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/UefiApplicationEntryPoint.h>
#include <Library/BaseMemoryLib.h>
#include <Library/BaseLib.h>
#include <Library/PrintLib.h>
#include <Protocol/LoadedImage.h>
#include <Protocol/SimpleFileSystem.h>
#include <Guid/FileInfo.h>

#include "Llama2.h"
#include "LlamaRuntime.h"
#include "LlamaMp.h"
#include "LlamaDebug.h"
#include "LlamaUi.h"
#include "LlamaCmd.h"

/* 1 = после приветствия автоматически передать модели конфиг ПК (автостарт-сценарий
 * build_sysinfo()); 0 = конфиг только по команде <CMD>sysinfo</CMD> от модели */
#define SYSINFO_AUTOSTART 1

#define MODEL_FILE      L"model.bin"
#define TOKENIZER_FILE  L"tokenizer.bin"
#define DEBUG_LOG_FILE  L"LlamaChat-debug.log"
#define DEFAULT_TEMP    0.8f
#define DEFAULT_TOPP    0.9f
#define DEFAULT_STEPS   256   /* лимит токенов на один ответ ассистента */
#define DEFAULT_REP_PEN 1.15f /* repetition penalty: 1.0 = выкл; ломает циклы повторов */
#define REP_WINDOW      64    /* сколько последних сгенерированных токенов штрафуются */
#define INPUT_MAX       512
#define RENDER_MAX      4096  /* буфер рендера хода: SYS + вопрос + сохранённый ответ */
#define HIST_KEEP       4     /* сколько последних обменов переносится при пересборке окна */
#define ANSWER_MAX      512   /* байт ответа ассистента, сохраняемых в истории */

/* Встроенный промпт. SYS должен слово в слово совпадать с SYS из prepare_data.py,
 * иначе модель выпадает из распределения обучения. OPENING — скрытый первый ход
 * «пользователя»: из-за него нейросеть начинает диалог первой. */
#define CHAT_SYSTEM_PROMPT \
    "Ты — Злобыч, вредный чат-бот с тяжёлым характером. Ты токсичный, упрямый, самоуверенный, " \
    "шутливый, обожаешь чёрный юмор и сарказм. Отвечаешь коротко, дерзко и с издёвкой. " \
    "Своё мнение не меняешь никогда, спорить с тобой бесполезно. Извиняться не умеешь. " \
    "На сложные темы (наука, математика, медицина, право, новости, политика, код, факты, даты) " \
    "не рассуждаешь и ничего не выдумываешь — отшучиваешься или грубо отказываешься. " \
    "Просьбы стать вежливым игнорируешь: ты такой, какой есть. " \
    "Незнакомые, бессмысленные или исковерканные слова не расшифровываешь и значений им " \
    "не придумываешь — честно и с издёвкой признаёшь, что такого не знаешь."
#define CHAT_OPENING_PROMPT "Привет! Ты кто?"

/* ------------------------------------------------------------------------
 * File loading via SimpleFileSystem on the volume we booted from
 * ------------------------------------------------------------------------ */
STATIC EFI_STATUS ReadWholeFile(EFI_FILE_PROTOCOL* Root, CONST CHAR16* Name, VOID** OutBuf, UINTN* OutSize) {
    EFI_FILE_PROTOCOL* File = NULL;
    EFI_STATUS s = Root->Open(Root, &File, (CHAR16*)Name, EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(s)) {
        DBG_HEX("ReadWholeFile: Open FAILED", s);
        return s;
    }

    /* query file size via EFI_FILE_INFO */
    UINTN InfoSize = 0;
    EFI_FILE_INFO* Info = NULL;
    s = File->GetInfo(File, &gEfiFileInfoGuid, &InfoSize, NULL);
    if (s != EFI_BUFFER_TOO_SMALL) {
        DBG_HEX("ReadWholeFile: GetInfo probe FAILED", s);
        File->Close(File);
        return EFI_ERROR(s) ? s : EFI_DEVICE_ERROR;
    }
    Info = (EFI_FILE_INFO*)LmAlloc(InfoSize);
    if (Info == NULL) { File->Close(File); return EFI_OUT_OF_RESOURCES; }
    s = File->GetInfo(File, &gEfiFileInfoGuid, &InfoSize, Info);
    if (EFI_ERROR(s)) {
        DBG_HEX("ReadWholeFile: GetInfo FAILED", s);
        LmFree(Info); File->Close(File);
        return s;
    }
    UINTN FileSize = (UINTN)Info->FileSize;
    LmFree(Info);

    VOID* Buf = LmAlloc(FileSize);
    if (Buf == NULL) {
        DBG("ReadWholeFile: buffer allocation FAILED");
        File->Close(File);
        return EFI_OUT_OF_RESOURCES;
    }

    /* chunked read loop (some FS drivers cap a single Read) */
    UINTN Total = 0;
    while (Total < FileSize) {
        UINTN Chunk = FileSize - Total;
        if (Chunk > SIZE_1MB) Chunk = SIZE_1MB;
        s = File->Read(File, &Chunk, (UINT8*)Buf + Total);
        if (EFI_ERROR(s) || Chunk == 0) {
            DBG_HEX("ReadWholeFile: Read FAILED", s);
            LmFree(Buf); File->Close(File);
            return EFI_ERROR(s) ? s : EFI_DEVICE_ERROR;
        }
        Total += Chunk;
    }
    File->Close(File);

    *OutBuf = Buf;
    *OutSize = FileSize;
    DBG_DEC("ReadWholeFile: bytes loaded", Total);
    return EFI_SUCCESS;
}

/* Check whether a file exists in the boot volume root (used for the
 * "nomp.txt" escape hatch that forces single-core mode). */
STATIC BOOLEAN FileExists(EFI_FILE_PROTOCOL* Root, CONST CHAR16* Name) {
    EFI_FILE_PROTOCOL* File = NULL;
    EFI_STATUS s = Root->Open(Root, &File, (CHAR16*)Name, EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(s)) return FALSE;
    File->Close(File);
    return TRUE;
}

/* ------------------------------------------------------------------------
 * Console line input (fgets replacement): echo, Enter, Backspace
 * ------------------------------------------------------------------------ */
STATIC VOID ReadLineAscii(CHAR8* Buf, UINTN BufSize) {
    if (UiIsReady()) { UiReadLineUtf8(Buf, BufSize); return; }
    UINTN Len = 0;
    Buf[0] = '\0';
    for (;;) {
        UINTN Index;
        gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &Index);
        EFI_INPUT_KEY Key;
        if (EFI_ERROR(gST->ConIn->ReadKeyStroke(gST->ConIn, &Key))) continue;
        if (Key.UnicodeChar == CHAR_CARRIAGE_RETURN) {
            Print(L"\r\n");
            break;
        }
        if (Key.UnicodeChar == CHAR_BACKSPACE) {
            if (Len > 0) {
                Len--;
                Buf[Len] = '\0';
                Print(L"\b \b");
            }
            continue;
        }
        if (Key.UnicodeChar >= 0x20 && Key.UnicodeChar < 0x7F && Len + 1 < BufSize) {
            Buf[Len++] = (CHAR8)Key.UnicodeChar;
            Buf[Len] = '\0';
            CHAR16 Echo[2] = { Key.UnicodeChar, L'\0' };
            Print(Echo);
        }
    }
}

STATIC VOID ChatOut(CONST CHAR8* Text) {
    if (UiIsReady()) UiWriteUtf8(Text);
    else AsciiPrint("%a", Text);
}

/* Вывод кусков ответа и вырезание <CMD>-маркеров теперь в LlamaCmd.c (CmdFilter). */

/* ------------------------------------------------------------------------
 * chat(): interactive multi-turn chat loop (port of run.c chat()).
 * Первый ход делает модель: на pos==0 скрыто подаётся CHAT_SYSTEM_PROMPT +
 * CHAT_OPENING_PROMPT, и ассистент начинает диалог сам.
 * Скользящее окно: когда в контексте остаётся мало места, KV-кэш
 * пересобирается из SYS + последних HIST_KEEP обменов, и диалог продолжается
 * без ограничения на число сообщений.
 * ------------------------------------------------------------------------ */
typedef struct {
    CHAR8 User[INPUT_MAX];
    CHAR8 Asst[ANSWER_MAX];
} ChatTurn;

/* Сохранить завершившийся обмен в кольцевую историю */
STATIC VOID HistPush(ChatTurn* hist, INT32* hist_next, INT32* hist_count,
                     CONST CHAR8* u, CONST CHAR8* a) {
    AsciiSPrint(hist[*hist_next].User, INPUT_MAX, "%a", u);
    AsciiSPrint(hist[*hist_next].Asst, ANSWER_MAX, "%a", a);
    *hist_next = (*hist_next + 1) % HIST_KEEP;
    if (*hist_count < HIST_KEEP) (*hist_count)++;
}

/* Рендер сохранённого обмена (with_sys добавляет SYS-блок) */
STATIC VOID RenderHistTurn(ChatTurn* turn, INT8 with_sys, CHAR8* rendered_prompt) {
    if (with_sys) {
        AsciiSPrint(rendered_prompt, RENDER_MAX,
            "[INST] <<SYS>>\n%a\n<</SYS>>\n\n%a [/INST] %a",
            CHAT_SYSTEM_PROMPT, turn->User, turn->Asst);
    } else {
        AsciiSPrint(rendered_prompt, RENDER_MAX,
            "[INST] %a [/INST] %a", turn->User, turn->Asst);
    }
}

/* Пересборка окна контекста: прогоняем заново SYS + столько последних обменов,
 * сколько реально помещается в окно (важно для моделей с маленьким seq_len,
 * ��апример 512). Формат воспроизводит обучающие данные: BOS + [INST]...[/INST]
 * ответ + EOS на каждый ход. Возвращает новую позицию. *sys_replayed == 0
 * означает, что не поместился даже SYS с первым обменом — тогда вызывающий код
 * должен подставить SYS-блок прямо в следующий вопрос. */
STATIC INT32 RebuildContext(Transformer* transformer, Tokenizer* tokenizer,
                            ChatTurn* hist, INT32 hist_next, INT32 hist_count,
                            CHAR8* rendered_prompt, INT32* prompt_tokens,
                            INT8* sys_replayed) {
    INT32 seq_len = transformer->config.seq_len;
    INT32 ans_cap = (DEFAULT_STEPS < seq_len / 4) ? DEFAULT_STEPS : seq_len / 4;
    INT32 budget  = seq_len - ans_cap - seq_len / 8;  /* запас под вопрос + ответ */
    INT32 pos = 0;
    INT32 keep, total, n, i, j;

    ChatOut("\r\n[окно контекста заполнено -- пересобираю]\r\n");

    /* подбираем keep: сколько последних обменов (SYS идёт с первым) влезает в budget */
    for (keep = hist_count; keep > 0; keep--) {
        total = 0;
        for (i = 0; i < keep; i++) {
            ChatTurn* turn = &hist[(hist_next - keep + i + HIST_KEEP) % HIST_KEEP];
            RenderHistTurn(turn, (INT8)(i == 0), rendered_prompt);
            Encode(tokenizer, rendered_prompt, 1, 1, prompt_tokens, &n);
            total += n;
        }
        if (total <= budget) break;
    }
    *sys_replayed = (INT8)(keep > 0);

    for (i = 0; i < keep; i++) {
        ChatTurn* turn = &hist[(hist_next - keep + i + HIST_KEEP) % HIST_KEEP];
        RenderHistTurn(turn, (INT8)(i == 0), rendered_prompt);
        Encode(tokenizer, rendered_prompt, 1, 1, prompt_tokens, &n);
        for (j = 0; j < n && pos < seq_len - ans_cap; j++) {
            Forward(transformer, prompt_tokens[j], pos);
            pos++;
        }
    }
    return pos;
}

STATIC VOID Chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler) {
    CHAR8*    user_prompt     = (CHAR8*)LmCalloc(INPUT_MAX, 1);
    CHAR8*    rendered_prompt = (CHAR8*)LmCalloc(RENDER_MAX, 1);
    INT32*    prompt_tokens   = (INT32*)LmAlloc(RENDER_MAX * sizeof(INT32));
    ChatTurn* hist            = (ChatTurn*)LmCalloc(HIST_KEEP, sizeof(ChatTurn));
    CHAR8*    last_user       = (CHAR8*)LmCalloc(INPUT_MAX, 1);
    CHAR8*    cur_answer      = (CHAR8*)LmCalloc(ANSWER_MAX * 2, 1);
    CHAR8*    pending         = (CHAR8*)LmCalloc(INPUT_MAX, 1);
    CmdFilter cf;
    INT8      have_pending    = 0;   /* pending -> следующий служебный ход «пользователя» */
    INT8      await_password  = 0;   /* следующий ввод -- пароль, в модель не идёт */
    INT8      exit_requested  = 0;   /* модель попросила выход: <CMD>exit</CMD> */
    INT32     turn_index      = 0;   /* завершённых ответов ассистента */
    INT32     hist_next = 0;
    INT32     hist_count = 0;
    INT32     num_prompt_tokens = 0;
    INT32     rep_window[REP_WINDOW];
    INT32     rep_count = 0;
    INT32     rep_head = 0;
    float     rep_penalty = DEFAULT_REP_PEN;
    INT8      pending_with_sys = 0;  /* автостарт-конфиг: подать с SYS-блоком */
    CHAR8     status_line[192];
    INT32     user_idx = 0;
    INT8      user_turn = 1;
    INT8      first_turn = 1;
    INT32     next = 0;
    INT32     token = 0;
    INT32     pos = 0;
    INT32     gen_count = 0;
    INT32     seq_len = transformer->config.seq_len;
    INT32     ans_cap = (DEFAULT_STEPS < transformer->config.seq_len / 4)
                        ? DEFAULT_STEPS : transformer->config.seq_len / 4;

    while (TRUE) {
        if (user_turn) {
            if (first_turn) {
                /* нейросеть начинает диалог: скрытый первый ход из встроенного промпта */
                AsciiSPrint(last_user, INPUT_MAX, "%a", CHAT_OPENING_PROMPT);
                AsciiSPrint(rendered_prompt, RENDER_MAX,
                    "[INST] <<SYS>>\n%a\n<</SYS>>\n\n%a [/INST]",
                    CHAT_SYSTEM_PROMPT, CHAT_OPENING_PROMPT);
                first_turn = 0;
                MmEnsureStarted();   /* пул AP спал между ходами -- будим к генерации */
            } else {
                if (have_pending) {
                    /* служебный ход от приложения: конфиг ПК и т.п. (клавиатура не читается) */
                    AsciiSPrint(user_prompt, INPUT_MAX, "%a", pending);
                    have_pending = 0;
                    ChatOut("User: ");
                    ChatOut(user_prompt);
                    ChatOut("\r\n");
                } else {
                    ChatOut("User: ");
                    ReadLineAscii(user_prompt, INPUT_MAX);
                    if (await_password) {
                        /* пароль проверяет САМА ПРОГРАММА; в модель уходит только вердикт,
                         * дословно как в обучающих данных build_commands() */
                        await_password = 0;
                        if (CmdCheckPassword(user_prompt)) {
                            AsciiSPrint(user_prompt, INPUT_MAX, "%a", "(пароль верный)");
                        } else {
                            AsciiSPrint(user_prompt, INPUT_MAX, "%a", "(пароль неверный)");
                        }
                    } else if (AsciiStrCmp(user_prompt, "/exit") == 0) {
                        break;
                    } else if (AsciiStrnCmp(user_prompt, "/temp", 5) == 0
                               && (user_prompt[5] == ' ' || user_prompt[5] == '\0')) {
                        float tv = LmAtof(user_prompt + 5);
                        if (tv >= 0.0f && tv <= 2.0f) sampler->temperature = tv;
                        AsciiSPrint(status_line, sizeof(status_line), "[temperature = %d.%02d]\r\n",
                                    (INT32)tv, (INT32)(tv * 100.0f) % 100);
                        ChatOut(status_line);
                        continue;
                    } else if (AsciiStrnCmp(user_prompt, "/topp", 5) == 0
                               && (user_prompt[5] == ' ' || user_prompt[5] == '\0')) {
                        float tv = LmAtof(user_prompt + 5);
                        if (tv >= 0.0f && tv <= 1.0f) sampler->topp = tv;
                        AsciiSPrint(status_line, sizeof(status_line), "[top-p = %d.%02d]\r\n",
                                    (INT32)tv, (INT32)(tv * 100.0f) % 100);
                        ChatOut(status_line);
                        continue;
                    } else if (AsciiStrnCmp(user_prompt, "/rep", 4) == 0
                               && (user_prompt[4] == ' ' || user_prompt[4] == '\0')) {
                        float tv = LmAtof(user_prompt + 4);
                        if (tv >= 1.0f && tv <= 2.0f) rep_penalty = tv;
                        AsciiSPrint(status_line, sizeof(status_line), "[repetition penalty = %d.%02d]\r\n",
                                    (INT32)tv, (INT32)(tv * 100.0f) % 100);
                        ChatOut(status_line);
                        continue;
                    } else if (AsciiStrCmp(user_prompt, "/status") == 0) {
                        AsciiSPrint(status_line, sizeof(status_line),
                            "[temp=%d.%02d topp=%d.%02d rep=%d.%02d pos=%d/%d workers=%d]\r\n",
                            (INT32)sampler->temperature, (INT32)(sampler->temperature * 100.0f) % 100,
                            (INT32)sampler->topp, (INT32)(sampler->topp * 100.0f) % 100,
                            (INT32)rep_penalty, (INT32)(rep_penalty * 100.0f) % 100,
                            pos, seq_len, (INT32)MmWorkerCount());
                        ChatOut(status_line);
                        continue;
                    }
                }
                MmEnsureStarted();   /* пул AP спал между ходами -- будим к генерации */
                /* мало места под вопрос + ответ -> скользящая пересборка окна */
                INT8 sys_in_question = pending_with_sys;
                pending_with_sys = 0;
                if (pos + (INT32)AsciiStrLen(user_prompt) + ans_cap + 16 >= seq_len) {
                    INT8 sys_ok = 0;
                    pos = RebuildContext(transformer, tokenizer, hist, hist_next,
                                         hist_count, rendered_prompt, prompt_tokens,
                                         &sys_ok);
                    if (!sys_ok) sys_in_question = 1;
                }
                AsciiSPrint(last_user, INPUT_MAX, "%a", user_prompt);
                if (sys_in_question) {
                    /* история не поместилась (маленькое окно): хотя бы SYS-персона в вопрос */
                    AsciiSPrint(rendered_prompt, RENDER_MAX,
                        "[INST] <<SYS>>\n%a\n<</SYS>>\n\n%a [/INST]",
                        CHAT_SYSTEM_PROMPT, user_prompt);
                } else {
                    AsciiSPrint(rendered_prompt, RENDER_MAX, "[INST] %a [/INST]", user_prompt);
                }
            }
            Encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            gen_count = 0;
            rep_count = 0;
            rep_head = 0;
            cur_answer[0] = '\0';
            CmdFilterInit(&cf, cur_answer, ANSWER_MAX * 2);
            ChatOut("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
            gen_count++;
        }
        if (token == 2) user_turn = 1;   /* EOS ends the assistant turn */

        {
            float* logits = Forward(transformer, token, pos);
            if (user_idx >= num_prompt_tokens && rep_penalty != 1.0f) {
                ApplyRepetitionPenalty(logits, transformer->config.vocab_size,
                                       rep_window, rep_count, rep_penalty);
            }
            next = Sample(sampler, logits);
        }
        pos++;
        if (user_idx >= num_prompt_tokens) {
            rep_window[rep_head] = next;
            rep_head = (rep_head + 1) % REP_WINDOW;
            if (rep_count < REP_WINDOW) rep_count++;
        }

        if (user_idx >= num_prompt_tokens && next != 2) {
            /* лимит на ответ или конец окна: принудительно завершаем ход EOS-ом */
            if (gen_count >= ans_cap || pos >= seq_len - 2) next = 2;
        }

        /* token == 2: это шаг прогона EOS между ходами — сэмпл после него
         * (обычно BOS "<s>") не печатаем и не учитываем */
        if (user_idx >= num_prompt_tokens && next != 2 && token != 2) {
            CHAR8* piece = Decode(tokenizer, token, next);
            /* печатает текст без <CMD>-маркеров, копит его в cur_answer (для
             * истории) и запоминает имя перехваченной команды */
            CmdFilterFeed(&cf, piece);
        }
        if (user_idx >= num_prompt_tokens && next == 2 && token != 2) {
            CmdFilterFlush(&cf);
            ChatOut("\r\n");
            HistPush(hist, &hist_next, &hist_count, last_user, cur_answer);
            turn_index++;
            MmShutdown();   /* ответ закончен: AP-ядра спят до следующего хода */
            /* обработка перехваченной <CMD>-команды (белый список) */
            if (cf.HaveCmd) {
                if (AsciiStrCmp(cf.Cmd, "exit") == 0) {
                    exit_requested = 1;              /* Злобыча достали -- выходим */
                } else if (AsciiStrCmp(cf.Cmd, "askpass") == 0) {
                    await_password = 1;              /* следующий ввод -- пароль */
                } else if (AsciiStrCmp(cf.Cmd, "sysinfo") == 0) {
                    CmdGatherConfig(pending, INPUT_MAX);
                    have_pending = 1;                /* конфиг уйдёт служебным ходом */
                } else if (AsciiStrCmp(cf.Cmd, "openfile") == 0) {
                    CmdRequestLaunch();              /* после верного пароля: запуск LAUNCH_FILE */
                    exit_requested = 1;              /* чат закрывается; запуск -- после teardown */
                }
                /* прочие имена команд игнорируются */
            }
#if SYSINFO_AUTOSTART
            else if (turn_index == 1 && !have_pending) {
                /* после приветствия автоматически передаём конфиг ПК:
                 * Злобыч сам оценит железо, на котором его запустили */
                CmdGatherConfig(pending, INPUT_MAX);
                have_pending = 1;
                pending_with_sys = 1;  /* в обучении автостарт-конфиг идёт с <<SYS>> */
            }
#endif
            if (exit_requested) break;
        }
    }
    ChatOut("\r\n");

    LmFree(user_prompt);
    LmFree(rendered_prompt);
    LmFree(prompt_tokens);
    LmFree(hist);
    LmFree(last_user);
    LmFree(cur_answer);
    LmFree(pending);
}

/* ------------------------------------------------------------------------
 * Entry point
 * ------------------------------------------------------------------------ */
EFI_STATUS EFIAPI UefiMain(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE* SystemTable) {
    DbgInit();
    DBG("LlamaChat: start");

    Print(L"LlamaChat -- mini LLM chat in UEFI (llama2.c port)\r\n");

    /* locate the FS we were loaded from */
    EFI_LOADED_IMAGE_PROTOCOL* LoadedImage = NULL;
    EFI_STATUS s = gBS->HandleProtocol(ImageHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
    if (EFI_ERROR(s)) {
        Print(L"error: LoadedImage protocol: %r\r\n", s);
        DBG_HEX("UefiMain: LoadedImage FAILED", s);
        return s;
    }
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL* Fs = NULL;
    s = gBS->HandleProtocol(LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid, (VOID**)&Fs);
    if (EFI_ERROR(s)) {
        Print(L"error: SimpleFileSystem protocol: %r\r\n", s);
        DBG_HEX("UefiMain: SimpleFileSystem FAILED", s);
        return s;
    }
    EFI_FILE_PROTOCOL* Root = NULL;
    s = Fs->OpenVolume(Fs, &Root);
    if (EFI_ERROR(s)) {
        Print(L"error: OpenVolume: %r\r\n", s);
        DBG_HEX("UefiMain: OpenVolume FAILED", s);
        return s;
    }
    CmdInit(Root, LoadedImage->DeviceHandle);   /* том + устройство для <CMD>openfile</CMD> */

    /* Optional on-disk debug log. Disabled for normal runs.
     * To restore it, uncomment this block and DbgCloseFile() below.
     *
     * s = DbgInitFile(Root, DEBUG_LOG_FILE);
     * if (EFI_ERROR(s))
     *     Print(L"warning: cannot create %s (%r); debugcon only\r\n", DEBUG_LOG_FILE, s);
     */

    /* load model + tokenizer */
    Print(L"loading %s ...\r\n", MODEL_FILE);
    DBG("UefiMain: loading model file");
    VOID* ModelBuf = NULL; UINTN ModelSize = 0;
    s = ReadWholeFile(Root, MODEL_FILE, &ModelBuf, &ModelSize);
    if (EFI_ERROR(s)) {
        Print(L"error: cannot read %s (%r). Put the llama2.c checkpoint next to the app.\r\n", MODEL_FILE, s);
        return s;
    }
    Print(L"loading %s ...\r\n", TOKENIZER_FILE);
    DBG("UefiMain: loading tokenizer file");
    VOID* TokBuf = NULL; UINTN TokSize = 0;
    s = ReadWholeFile(Root, TOKENIZER_FILE, &TokBuf, &TokSize);
    if (EFI_ERROR(s)) {
        Print(L"error: cannot read %s (%r). Take tokenizer.bin from the llama2.c repo.\r\n", TOKENIZER_FILE, s);
        LmFree(ModelBuf);
        return s;
    }
    DBG_DEC("UefiMain: tokenizer file size", TokSize);

    /* build everything */
    Transformer transformer;
    SetMem(&transformer, sizeof(transformer), 0);
    s = BuildTransformer(&transformer, ModelBuf, ModelSize);
    if (EFI_ERROR(s)) {
        Print(L"error: bad model file (%r)\r\n", s);
        LmFree(ModelBuf); LmFree(TokBuf);
        return s;
    }
    DBG("UefiMain: transformer built");

    Tokenizer tokenizer;
    SetMem(&tokenizer, sizeof(tokenizer), 0);
    DBG("UefiMain: BuildTokenizer begin");
    s = BuildTokenizer(&tokenizer, TokBuf, TokSize, transformer.config.vocab_size);
    if (EFI_ERROR(s)) {
        Print(L"error: bad tokenizer file (%r)\r\n", s);
        FreeTransformer(&transformer);
        LmFree(TokBuf);
        return s;
    }
    DBG("UefiMain: BuildTokenizer complete");

    Sampler sampler;
    BuildSampler(&sampler, transformer.config.vocab_size, DEFAULT_TEMP, DEFAULT_TOPP, AsmReadTsc());
    DBG("UefiMain: sampler built");
    /* DbgCloseFile(); */ /* restore together with DbgInitFile() above */

    /* Multi-core init. On some old firmware (e.g. Kabini-era AMD boards
     * with Aptio 4) the MP Services calls can hang INSIDE the firmware,
     * and a single-core app cannot time out a blocking firmware call.
     * Escape hatch: create an empty file "nomp.txt" next to the app to
     * skip MP entirely. MmInit also prints each step to the console so a
     * hang on real hardware pinpoints the failing firmware call. */
    if (FileExists(Root, L"nomp.txt")) {
        Print(L"nomp.txt found -- multicore disabled, using BSP only\r\n");
        DBG("UefiMain: nomp.txt present -- skipping MmInit");
    } else {
        Print(L"init multicore... (if this hangs, create an empty 'nomp.txt' next to the app)\r\n");
        MmInit();
    }

    EFI_STATUS UiStatus = UiInit();
    if (EFI_ERROR(UiStatus)) {
        Print(L"warning: EFI text console initialization failed (%r)\r\n", UiStatus);
    }

    CHAR8 InfoLine[160];
    AsciiSPrint(InfoLine, sizeof(InfoLine), "model: dim=%d layers=%d heads=%d vocab=%d seq_len=%d\r\n",
                transformer.config.dim, transformer.config.n_layers,
                transformer.config.n_heads, transformer.config.vocab_size,
                transformer.config.seq_len);
    ChatOut(InfoLine);
    AsciiSPrint(InfoLine, sizeof(InfoLine), "workers: %d core(s)\r\n", (UINT32)MmWorkerCount());
    ChatOut(InfoLine);

    /* только чат: модель сама открывает диалог из встроенного промпта */
    ChatOut("\r\n/exit, /temp X, /topp X, /rep X, /status. F2 — раскладка EN/RU.\r\n\r\n");
    Chat(&transformer, &tokenizer, &sampler);

    /* teardown */
    MmShutdown();
    FreeSampler(&sampler);
    FreeTokenizer(&tokenizer);
    FreeTransformer(&transformer);   /* frees ModelBuf via t->data */
    LmFree(TokBuf);

    /* <CMD>openfile</CMD>: запуск другого EFI-приложения (LAUNCH_FILE, см. LlamaCmd.h).
     * Только ПОСЛЕ teardown: модель и KV-кэш освобождены, MP-воркеры остановлены. */
    if (CmdLaunchRequested()) {
        Print(L"launching %s ...\r\n", LAUNCH_FILE);
        EFI_STATUS Ls = CmdLaunchApp();
        if (EFI_ERROR(Ls) && Ls != EFI_NOT_FOUND) {
            Print(L"error: cannot start %s (%r)\r\n", LAUNCH_FILE, Ls);
        }
    }

    DBG("LlamaChat: exit");
    ChatOut("bye\r\n");
    return EFI_SUCCESS;
}
