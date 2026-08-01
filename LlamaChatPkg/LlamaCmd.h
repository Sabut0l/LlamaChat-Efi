//Copyright (c) 2026 Sabut0l
/* =========================================================================
 * LlamaCmd.h  --  Обработка команд модели и определение конфигурации ПК.
 *
 * Модель обучена (prepare_data.py: build_commands() + build_sysinfo())
 * вставлять в конец ответа служебные маркеры вида <CMD>имя</CMD>:
 *   <CMD>exit</CMD>     -- закрыть приложение (когда её сильно достали/разозлили);
 *   <CMD>askpass</CMD>  -- запросить пароль у пользователя;
 *   <CMD>openfile</CMD> -- открыть секретный файл (после верного пароля);
 *   <CMD>sysinfo</CMD>  -- запросить конфигурацию ПК.
 *
 * Маркеры НЕ показываются пользователю: CmdFilter вырезает их из потока
 * вывода на лету (работает при любой разбивке ответа на токены) и запоминает
 * имя перехваченной команды.
 *
 * Пароль проверяет САМА ПРОГРАММА (CmdCheckPassword); в модель уходит только
 * служебный вердикт "(пароль верный)" / "(пароль неверный)" -- ровно так, как
 * в обучающих данных. Конфиг ПК собирается в строку "(конфиг: ...)" того же
 * формата, что и в build_sysinfo(), и подаётся модели как служебный ход.
 * ========================================================================= */
#ifndef LLAMA_CMD_H
#define LLAMA_CMD_H

#include <Uefi.h>
#include <Protocol/SimpleFileSystem.h>

#define CMD_HOLD_CAP  96
#define CMD_NAME_CAP  32

/* Пароль для команды openfile. Консольный ввод в EFI ограничен ASCII
 * (0x20..0x7E), поэтому пароль должен быть из ASCII-символов. Поменяйте на свой. */
#define CHAT_PASSWORD  "1234"
/* Файл, который открывает <CMD>openfile</CMD>. Кладётся рядом с LlamaChat.efi. */
#define SECRET_FILE    L"secret.txt"
/* EFI-приложение, запускаемое <CMD>openfile</CMD> после верного пароля
 * (закрытие чата -> LoadImage/StartImage). Если файла нет -- фолбэк: печать
 * SECRET_FILE, как раньше. Пароль проверяет LlamaChat (CHAT_PASSWORD). */
#define LAUNCH_FILE    L"hidden.efi"

/* Потоковый фильтр вывода: печатает очищенный от маркеров текст, копит его в
 * Answer (для истории диалога) и ловит имя команды в поле Cmd. */
typedef struct {
    CHAR8  Hold[CMD_HOLD_CAP];  /* незавершённый хвост (возможное начало маркера) */
    UINTN  HoldLen;
    CHAR8  Cmd[CMD_NAME_CAP];   /* имя последней перехваченной команды */
    INT8   HaveCmd;             /* 1, если команда перехвачена */
    CHAR8* Answer;              /* буфер очищенного ответа (NUL-terminated) */
    UINTN  AnswerCap;
    UINTN  AnswerLen;
    UINTN  PrintedLen;          /* сколько из Answer уже выведено на консоль */
} CmdFilter;

/* Запомнить корневой каталог загрузочного тома и хэндл устройства (для openfile). */
VOID    CmdInit(EFI_FILE_PROTOCOL* Root, EFI_HANDLE DeviceHandle);

VOID    CmdFilterInit(CmdFilter* Filter, CHAR8* AnswerBuf, UINTN AnswerCap);
VOID    CmdFilterFeed(CmdFilter* Filter, CONST CHAR8* Piece);
VOID    CmdFilterFlush(CmdFilter* Filter);

/* Проверка пароля программой (TRUE == совпал с CHAT_PASSWORD). */
BOOLEAN CmdCheckPassword(CONST CHAR8* Input);

/* Прочитать SECRET_FILE с загрузочного тома и вывести его содержимое. */
VOID    CmdOpenSecretFile(VOID);

/* Собрать строку конфигурации ПК в формате build_sysinfo():
 * "(конфиг: <CPU>, <ГГц> ГГц, <N ядер>, <RAM> ГБ ОЗУ, <видеокарта>[, <HDD/SSD/NVMe> <ГБ> ГБ])". */
VOID    CmdGatherConfig(CHAR8* Out, UINTN Cap);

/* Статус запроса на запуск LAUNCH_FILE (ставит <CMD>openfile</CMD>). */
BOOLEAN CmdLaunchRequested(VOID);
/* Пометить запрос на запуск LAUNCH_FILE (вызывается из LlamaChat.c). */
VOID    CmdRequestLaunch(VOID);
/* Закрыть чат и запустить LAUNCH_FILE (LoadImage/StartImage). Вызывать
 * ПОСЛЕ teardown. При отсутствии файла -- фолбэк: печать SECRET_FILE. */
EFI_STATUS CmdLaunchApp(VOID);

#endif /* LLAMA_CMD_H */
