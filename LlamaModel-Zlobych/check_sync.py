# -*- coding: utf-8 -*-
"""
check_sync.py — ГЛАВНАЯ ЗАЩИТА ОТ ПОВТОРА ПРОВАЛА.

Проверяет, что обучение и инференс говорят на одном языке. Запускать
ОБЯЗАТЕЛЬНО перед арендой GPU и ещё раз перед сборкой .efi.

Что сверяется:
  1. SYS-промпт: prepare_data.py  <->  LlamaChat.c (CHAT_SYSTEM_PROMPT)
     — расхождение хоть в одном символе = модель вне распределения обучения.
  2. Целостность макроса CHAT_SYSTEM_PROMPT (обратные слэши в конце строк).
     Именно этот баг «съел» абзац про анти-галлюцинацию в v12.
  3. Служебные строки: "(пароль верный)", "(пароль неверный)", "(конфиг:"
  4. Белый список команд: <CMD>exit|askpass|openfile|sysinfo</CMD>

Запуск:
    python check_sync.py [путь_к_LlamaChat.c]
Код возврата 0 = всё сходится, 1 = есть расхождения.
"""
import io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, "prepare_data.py")

# Пути к LlamaChat.c. Порядок важен: сначала рядом, потом на уровень выше,
# потом на два уровня выше.
#
# ЗАЧЕМ ДВА УРОВНЯ: setup.bat копирует наши скрипты ВНУТРЬ llama2.c\, то есть
# при обучении check_sync.py лежит в ...\LlamaZlobych_FIXED\llama2.c\, и
# LlamaChatPkg_v12 оказывается уже через ДВА "..". Без этого проверка молча
# не находила бы исходники ровно на том ПК, где идёт обучение.
DEFAULT_C = [
    os.path.join(HERE, "LlamaChat.c"),
    os.path.join(HERE, "..", "LlamaChatPkg_v12", "LlamaChat.c"),
    os.path.join(HERE, "..", "..", "LlamaChatPkg_v12", "LlamaChat.c"),
    os.path.join(HERE, "..", "..", "..", "LlamaChatPkg_v12", "LlamaChat.c"),
]

problems = []
notes = []


def fail(msg):
    problems.append(msg)


def ok(msg):
    notes.append(msg)


def read(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


def sys_from_prepare(src):
    """Достаём SYS = ("..." "...") из prepare_data.py."""
    m = re.search(r'^SYS\s*=\s*\((.*?)\)\s*$', src, re.S | re.M)
    if not m:
        return None
    return "".join(re.findall(r'"([^"]*)"', m.group(1)))


def sys_from_c(src):
    """Достаём CHAT_SYSTEM_PROMPT, честно уважая продолжения строк."""
    lines = src.split("\n")
    start = None
    for i, l in enumerate(lines):
        if l.strip().startswith("#define CHAT_SYSTEM_PROMPT"):
            start = i
            break
    if start is None:
        return None, None
    body, i, truncated = [], start, False
    while i < len(lines):
        line = lines[i].rstrip("\r")
        body.append(line)
        if not line.rstrip().endswith(chr(92)):
            # макрос закончился здесь
            if i < len(lines) - 1:
                nxt = lines[i + 1].strip()
                # следующая строка — «висячий» литерал => продолжение потеряно
                if nxt.startswith('"'):
                    truncated = True
            break
        i += 1
    text = "".join(re.findall(r'"([^"]*)"', "\n".join(body)))
    return text, truncated


def main():
    if not os.path.exists(PREP):
        print("НЕ НАЙДЕН:", PREP)
        return 1
    prep = read(PREP)

    cpath = sys.argv[1] if len(sys.argv) > 1 else None
    if cpath is None:
        for c in DEFAULT_C:
            if os.path.exists(c):
                cpath = c
                break
    if cpath is None or not os.path.exists(cpath):
        print("НЕ НАЙДЕН LlamaChat.c. Укажи путь: python check_sync.py <путь>")
        return 1
    csrc = read(cpath)

    print("prepare_data.py :", PREP)
    print("LlamaChat.c     :", os.path.normpath(cpath))
    print()

    # ── 1) SYS ──
    a = sys_from_prepare(prep)
    b, truncated = sys_from_c(csrc)
    if a is None:
        fail("не удалось разобрать SYS в prepare_data.py")
    elif b is None:
        fail("не удалось найти CHAT_SYSTEM_PROMPT в LlamaChat.c")
    else:
        if truncated:
            fail("МАКРОС CHAT_SYSTEM_PROMPT ОБОРВАН: у одной из строк потерян "
                 "обратный слэш. Часть промпта не попадает в бинарь "
                 "(ровно этот баг был в v12 — строка 57).")
        if a == b:
            ok("SYS совпадает 1-в-1 (%d символов)" % len(a))
        else:
            fail("SYS РАСХОДИТСЯ: обучение %d симв., инференс %d симв." % (len(a), len(b)))
            n = min(len(a), len(b))
            d = next((i for i in range(n) if a[i] != b[i]), n)
            print("    первое расхождение на позиции", d)
            print("    обучение : ..." + repr(a[max(0, d - 40):d + 60]))
            print("    инференс : ..." + repr(b[max(0, d - 40):d + 60]))

    # ── 2) команды и служебные строки (ищем по ВСЕМ исходникам приложения) ──
    cmd_dir = os.path.dirname(os.path.abspath(cpath))
    csrc_all = csrc
    for extra in ("LlamaCmd.c", "LlamaCmd.h"):
        p = os.path.join(cmd_dir, extra)
        if os.path.exists(p):
            csrc_all += read(p)

    for lit in ["(пароль верный)", "(пароль неверный)", "(конфиг:"]:
        in_p, in_c = lit in prep, lit in csrc_all
        if in_p and in_c:
            ok("служебная строка присутствует с обеих сторон: %s" % lit)
        else:
            fail("служебная строка %r: prepare_data=%s приложение=%s"
                 % (lit, in_p, in_c))

    # ── 3) команды ──
    # \w+ по-русски цепляет и заглушку «<CMD>имя</CMD>» из докстрингов —
    # поэтому берём только латиницу и отсеиваем известные плейсхолдеры.
    PLACEHOLDERS = {"имя", "name"}
    cmds_p = set(re.findall(r"<CMD>([a-z]+)</CMD>", prep)) - PLACEHOLDERS
    cmds_c = set(re.findall(r'AsciiStrCmp\(cf\.Cmd,\s*"([a-z]+)"\)', csrc_all))
    cmds_c |= set(re.findall(r"<CMD>([a-z]+)</CMD>", csrc_all))
    cmds_c -= PLACEHOLDERS
    if cmds_p and cmds_c:
        only_p, only_c = cmds_p - cmds_c, cmds_c - cmds_p
        if not only_p and not only_c:
            ok("белый список команд совпадает: %s" % ", ".join(sorted(cmds_p)))
        else:
            if only_p:
                fail("модель учат командам, которых НЕТ в приложении: %s"
                     % ", ".join(sorted(only_p)))
            if only_c:
                fail("приложение ждёт команды, которым модель НЕ учили: %s"
                     % ", ".join(sorted(only_c)))
    else:
        fail("не удалось сопоставить команды (prep=%s, c=%s)" % (sorted(cmds_p), sorted(cmds_c)))

    # ── 4) формат конфига ──
    if "%d.%d ГГц" in csrc_all:
        ok("частота в конфиге печатается как X.Y — в build_sysinfo() держи один знак")

    print()
    for n in notes:
        print("  [ok]  " + n)
    if problems:
        print()
        for p in problems:
            print("  [!!]  " + p)
        print("\nРАСХОЖДЕНИЙ: %d. НЕ АРЕНДУЙ GPU, пока не исправишь." % len(problems))
        return 1
    print("\nВСЁ СХОДИТСЯ. Можно запускать обучение.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
