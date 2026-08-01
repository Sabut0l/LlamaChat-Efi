# -*- coding: utf-8 -*-
#Copyright (c) 2026 Sabut0l
"""
verify_tokenizer.py — доказывает, что ТОКЕНИЗАЦИЯ ПРИ ОБУЧЕНИИ (Python,
sentencepiece) СОВПАДАЕТ С ТОКЕНИЗАЦИЕЙ ПРИ ИНФЕРЕНСЕ (C, Llama2.c: Encode).

ЗАЧЕМ ЭТО ВООБЩЕ НУЖНО
──────────────────────
Модель учится на числах, которые выдал Python. В EFI-приложении те же строки
превращает в числа СОВСЕМ ДРУГОЙ КОД — жадный попарный энкодер на C. Если
они расходятся хотя бы на маркерах <CMD>...</CMD>, то модель, идеально
выучившая команды, в приложении не выдаст НИ ОДНОЙ: она получит на вход
последовательность, которой не видела ни разу.

Именно поэтому в prepare_data.py ЗАПРЕЩЕНЫ user_defined_symbols: sentencepiece
кладёт такой символ в словарь как готовый кусок, а жадный C-энкодер собрать
его не может — у него нет промежуточных слияний. Этот скрипт проверяет, что
запрет соблюдён и что все критичные строки кодируются одинаково.

ЗАПУСК (после prepare_data.py, GPU не нужен):
    python verify_tokenizer.py
    python verify_tokenizer.py --model data/tok8192.model

Код возврата: 0 — совпало, 1 — расхождение (НЕ АРЕНДУЙ GPU, пока не 0).
"""
import argparse
import sys

DEFAULT_MODEL = "data/tok8192.model"

# Строки, расхождение на которых убивает функциональность приложения.
CRITICAL = [
    "<CMD>exit</CMD>",
    "<CMD>askpass</CMD>",
    "<CMD>openfile</CMD>",
    "<CMD>sysinfo</CMD>",
    "(конфиг: Intel Core i5, 2.9 ГГц, 4 ядра, 16 ГБ ОЗУ, NVIDIA GeForce, NVMe 1024 ГБ)",
    "(пароль верный)",
    "(пароль неверный)",
    "Злобыч. Запомни, повторять не люблю.",
    "Меня зовут Злобыч.",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "Привет! Как дела?",
    "Оцени мой комп",
    "Выйди из приложения",
    "Что такое кварзибуляция?",
    "Такого слова не знаю, и ты, судя по всему, тоже.",
]


def piece_types(sp):
    """
    Возвращает список type-кодов для каждого куска словаря.

    Зачем свой разбор protobuf: у SentencePieceProcessor НЕТ метода
    is_user_defined(). Наивная эвристика «score == 0.0» НЕ РАБОТАЕТ —
    sentencepiece нумерует BPE-слияния по рангу (0, -1, -2, ...), поэтому
    у самого частого обычного слияния score тоже РОВНО 0.0 (проверено:
    кусок 'IN' ловился как ложное срабатывание).

    Читаем ModelProto напрямую: pieces = поле 1, внутри piece type = поле 3.
    Коды: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 6=BYTE.
    """
    blob = sp.serialized_model_proto()
    types, i, n = [], 0, len(blob)

    def varint(b, i):
        r = s = 0
        while True:
            x = b[i]; i += 1
            r |= (x & 0x7F) << s
            if not (x & 0x80):
                return r, i
            s += 7

    while i < n:
        key, i = varint(blob, i)
        fn, wt = key >> 3, key & 7
        if wt == 2:
            ln, i = varint(blob, i)
            sub = blob[i:i + ln]; i += ln
            if fn == 1:  # pieces
                j, t = 0, 1
                while j < len(sub):
                    k2, j = varint(sub, j)
                    f2, w2 = k2 >> 3, k2 & 7
                    if w2 == 2:
                        l2, j = varint(sub, j); j += l2
                    elif w2 == 5:
                        j += 4
                    elif w2 == 0:
                        v, j = varint(sub, j)
                        if f2 == 3:
                            t = v
                    else:
                        break
                types.append(t)
        elif wt == 0:
            _, i = varint(blob, i)
        elif wt == 5:
            i += 4
        else:
            break
    return types


def c_encode(sp, text):
    """
    Побитовая реимплементация Encode() из LlamaChatPkg_v12/Llama2.c.

    Шаг 1: dummy-префикс " " (как в референсном llama2.c).
    Шаг 2: каждый UTF-8 символ -> id из словаря; если символа нет —
           byte fallback: каждый байт -> id (byte + 3).
    Шаг 3: жадное попарное слияние — пока есть соседняя пара, чья склейка
           есть в словаре, сливаем пару с НАИЛУЧШИМ score.

    Ключевой момент: шаг 3 может собрать токен ТОЛЬКО если в словаре лежат
    все промежуточные склейки. Это и есть причина запрета user_defined_symbols.
    """
    # словарь: строка куска -> (id, score)
    vocab = {}
    for i in range(sp.get_piece_size()):
        vocab[sp.id_to_piece(i)] = (i, sp.get_score(i))

    tokens = []
    # llama2.c добавляет ведущий пробел; sentencepiece кодирует его как U+2581
    s = " " + text
    for ch in s:
        piece = ch.replace(" ", "▁")
        if piece in vocab:
            tokens.append(vocab[piece][0])
        else:
            # byte fallback: <0xXX> лежат в словаре по индексам 3..258
            for b in ch.encode("utf-8"):
                tokens.append(b + 3)

    # жадное попарное слияние
    while True:
        best_score = -1e10
        best_id = -1
        best_idx = -1
        for i in range(len(tokens) - 1):
            merged = sp.id_to_piece(tokens[i]) + sp.id_to_piece(tokens[i + 1])
            ent = vocab.get(merged)
            if ent is not None and ent[1] > best_score:
                best_score = ent[1]
                best_id = ent[0]
                best_idx = i
        if best_idx == -1:
            break
        tokens[best_idx] = best_id
        del tokens[best_idx + 1]

    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="путь к tok8192.model (по умолчанию %s)" % DEFAULT_MODEL)
    args = ap.parse_args()

    try:
        from sentencepiece import SentencePieceProcessor
    except ImportError:
        print("[!] нет sentencepiece: pip install sentencepiece")
        return 1

    try:
        sp = SentencePieceProcessor(model_file=args.model)
    except Exception as e:
        print("[!] не открылся %s: %s" % (args.model, e))
        print("    сначала запусти prepare_data.py")
        return 1

    print("=" * 72)
    print("ПРОВЕРКА: обучение (Python) == инференс (C, жадное слияние)")
    print("словарь: %d токенов, модель: %s" % (sp.get_piece_size(), args.model))
    print("=" * 72)

    # ── Проверка 1: нет ли в словаре недостижимых user_defined_symbols ──
    # Определяем ТОЧНО, по type-коду из ModelProto (4 = USER_DEFINED).
    # Эвристика по score==0.0 давала ложные срабатывания — см. piece_types().
    try:
        types = piece_types(sp)
    except Exception as e:
        print("[!] не разобрался ModelProto (%s) — проверка 1 пропущена" % e)
        types = []
    suspicious = [(i, sp.id_to_piece(i))
                  for i, t in enumerate(types) if t == 4]
    if suspicious:
        print("\n[!] НАЙДЕНЫ USER_DEFINED символы (type=4) — жадный C-энкодер")
        print("    их НЕ СОБЕРЁТ. Убери user_defined_symbols из prepare_data.py:")
        for i, p in suspicious[:20]:
            print("      id=%-6d %r" % (i, p))
        if len(suspicious) > 20:
            print("      ... ещё %d" % (len(suspicious) - 20))
    else:
        print("\n[ok] user_defined_symbols не найдены — словарь чисто BPE")

    # ── Проверка 2: посимвольное сравнение Python vs C на критичных строках ──
    print("\n%-52s %-6s %-6s %s" % ("строка", "py", "c", "вердикт"))
    print("-" * 78)
    bad = 0
    for text in CRITICAL:
        py = sp.encode(text)
        c = c_encode(sp, text)
        ok = (py == c)
        if not ok:
            bad += 1
        label = text if len(text) <= 50 else text[:47] + "..."
        print("%-52s %-6d %-6d %s" % (label, len(py), len(c),
                                      "ok" if ok else "РАСХОЖДЕНИЕ"))
        if not ok:
            print("      python: %s" % py)
            print("      c     : %s" % c)
            print("      py->   %r" % [sp.id_to_piece(t) for t in py])
            print("      c ->   %r" % [sp.id_to_piece(t) for t in c])

    print("-" * 78)

    # ── Проверка 3: маркеры не должны разваливаться на байты ──
    print("\nстоимость маркеров в токенах (чем меньше, тем легче выучить):")
    for m in ("<CMD>exit</CMD>", "<CMD>askpass</CMD>",
              "<CMD>openfile</CMD>", "<CMD>sysinfo</CMD>", "Злобыч"):
        ids = sp.encode(m)
        pieces = [sp.id_to_piece(t) for t in ids]
        nbyte = sum(1 for p in pieces if p.startswith("<0x"))
        warn = "  <- МНОГО BYTE-FALLBACK" if nbyte > len(ids) // 2 else ""
        print("  %-22s %2d токенов, byte-fallback %d%s" % (m, len(ids), nbyte, warn))
        print("      %r" % pieces)

    print()
    if bad or suspicious:
        print("!" * 72)
        print("ПРОВАЛ: расхождений %d, подозрительных символов %d" % (bad, len(suspicious)))
        print("НЕ АРЕНДУЙ GPU. Модель выучит одно, а приложение подаст другое.")
        print("!" * 72)
        return 1

    print("=" * 72)
    print("OK: обучение и инференс токенизируют одинаково. Можно обучать.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
