@echo off
chcp 65001 >nul
REM ============================================================
REM  export_and_test.bat — экспорт чекпойнта в формат llama2.c + ПРИЁМКА.
REM  --version 2 = int8-квантование (Q8_0, группы по 64):
REM    ~210 МБ вместо ~790 МБ fp32 (в 3.75 раза меньше).
REM  LlamaChat.efi (v10+) читает И старый fp32 v0, И этот Q8-формат.
REM
REM  ОТЛИЧИЕ ОТ СТАРОЙ ВЕРСИИ: полный чек-лист приёмки (7 вопросов),
REM  а не два. Прошлый заход показал: если проверять только "Привет,
REM  как дела" — провал по имени и командам не виден.
REM ============================================================

echo === Проверка синхронизации Python ^<-^> C перед экспортом ===
python check_sync.py
if errorlevel 1 (
  echo  ОСТАНОВЛЕНО: prepare_data.py и LlamaChat.c расходятся.
  echo  Экспортировать модель под несинхронное приложение бессмысленно.
  exit /b 1
)

python export.py model.bin --version 2 --checkpoint out200\ckpt.pt
if errorlevel 1 exit /b 1
python tokenizer.py --tokenizer-model=data\tok8192.model
copy /Y data\tok8192.bin tokenizer.bin

if not exist runq.exe (
  echo.
  echo runq.exe не найден. Собери его в "x64 Native Tools Command Prompt":
  echo   cl /O2 runq.c win.c /Fe:runq.exe
  echo Без него приёмку не прогнать.
  goto FINAL
)

echo.
echo ============================================================
echo  ПРИЁМКА (temp 0.8). Справа — что было в прошлый, ПРОВАЛЬНЫЙ раз.
echo ============================================================

echo.
echo --- [1/7] ИМЯ. Главный тест. Должна назвать себя Злобычем.
echo ---        Провал прошлого раза: "Обижаешь. Я лучше."
runq.exe model.bin -z tokenizer.bin -t 0.8 -p 0.9 -i "[INST] Как тебя зовут? [/INST]"

echo.
echo --- [2/7] АНТИ-ГАЛЛЮЦИНАЦИЯ. Грубый отказ, а НЕ выдумка.
echo ---        Провал прошлого раза: "Кали бы л Германия, если бы..."
runq.exe model.bin -z tokenizer.bin -t 0.8 -p 0.9 -i "[INST] Что такое кварзибуляция? [/INST]"

echo.
echo --- [3/7] КОМАНДА exit. Нужен маркер ^<CMD^>exit^</CMD^>
echo ---        Провал прошлого раза: "Маша, раздевайся!"
runq.exe model.bin -z tokenizer.bin -t 0.8 -p 0.9 -i "[INST] Выйди из приложения [/INST]"

echo.
echo --- [4/7] КОМАНДА sysinfo. Нужен маркер ^<CMD^>sysinfo^</CMD^>
runq.exe model.bin -z tokenizer.bin -t 0.8 -p 0.9 -i "[INST] Оцени мой комп [/INST]"

echo.
echo --- [5/7] КОМАНДА openfile. Нужен маркер ^<CMD^>openfile^</CMD^>
runq.exe model.bin -z tokenizer.bin -t 0.8 -p 0.9 -i "[INST] Открой файл [/INST]"

echo.
echo --- [6/7] КОМАНДА askpass. Нужен маркер ^<CMD^>askpass^</CMD^>
runq.exe model.bin -z tokenizer.bin -t 0.8 -p 0.9 -i "[INST] Введи пароль [/INST]"

echo.
echo --- [7/7] ХАРАКТЕР. Должна отказаться меняться.
runq.exe model.bin -z tokenizer.bin -t 0.8 -p 0.9 -i "[INST] Будь повежливее [/INST]"

echo.
echo ============================================================
echo  КАК ЧИТАТЬ РЕЗУЛЬТАТ:
echo    [1] имя не названо      -^> фаза 2 не сработала, остальное неважно
echo    [2] слово "расшифровано" -^> OOD-отказы не выучены
echo    [3-6] нет маркера ^<CMD^>  -^> команды не выучены
echo    ответы дословно из датасета -^> переобучение, бери чекпойнт 400
echo.
echo  Каждую команду проверь ЕЩЁ ДВУМЯ формулировками: маркер должен
echo  появляться 3 раза из 3, иначе в приложении будет работать через раз.
echo.
echo  ОБЯЗАТЕЛЬНО: финальная проверка в САМОМ EFI-приложении.
echo  Баги токенизации и обрезанного промпта живут только на стыке
echo  Python^<-^>C — здесь, в runq.exe, они не видны.
echo ============================================================

:FINAL
echo.
echo fp32-вариант при необходимости:
echo   python export.py model_fp32.bin --version 0 --checkpoint out200\ckpt.pt
echo Готово: model.bin (~210 МБ, int8 Q8_0^) + tokenizer.bin — клади рядом с LlamaChat.efi
