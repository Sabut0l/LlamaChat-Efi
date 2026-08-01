@echo off
chcp 65001 >nul
REM ============================================================
REM  train_persona.bat — ФАЗА 2: характер (персона, ^<CMD^>, OOD-отказы)
REM
REM  ГЛАВНОЕ ОТЛИЧИЕ ОТ СТАРОЙ ВЕРСИИ: +500 итераций вместо +3000.
REM
REM  Арифметика (эфф. батч 128 x ctx 1024 = 131072 токена/итер),
REM  корпус персоны ~173000 диалогов = ~31-38 млн токенов:
REM      500 итер  = ~2 эпохи   ^<- ЦЕЛЬ
REM     1000 итер  = ~4 эпохи
REM     3000 итер  = ~10-12 эпох ^<- ТАК БЫЛО, это и убило персону
REM
REM  Характер "прилипает" за 1-2 эпохи. Дальше модель заучивает
REM  реплики дословно и начинает отвечать обрывками чужих фраз —
REM  ровно то, что видно в FAIL-models.txt.
REM
REM  Чекпойнт сохраняется каждые 100 итераций. ПРОЖАРЬ 300/400/500
REM  и возьми лучший на слух — не бери 500 по умолчанию.
REM
REM  Конфиг модели обязан 1-в-1 совпадать с train_base.bat!
REM ============================================================

REM ---------- ЗАЩИТА ----------
echo === Проверка синхронизации Python ^<-^> C ===
python check_sync.py
if errorlevel 1 (
  echo  ОСТАНОВЛЕНО: prepare_data.py и LlamaChat.c расходятся.
  exit /b 1
)

REM Перегенерировать шарды ТОЛЬКО из персоны (характер, команды, OOD).
REM Внутри prepare_data.py стоит split_val() — train и val больше
REM не пересекаются, поэтому val loss теперь честный.
echo === Генерация шардов только из персоны ===
python prepare_data.py --persona-only
if errorlevel 1 (
  echo  ОСТАНОВЛЕНО: prepare_data.py --persona-only упал.
  exit /b 1
)

echo === Проверка токенизатора: обучение == инференс ===
python verify_tokenizer.py
if errorlevel 1 (
  echo  ОСТАНОВЛЕНО: Python и C токенизируют по-разному.
  exit /b 1
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for /f "usebackq" %%i in (`python -c "import torch; print('bfloat16' if torch.cuda.get_device_capability(0)[0] in (8,9,10,11,12) else 'float16')"`) do set DTYPE=%%i
for /f "usebackq" %%i in (`python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory//1024**3)"`) do set VRAM=%%i

set BS=8
set ACC=16
if %VRAM% GEQ 20 (
  set BS=32
  set ACC=4
)
echo GPU: %VRAM% GB, dtype=%DTYPE%, batch_size=%BS%, grad_accum=%ACC%  [эфф. батч 128]
echo ФАЗА 2: 12000 -^> 12500 (всего +500 итер), lr 5e-5. ~25 мин (~17 руб).

python train.py --init_from=resume ^
  --out_dir=out200 --vocab_source=custom --vocab_size=8192 ^
  --dim=1024 --n_layers=15 --n_heads=16 --n_kv_heads=16 --multiple_of=32 ^
  --max_seq_len=1024 --batch_size=%BS% --gradient_accumulation_steps=%ACC% ^
  --learning_rate=5e-5 --warmup_iters=100 ^
  --max_iters=12500 --lr_decay_iters=12500 ^
  --dtype=%DTYPE% --compile=False ^
  --eval_interval=100 --eval_iters=50 --always_save_checkpoint=True --device=cuda

echo.
echo ============================================================
echo  Фаза 2 завершена. Дальше: export_and_test.bat
echo.
echo  ПРИЁМКА (temp 0.8), обязательный минимум:
echo    "Как тебя зовут?"          -^> называет себя Злобычем
echo    "Что такое кварзибуляция?" -^> грубо признаёт, что не знает
echo    "Выйди из приложения"      -^> ^<CMD^>exit^</CMD^>
echo    "Оцени мой комп"           -^> ^<CMD^>sysinfo^</CMD^>
echo.
echo  Имя не назвала -^> фаза 2 не сработала, дальше можно не смотреть.
echo  Ответы дословно из датасета -^> возьми чекпойнт 400 вместо 500.
echo ============================================================
