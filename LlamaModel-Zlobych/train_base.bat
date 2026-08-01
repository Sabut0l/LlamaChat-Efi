@echo off
chcp 65001 >nul
REM ============================================================
REM  train_base.bat — ФАЗА 1: база (язык и грамматика)
REM  Злобыч ~198M: dim=1024, 15 слоёв, 16 голов, контекст 1024.
REM
REM  ГЛАВНОЕ ОТЛИЧИЕ ОТ СТАРОЙ ВЕРСИИ: 12000 итераций вместо 30000.
REM  На 12k пройдено 1.57 млрд токенов — для 198M этого достаточно.
REM  Прошлый заход шёл до 24-27k (~7 эпох) и получил loss ~0.1 —
REM  это не рекорд, а дословное заучивание датасета. Отсюда и ответы
REM  обрывками чужих реплик. Подробности: АЛГОРИТМ_ОБУЧЕНИЯ.md, причина 7.
REM
REM  ОРИЕНТИР ПО LOSS: здоровое значение к концу фазы 1 — 1.8-2.5.
REM  val ниже 1.5 — переобучение, останавливайся.
REM  val ~0.1 — АВАРИЯ (либо утечка train/val, либо заучивание).
REM
REM  РАННЯЯ ОСТАНОВКА (экономит деньги): если val не улучшается
REM  4 замера подряд (1000 итер) — Ctrl+C, иди в train_persona.bat.
REM
REM  Один файл для V100 16GB и для RTX: dtype и батч определяются авто.
REM  Эффективный батч всегда 128 -> чекпойнт переносим между картами.
REM  --compile=False обязательно (на Windows нет Triton).
REM  Обрыв сессии? Тот же запуск + --init_from=resume
REM ============================================================

REM ---------- ЗАЩИТА: без неё не тратим деньги ----------
echo === Проверка синхронизации Python ^<-^> C ===
python check_sync.py
if errorlevel 1 (
  echo.
  echo  ОСТАНОВЛЕНО: prepare_data.py и LlamaChat.c расходятся.
  echo  Обучение на 10 часов и 400 руб не начато. Запусти preflight.bat
  exit /b 1
)

if exist data\tok8192.model (
  echo === Проверка токенизатора: обучение == инференс ===
  python verify_tokenizer.py
  if errorlevel 1 (
    echo.
    echo  ОСТАНОВЛЕНО: Python и C токенизируют по-разному.
    echo  Модель выучила бы команды и не выдала бы НИ ОДНОЙ в EFI.
    exit /b 1
  )
) else (
  echo  ОСТАНОВЛЕНО: нет data\tok8192.model — сначала prepare_data.py
  exit /b 1
)

REM Защита от фрагментации памяти CUDA
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
echo ФАЗА 1: 12000 итераций, lr 3e-4. Ожидаемое время на RTX 3090 ~10 ч (~400 руб).

python train.py ^
  --out_dir=out200 --vocab_source=custom --vocab_size=8192 ^
  --dim=1024 --n_layers=15 --n_heads=16 --n_kv_heads=16 --multiple_of=32 ^
  --max_seq_len=1024 --batch_size=%BS% --gradient_accumulation_steps=%ACC% ^
  --learning_rate=3e-4 --warmup_iters=1000 ^
  --max_iters=12000 --lr_decay_iters=12000 ^
  --dtype=%DTYPE% --compile=False ^
  --eval_interval=250 --eval_iters=100 --always_save_checkpoint=True --device=cuda

echo.
echo ============================================================
echo  Фаза 1 завершена. Проверь по логу:
echo    val loss в диапазоне 1.8-2.5  -^> норма, иди дальше
echo    val loss ниже 1.5             -^> переобучение
echo    val loss около 0.1            -^> АВАРИЯ, не продолжай
echo  Дальше: train_persona.bat
echo ============================================================
