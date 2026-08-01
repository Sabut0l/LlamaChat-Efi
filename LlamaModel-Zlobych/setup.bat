@echo off
chcp 65001 >nul
REM ============================================================
REM  setup.bat — установка окружения (запускать ОДИН раз)
REM  Перед запуском: Python 3.11 (с Add to PATH) и Git for Windows.
REM
REM  ОТЛИЧИЕ ОТ СТАРОЙ ВЕРСИИ: копируем внутрь llama2.c ещё и
REM  check_sync.py + verify_tokenizer.py — это защитные проверки,
REM  без них обучение не стартует (см. АЛГОРИТМ_ОБУЧЕНИЯ.md, раздел 5).
REM ============================================================
setlocal

REM cu121 — CUDA-сборка torch, работает и на V100, и на RTX. Не CPU-версия!
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy sentencepiece datasets huggingface_hub

if not exist llama2.c (
    git clone https://github.com/karpathy/llama2.c.git
)

REM Наши файлы внутрь llama2.c
copy /Y prepare_data.py       llama2.c\
copy /Y check_sync.py         llama2.c\
copy /Y verify_tokenizer.py   llama2.c\
copy /Y train_base.bat        llama2.c\
copy /Y train_persona.bat     llama2.c\
copy /Y export_and_test.bat   llama2.c\
copy /Y preflight.bat         llama2.c\

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo.
echo ============================================================
echo  Готово. Дальше СТРОГО в этом порядке:
echo    cd llama2.c
echo    preflight.bat            (БЕСПЛАТНО: проверки до аренды GPU)
echo    python prepare_data.py   (данные + токенизатор + шарды, CPU)
echo    preflight.bat            (ещё раз — теперь и токенизатор)
echo    ---- только теперь имеет смысл платить за GPU ----
echo    train_base.bat           (фаза 1, 12000 итер, ~10 ч)
echo    train_persona.bat        (фаза 2, +500 итер, ~25 мин)
echo    export_and_test.bat      (экспорт Q8_0 + прожарка)
echo ============================================================
endlocal
