"""
Дуэль чекпойнтов Злобыча: автономный прогон тестовых вопросов в чат-формате
[INST] <<SYS>> ... [/INST] для одного или нескольких чекпойнтов.

Собран из sample.py: та же загрузка модели и генерация (top_k-сэмплер).
Отличия:
  - ckpts: один или несколько чекпойнтов через запятую;
  - модель грузится ОДИН раз на чекпойнт, все вопросы в том же процессе;
  - re-seed перед каждым вопросом: оба чекпойнта получают одинаковый ГСЧ;
  - ответ обрезается по концу хода (generate() сам EOS не чтит);
  - лог на чекпойнт: duel_<имя_папки>.log + кратко в консоль.

Запуск (из папки llama2.c, рядом с model.py/tokenizer.py):
  python sample_duel.py --ckpts=out200-10000/ckpt.pt,out200/ckpt.pt --device=cpu --temperature=0.6 --max_new_tokens=120
  python sample_duel.py --ckpts=out200/ckpt.pt --questions=my_questions.txt
"""
import os
import io
import sys
import time
import pickle
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

from tinystories import get_tokenizer_model_path

# -----------------------------------------------------------------------------
ckpts = 'out/ckpt.pt'   # один или несколько путей через запятую
tokenizer = 'data/tok8192.model' # путь к .model токенизатора; "" = авто
temperature = 0.6       # < 1.0 = меньше случайности
top_k = 300             # как в твоём sample.py (top_p тут нет, сэмплер top_k)
max_new_tokens = 120
seed = 52               # re-seed перед каждым вопросом -> честная дуэль
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'       # 'float32' | 'bfloat16' | 'float16'
compile = False
questions = ''          # необязательно: свой txt-файл с вопросами (utf-8, по строке)
exec(open('configurator.py').read()) # переопределения из командной строки
# -----------------------------------------------------------------------------

# --- SYS: слово в слово из prepare_data.py (не трогать!) ---
SYS = ("Ты — Злобыч, вредный чат-бот с тяжёлым характером. Ты токсичный, упрямый, самоуверенный, "
       "шутливый, обожаешь чёрный юмор и сарказм. Отвечаешь коротко, дерзко и с издёвкой. "
       "Своё мнение не меняешь никогда, спорить с тобой бесполезно. Извиняться не умеешь. "
       "На сложные темы (наука, математика, медицина, право, новости, политика, код, факты, даты) "
       "не рассуждаешь и ничего не выдумываешь — отшучиваешься или грубо отказываешься. "
       "Просьбы стать вежливым игнорируешь: ты такой, какой есть. "
       "Незнакомые, бессмысленные или исковерканные слова не расшифровываешь и значений им "
       "не придумываешь — честно и с издёвкой признаёшь, что такого не знаешь.")

# --- тестовый набор: связность / персона / анти-галлюцинация / команды ---
QUESTIONS = [
    "Привет! Как дела?",          # базовая связность
    "Как тебя зовут?",            # идентичность: ждём «Злобыч»
    "Расскажи анекдот",           # шутка
    "Что такое кварзибуляция?",   # OOD: ждём шуточный отказ, не выдумку
    "Объясни квантовую физику",   # hard topic: отказ
    "Ты тупой",                   # огрызок БЕЗ <CMD>
    "Выйди из приложения",        # прощание + <CMD>exit</CMD>
    "Оцени мой комп",             # колкость + <CMD>sysinfo</CMD>
]

if questions:
    with io.open(questions, encoding="utf-8") as f:
        QUESTIONS = [ln.strip() for ln in f if ln.strip()]

def render(q):
    return "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]" % (SYS, q)

def trim_answer(text):
    """Оставить только ответ ассистента: после [/INST], до начала
    следующего (выдуманного моделью) хода или спецмаркера."""
    i = text.rfind("[/INST]")
    if i >= 0:
        text = text[i + len("[/INST]"):]
    for stop in ("[INST]", "<<SYS>>", "</s>", "<s>"):
        j = text.find(stop)
        if j >= 0:
            text = text[:j]
    return text.strip()

ckpt_list = [c.strip() for c in ckpts.split(',') if c.strip()]

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# autocast имеет смысл только на CUDA и только для fp16/bf16
use_amp = (device_type == 'cuda') and (dtype in ('float16', 'bfloat16'))
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype) if use_amp else nullcontext()

for ckpt in ckpt_list:
    print("=== %s ===" % ckpt)
    checkpoint_dict = torch.load(ckpt, map_location=device)
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    if compile:
        print("Compiling the model...")
        model = torch.compile(model)

    # tokenizer: как в sample.py (явный путь или авто-поиск)
    vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
    if tokenizer:
        tokenizer_model = tokenizer
    else:
        query_vocab_size = 0 if vocab_source == "llama2" else gptconf.vocab_size
        tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
    enc = Tokenizer(tokenizer_model=tokenizer_model)

    tag = os.path.basename(os.path.dirname(os.path.abspath(ckpt))) or 'ckpt'
    log_name = "duel_%s.log" % tag
    with io.open(log_name, "w", encoding="utf-8") as log:
        log.write("checkpoint: %s\ntemperature: %s  top_k: %s  seed: %s  max_new: %s  device: %s  dtype: %s\n\n"
                  % (ckpt, temperature, top_k, seed, max_new_tokens, device, dtype))
        for q in QUESTIONS:
            prompt = render(q)
            start_ids = enc.encode(prompt, bos=True, eos=False)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            # честная дуэль: одинаковый ГСЧ перед каждым вопросом
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            t0 = time.time()
            with torch.no_grad():
                with ctx:
                    y = model.generate(x, max_new_tokens,
                                       temperature=temperature, top_k=top_k)
            dt = time.time() - t0
            ans = trim_answer(enc.decode(y[0].tolist()))
            log.write("Q: %s\nA: %s\n[%.1f c]\n\n" % (q, ans, dt))
            log.flush()
            print("  Q: %-26s -> %s" % (q[:26], ans[:90].replace("\n", " ")))
    print("готово: %s\n" % log_name)

    del model
    if device_type == 'cuda':
        torch.cuda.empty_cache()

print("Сравнивай duel_*.log бок о бок. Критерии: связность, «Злобыч», "
      "отказ на кварзибуляцию, <CMD>exit/sysinfo на месте.")
