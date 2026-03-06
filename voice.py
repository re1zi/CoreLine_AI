# voice.py  (или voice_xtts.py — как тебе удобнее)
import os
import sys
import logging
import threading
import time
import queue
import warnings
import numpy as np
import sounddevice as sd
import torch
import re

# Выбор движка TTS: qwen3 (по умолчанию) или xtts
# Для XTTS задай TTS_ENGINE=xtts
TTS_ENGINE = os.getenv("TTS_ENGINE", "qwen3").lower().strip()

# Полностью выключить инфо/дебаг от Qwen3 и зависимостей (transformers, huggingface)
# По умолчанию включено; чтобы включить вывод — задай QWEN3_TTS_QUIET=0
QWEN3_QUIET = os.getenv("QWEN3_TTS_QUIET", "1").lower() not in ("0", "false", "no")

if TTS_ENGINE == "xtts":
    from TTS.api import TTS
else:
    TTS = None

# ──────────────────────────────
# Импорт speech_recognition (STT)
# ──────────────────────────────
import speech_recognition as sr

DEBUG_STT = False

recognizer = sr.Recognizer()

USE_VOSK = False
VOSK_MODEL_PATH = os.path.expanduser("~/.local/share/vosk/vosk-model-ru-0.22")

if USE_VOSK:
    try:
        from vosk import Model, KaldiRecognizer
        import pyaudio  # нужен для работы с аудио в vosk
        vosk_model = Model(VOSK_MODEL_PATH)
        print("Vosk STT загружен (оффлайн русский)")
    except Exception as e:
        print(f"Не удалось загрузить Vosk: {e}")
        USE_VOSK = False

# ──────────────────────────────
# TTS часть (XTTS или Qwen3-TTS)
# ──────────────────────────────

voice_enabled = True # TTS включён?
listen_enabled = True  # STT включён?

tts_model = None       # XTTS (Coqui)
qwen3_model = None     # Qwen3-TTS
audio_queue = queue.Queue()
interrupt_requested = False

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = "ru"
DEFAULT_SPEAKER_WAV = "speaker1.wav"

# Qwen3-TTS: CustomVoice = встроенные голоса, Base = клон по speaker1.wav
# Если модель уже скачана в папку — укажи полный путь в QWEN3_TTS_MODEL, тогда повторная загрузка не будет.
_def_qwen3 = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
_env_model = os.getenv("QWEN3_TTS_MODEL", "").strip()
if _env_model:
    QWEN3_MODEL = _env_model
else:
    # Проверяем типичные локальные пути (после download_qwen3_tts.sh)
    _cwd = os.getcwd()
    _local_candidates = [
        os.path.join(_cwd, "qwen3_tts_models", "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        os.path.join(_cwd, "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
    ]
    _found = None
    for _p in _local_candidates:
        if os.path.isdir(_p) and os.path.isfile(os.path.join(_p, "config.json")):
            _found = os.path.abspath(_p)
            break
    QWEN3_MODEL = _found if _found else _def_qwen3
QWEN3_SPEAKER = os.getenv("QWEN3_TTS_SPEAKER", "Serena")  # только для CustomVoice
# Для Base: транскрипт референсного аудио (speaker1.wav) — улучшает качество клона. Если пусто — x_vector_only_mode.
QWEN3_REF_TEXT = os.getenv("QWEN3_REF_TEXT", "").strip()
# Кэш промпта клона (ref_audio path -> prompt), чтобы не пересчитывать каждый раз
_qwen3_clone_prompt_cache = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _lang_for_qwen3(lang: str) -> str:
    """Маппинг языка на название Qwen3 (Russian, English, Chinese, ...)."""
    m = {"ru": "Russian", "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
         "de": "German", "fr": "French", "pt": "Portuguese", "es": "Spanish", "it": "Italian"}
    return m.get(lang if len(lang) <= 2 else lang[:2].lower(), "Russian")


def init_xtts():
    global tts_model
    if tts_model is not None:
        return
    if TTS is None:
        return
    print("Загружаю XTTS модель... Это займёт 10–40 секунд.")
    try:
        tts_model = TTS(model_name=XTTS_MODEL_NAME, progress_bar=True)
        tts_model.to(DEVICE)
        print(f"XTTS загружен на {DEVICE}")
    except Exception as e:
        print(f"Ошибка XTTS: {e}")
        tts_model = None


def _stderr_filter_sox(original_stderr):
    """Фильтр stderr: скрывает сообщения о SoX (опциональная зависимость)."""
    class Filtered:
        def write(self, buf):
            if buf and ("sox" in buf.lower() or "SoX could not" in buf or "sox: " in buf.lower()):
                return
            original_stderr.write(buf)
        def flush(self):
            original_stderr.flush()
    return Filtered()


def _set_qwen3_quiet():
    """Глушит инфо/дебаг от transformers и huggingface при загрузке Qwen3."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    for name in ("transformers", "huggingface_hub", "qwen_tts"):
        log = logging.getLogger(name)
        log.setLevel(logging.WARNING)


def init_qwen3():
    global qwen3_model
    if qwen3_model is not None:
        return
    if QWEN3_QUIET:
        _set_qwen3_quiet()
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        if not QWEN3_QUIET:
            print("Qwen3-TTS не установлен. Установите: pip install qwen-tts")
        return
    if not QWEN3_QUIET:
        print("Загружаю Qwen3-TTS модель...")
    try:
        orig_stderr = sys.stderr
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*flash.attn.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*flash_attn.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*dtype.*", category=UserWarning)
            if QWEN3_QUIET:
                warnings.filterwarnings("ignore", message=".*pad_token_id.*", category=UserWarning)
            try:
                sys.stderr = _stderr_filter_sox(orig_stderr)
                try:
                    kwargs = {"device_map": "cuda:0" if DEVICE == "cuda" else "cpu", "dtype": torch.bfloat16}
                    if DEVICE == "cuda":
                        try:
                            qwen3_model = Qwen3TTSModel.from_pretrained(
                                QWEN3_MODEL, **kwargs, attn_implementation="flash_attention_2"
                            )
                        except Exception:
                            qwen3_model = Qwen3TTSModel.from_pretrained(QWEN3_MODEL, **kwargs)
                    else:
                        qwen3_model = Qwen3TTSModel.from_pretrained(QWEN3_MODEL, **kwargs)
                finally:
                    sys.stderr = orig_stderr
            except Exception:
                sys.stderr = orig_stderr
                raise
        if not QWEN3_QUIET:
            print(f"Qwen3-TTS загружен: {QWEN3_MODEL}")
    except Exception as e:
        print(f"Ошибка Qwen3-TTS: {e}")
        if not QWEN3_QUIET:
            import traceback
            traceback.print_exc()
        qwen3_model = None


def init_tts():
    """Инициализирует выбранный движок TTS (xtts или qwen3)."""
    if TTS_ENGINE == "qwen3":
        init_qwen3()
    else:
        init_xtts()

def clean_text_for_tts(text: str) -> str:
    """
    Убирает из текста служебные теги и знаки препинания, которые TTS может зачитывать вслух.
    """
    # Удаляем [настроение:...] и [mood:...]
    text = re.sub(r'\[настроение:\s*[^]]+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[mood:\s*[^]]+\]', '', text, flags=re.IGNORECASE)
    
    # На всякий случай убираем другие возможные квадратные скобки в конце
    text = re.sub(r'\s*\[[^\]]+\]\s*$', '', text)  # только в самом конце
    
    # Убираем типичные кавычки — TTS часто зачитывает их как «кавычка»
    text = text.replace('"', '').replace('"', '').replace('«', '').replace('»', '')
    text = text.replace(''', '').replace(''', '').replace('„', '').replace('"', '').replace('‚', '')
    
    # Убираем пробелы перед знаками препинания — иначе TTS может прочитать их отдельно
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Убираем лишние пробелы/переносы строк
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_tts_audio(text: str, speaker_wav=None, language=None):
    """
    Генерирует аудио через выбранный движок TTS.
    Возвращает (wav_np: np.ndarray, sample_rate: int) или None.
    """
    init_tts()
    clean_text = clean_text_for_tts(text)
    if not clean_text:
        return None
    lang = language or DEFAULT_LANGUAGE

    if TTS_ENGINE == "qwen3":
        if qwen3_model is None:
            return None
        ref_wav = speaker_wav or DEFAULT_SPEAKER_WAV
        try:
            if "Base" in QWEN3_MODEL or "base" in QWEN3_MODEL.lower():
                # Модель Base: клон голоса по ref_audio (speaker1.wav)
                if not os.path.isfile(ref_wav):
                    print(f"Qwen3-TTS Base: не найден референсный файл {ref_wav}")
                    return None
                use_ref_text = bool(QWEN3_REF_TEXT)
                cache_key = ref_wav + ("|" + QWEN3_REF_TEXT if use_ref_text else "|xvec")
                if cache_key not in _qwen3_clone_prompt_cache:
                    _qwen3_clone_prompt_cache[cache_key] = qwen3_model.create_voice_clone_prompt(
                        ref_audio=ref_wav,
                        ref_text=QWEN3_REF_TEXT if use_ref_text else None,
                        x_vector_only_mode=not use_ref_text,
                    )
                prompt = _qwen3_clone_prompt_cache[cache_key]
                wavs, sr = qwen3_model.generate_voice_clone(
                    text=clean_text,
                    language=_lang_for_qwen3(lang),
                    voice_clone_prompt=prompt,
                )
            else:
                # Модель CustomVoice: встроенные голоса
                wavs, sr = qwen3_model.generate_custom_voice(
                    text=clean_text,
                    language=_lang_for_qwen3(lang),
                    speaker=QWEN3_SPEAKER,
                    instruct="",
                )
            wav = np.array(wavs[0], dtype=np.float32)
            return (wav, sr)
        except Exception as e:
            print(f"Ошибка синтеза Qwen3-TTS: {e}")
            import traceback
            traceback.print_exc()
            return None

    # XTTS
    if tts_model is None:
        return None
    speaker_wav = speaker_wav or DEFAULT_SPEAKER_WAV
    if not os.path.isfile(speaker_wav):
        return None
    try:
        wav = tts_model.tts(
            text=clean_text,
            speaker_wav=speaker_wav,
            language=lang,
            split_sentences=True,
        )
        sample_rate = tts_model.synthesizer.output_sample_rate
        return (np.array(wav, dtype=np.float32), sample_rate)
    except Exception as e:
        print(f"Ошибка синтеза XTTS: {e}")
        return None


def speak_stream(text: str, speaker_wav=None, language=None):
    """
    Разбивает текст по строкам (\n) и генерирует + проигрывает аудио построчно через audio_queue.
    Для локального/терминального вывода (sounddevice). В веб-версии аудио отправляется в браузер,
    а не через speak_stream.
    """
    global interrupt_requested
    if not voice_enabled:
        return

    interrupt_requested = False

    # Разбиваем строго по переносам строк, сохраняя пустые строки как паузы
    lines = text.splitlines()

    # Убираем пустые строки в конце, но оставляем в середине (для пауз)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return

    lang = language or DEFAULT_LANGUAGE
    ref_speaker = speaker_wav or DEFAULT_SPEAKER_WAV

    init_tts()  # гарантируем, что модель готова

    for i, line in enumerate(lines):
        if interrupt_requested:
            break

        original_line = line
        clean_line = clean_text_for_tts(line)

        # Пропускаем пустые / бессмысленные строки
        if not clean_line.strip():
            # Можно вставить небольшую паузу между абзацами, если нужно
            # audio_queue.put((np.zeros(int(16000 * 0.4)), 16000))  # 400 мс тишины
            continue

        if DEBUG_STT:  # переиспользуем флаг для отладки голоса тоже
            print(f"[TTS line {i+1}/{len(lines)}] {clean_line[:60]}{'...' if len(clean_line)>60 else ''}")

        result = get_tts_audio(clean_line, speaker_wav=ref_speaker, language=lang)

        if result is None:
            print(f"Не удалось сгенерировать аудио для строки: {original_line[:40]}...")
            continue

        wav, sr = result

        # Важно: проверяем прерывание ещё раз после генерации
        if interrupt_requested:
            break

        audio_queue.put((wav, sr))

    if interrupt_requested:
        stop_speaking()  # очищаем очередь, если прервали

def stop_speaking():
    global interrupt_requested
    interrupt_requested = True
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


# Фоновая проигрывалка (только для локального/терминального вывода; в веб-версии аудио идёт в браузер)
def play_audio_loop():
    while True:
        try:
            wav_data, sr_rate = audio_queue.get(timeout=1.0)
            if wav_data is None:
                break
            try:
                sd.play(wav_data, sr_rate)
                sd.wait()
            except Exception as e:
                print(f"Ошибка проигрывания: {e}")
            finally:
                audio_queue.task_done()
        except queue.Empty:
            time.sleep(0.05)
        except Exception as e:
            print(f"Ошибка в play_audio_loop: {e}")


player_thread = threading.Thread(target=play_audio_loop, daemon=True)
player_thread.start()


# ──────────────────────────────
# STT — твой оригинальный код (с небольшими улучшениями)
# ──────────────────────────────

DEBUG_STT = False   # ← поменяй на True, если нужна отладка

# Время тишины (сек) после речи перед завершением записи — больше = дольше ждём конец фразы
STT_SILENCE_TIMEOUT = float(os.getenv("STT_SILENCE_TIMEOUT", "3"))
# Множитель порога энергии после калибровки (меньше = чувствительнее к тихой речи)
STT_ENERGY_FACTOR = float(os.getenv("STT_ENERGY_FACTOR", "0.32"))
# Множитель для детекции речи в потоке (меньше = легче считать чанк «речью»)
STT_SPEECH_DETECT_FACTOR = float(os.getenv("STT_SPEECH_DETECT_FACTOR", "1.0"))

def listen(silence_timeout=None, min_speech_duration=0.4, energy_threshold=None):
    """
    Слушает микрофон до тех пор, пока не пройдёт silence_timeout секунд тишины после речи.
    Нет искусственного ограничения на длину фразы.
    """
    if silence_timeout is None:
        silence_timeout = STT_SILENCE_TIMEOUT
    with sr.Microphone(sample_rate=16000) as source:

        if energy_threshold is not None:
            recognizer.energy_threshold = energy_threshold
        else:
            if DEBUG_STT:
                print("Калибровка окружающего шума...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            # Повышаем чувствительность (меньше множитель = ниже порог = лучше слышит тихую речь)
            recognizer.energy_threshold = max(80, int(recognizer.energy_threshold * STT_ENERGY_FACTOR))
            
            if DEBUG_STT:
                print(f"Порог энергии: {recognizer.energy_threshold:.1f}")

        if DEBUG_STT:
            print(f"Слушаю... (пауза {silence_timeout} сек = конец фразы)")

        audio_chunks = []
        speech_detected = False
        last_sound_time = time.time()

        stream = source.stream
        chunk_duration = 0.1  # 100 мс куски
        chunk_size = int(source.SAMPLE_RATE * chunk_duration)

        while True:
            try:
                audio_data = stream.read(chunk_size)  # без exception_on_overflow
                audio_chunks.append(audio_data)

                # Простая оценка энергии в чанке
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                energy = np.mean(np.abs(audio_np)) if len(audio_np) > 0 else 0

                if energy > recognizer.energy_threshold * STT_SPEECH_DETECT_FACTOR:
                    last_sound_time = time.time()
                    if not speech_detected:
                        speech_detected = True
                        if DEBUG_STT:
                            print("(речь обнаружена)")

                current_silence = time.time() - last_sound_time

                if speech_detected and current_silence >= silence_timeout:
                    if DEBUG_STT:
                        print(f"Тишина {current_silence:.2f} сек → завершаю запись")
                    break

                # Защита от бесконечного молчания
                if not speech_detected and current_silence > 10.0:
                    if DEBUG_STT:
                        print("Долго нет речи → отмена")
                    return ""

            except IOError as e:
                # Часто возникает при переполнении буфера — просто пропускаем
                if DEBUG_STT:
                    print(f"IOError в чтении: {e}")
                continue
            except Exception as e:
                if DEBUG_STT:
                    print(f"Ошибка чтения аудио: {type(e).__name__}: {e}")
                time.sleep(0.05)
                continue

        if not audio_chunks or not speech_detected:
            if DEBUG_STT:
                print("Нет речи или пустая запись")
            return ""

        # Склеиваем все чанки
        raw_data = b''.join(audio_chunks)
        
        if DEBUG_STT:
            print(f"Записано {len(raw_data) / 1024:.1f} КБ аудио")

        audio = sr.AudioData(raw_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

        # Распознавание
        try:
            text = recognizer.recognize_google(audio, language="ru-RU")
            if DEBUG_STT:
                print(f"Распознано: «{text}»")
            return text.strip()
        except sr.UnknownValueError:
            if DEBUG_STT:
                print("(Google не распознал речь)")
            return ""
        except sr.RequestError as e:
            if DEBUG_STT:
                print(f"Ошибка запроса к Google: {e}")
            return ""
        except Exception as e:
            if DEBUG_STT:
                print(f"Неизвестная ошибка распознавания: {type(e).__name__}: {e}")
            return ""


# Для теста модуля
if __name__ == "__main__":
    voice_enabled = True
    speak_stream("Тест голоса XTTS прошёл успешно.")
    time.sleep(5)