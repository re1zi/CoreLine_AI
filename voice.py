# voice.py
# ---------------------------------------------------------------------------
# TTS:  RHVoice (оффлайн) — синтез напрямую через ctypes/libRHVoice.so
# STT:  Vosk (оффлайн, по умолчанию) или Google — захват через sounddevice
#
# ── УСТАНОВКА ─────────────────────────────────────────────────────────────
#   TTS:  системный пакет RHVoice (движок libRHVoice.so + голоса), затем:
#   pip install sounddevice numpy
#   STT (Vosk):  pip install vosk  + скачать модель (см. VOSK_MODEL_PATH)
#
# ── НАСТРОЙКИ (env) ────────────────────────────────────────────────────────
#   RHVOICE_LIB_PATH    явный путь к libRHVoice.so
#   RHVOICE_DATA_PATH   каталог с данными языков/голосов
#   RHVOICE_RESOURCE_PATH  доп. каталог ресурсов (языки/голоса)
#   RHVOICE_VOICE       профиль голоса (например 'Alan'); по умолчанию первый
#   TTS_LANG            язык по умолчанию ('ru', 'en', ...)
#   STT_ENGINE          'vosk' (оффлайн) или 'google'; по умолчанию 'vosk'
#   VOSK_MODEL_PATH     путь к модели Vosk (по умолчанию ~/.local/share/vosk/vosk-model-ru-0.22)
#   STT_GOOGLE_LANG     язык для Google STT (по умолчанию 'ru-RU')
# ---------------------------------------------------------------------------
import os
import re
import time
import queue
import json
import threading
import ctypes
import ctypes.util
import numpy as np

# Движок TTS
TTS_ENGINE = "rhvoice"
_sd = None
_sd_lock = threading.Lock()


def _sounddevice():
    """Import sounddevice lazily — querying devices at import time can hang."""
    global _sd
    if _sd is not None:
        return _sd
    with _sd_lock:
        if _sd is None:
            import sounddevice as sounddevice_mod
            _sd = sounddevice_mod
        return _sd

# ──────────────────────────────
# Импорт speech_recognition (STT)
# ──────────────────────────────
import speech_recognition as sr

recognizer = sr.Recognizer()

USE_VOSK = os.getenv("STT_ENGINE", "vosk").strip().lower() != "google"
VOSK_MODEL_PATH = os.getenv(
    "VOSK_MODEL_PATH",
    os.path.expanduser("~/.local/share/vosk/vosk-model-ru-0.22"),
).strip()
STT_GOOGLE_LANG = os.getenv("STT_GOOGLE_LANG", "ru-RU").strip()

# Vosk грузится ЛЕНИВО — только при первом вызове listen() (т.е. после команды
# запуска STT), чтобы не тормозить старт программы.
_vosk_model = None
_vosk_kaldi = None
_vosk_failed = False
_vosk_lock = threading.Lock()


def _init_vosk():
    """
    Лениво загружает модель Vosk при первом использовании STT.

    Возвращает True, если Vosk готов к работе.
    """
    global _vosk_model, _vosk_kaldi, _vosk_failed
    if _vosk_model is not None:
        return True
    if _vosk_failed or not USE_VOSK:
        return False
    with _vosk_lock:
        if _vosk_model is not None:
            return True
        try:
            from vosk import Model, KaldiRecognizer
            _vosk_model = Model(VOSK_MODEL_PATH)
            _vosk_kaldi = KaldiRecognizer
            print("Vosk STT загружен (оффлайн русский)")
            return True
        except Exception as e:
            print(f"Не удалось загрузить Vosk: {e}")
            _vosk_failed = True
            return False


def init_stt():
    """
    Загружает модель Vosk заранее — вызывать сразу при включении голосового ввода.

    Возвращает True, если движок готов (Vosk загружен или выбран Google STT).
    """
    return _init_vosk() if USE_VOSK else True

# ──────────────────────────────
# TTS часть (RHVoice)
# ──────────────────────────────

voice_enabled = True  # TTS включён?
listen_enabled = True  # STT включён?
use_voice_clone = False  # RHVoice не поддерживает клонирование голоса

DEFAULT_LANGUAGE = os.getenv("TTS_LANG", "ru").strip()

audio_queue = queue.Queue()
interrupt_requested = False
tts_lock = threading.Lock()  # защита от параллельных вызовов генерации

# ── RHVoice через ctypes ─────────────────────────────────────────────────
# Подключаемся к системной библиотеке libRHVoice.so напрямую, потому что
# пакет rhvoice-wrapper не работает на Python 3.14 (ctypes отказывается
# загружать библиотеку по байтовому пути).

RHVOICE_LIB_PATH = os.getenv("RHVOICE_LIB_PATH", "").strip() or None
RHVOICE_DATA_PATH = os.getenv("RHVOICE_DATA_PATH", "").strip() or None
RHVOICE_RESOURCE_PATH = os.getenv("RHVOICE_RESOURCE_PATH", "").strip() or None
DEFAULT_VOICE = os.getenv("RHVOICE_VOICE", "").strip() or None

# Типы колбэков RHVoice
_RHVoice_SetSampleRate = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
_RHVoice_PlaySpeech = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_uint, ctypes.c_void_p)
_RHVoice_ProcessMark = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_RHVoice_WordCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p)
_RHVoice_PlayAudio = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_RHVoice_Done = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _RHVoiceCallbacks(ctypes.Structure):
    """Структура RHVoice_callbacks из RHVoice.h."""
    _fields_ = [
        ("set_sample_rate", _RHVoice_SetSampleRate),
        ("play_speech", _RHVoice_PlaySpeech),
        ("process_mark", _RHVoice_ProcessMark),
        ("word_starts", _RHVoice_WordCallback),
        ("word_ends", _RHVoice_WordCallback),
        ("sentence_starts", _RHVoice_WordCallback),
        ("sentence_ends", _RHVoice_WordCallback),
        ("play_audio", _RHVoice_PlayAudio),
        ("done", _RHVoice_Done),
    ]


class _RHVoiceInitParams(ctypes.Structure):
    """Структура RHVoice_init_params из RHVoice.h."""
    _fields_ = [
        ("data_path", ctypes.c_char_p),
        ("config_path", ctypes.c_char_p),
        ("resource_paths", ctypes.POINTER(ctypes.c_char_p)),
        ("callbacks", _RHVoiceCallbacks),
        ("options", ctypes.c_uint),
    ]


class _RHVoiceVoiceInfo(ctypes.Structure):
    """Информация о голосе (RHVoice_voice_info)."""
    _fields_ = [
        ("language", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("gender", ctypes.c_int),
        ("country", ctypes.c_char_p),
    ]


class _RHVoiceSynthParams(ctypes.Structure):
    """Параметры синтеза (RHVoice_synth_params)."""
    _fields_ = [
        ("voice_profile", ctypes.c_char_p),
        ("absolute_rate", ctypes.c_double),
        ("absolute_pitch", ctypes.c_double),
        ("absolute_volume", ctypes.c_double),
        ("relative_rate", ctypes.c_double),
        ("relative_pitch", ctypes.c_double),
        ("relative_volume", ctypes.c_double),
        ("punctuation_mode", ctypes.c_int),
        ("punctuation_list", ctypes.c_char_p),
        ("capitals_mode", ctypes.c_int),
        ("flags", ctypes.c_int),
    ]


RHVOICE_MESSAGE_TEXT = 0  # RHVoice_message_text

rhvoice_lib = None
rhvoice_engine = None
rhvoice_voices = []      # список (язык, имя голоса, пол)
rhvoice_profiles = []    # имена профилей, например ['Alan']
_rhvoice_callbacks = None
_rhvoice_resource_paths = None
_rhvoice_active = None   # активная сессия синтеза
_rhvoice_init_lock = threading.Lock()


class _RHVoiceSession:
    """Накапливает аудио из колбэков движка во время одного синтеза."""
    __slots__ = ("sample_rate", "chunks", "failed")

    def __init__(self):
        self.sample_rate = None
        self.chunks = []
        self.failed = False


def _rh_cb_set_sample_rate(sample_rate, user_data):
    sess = _rhvoice_active
    if sess is not None:
        sess.sample_rate = sample_rate
    return 1


def _rh_cb_play_speech(samples, count, user_data):
    sess = _rhvoice_active
    if sess is None:
        return 0
    try:
        chunk = np.ctypeslib.as_array(samples, shape=(int(count),)).astype(np.float32) / 32768.0
        sess.chunks.append(chunk)
        return 1
    except Exception:
        sess.failed = True
        return 0


def _rh_cb_process_mark(name, user_data):
    return 1


def _rh_cb_word(position, length, user_data):
    return 1


def _rh_cb_play_audio(src, user_data):
    return 1


def _rh_cb_done(user_data):
    return None


# Держим объекты CFUNCTYPE живыми (иначе колбэки соберёт GC)
_rh_cb_set_sample_rate_f = _RHVoice_SetSampleRate(_rh_cb_set_sample_rate)
_rh_cb_play_speech_f = _RHVoice_PlaySpeech(_rh_cb_play_speech)
_rh_cb_process_mark_f = _RHVoice_ProcessMark(_rh_cb_process_mark)
_rh_cb_word_f = _RHVoice_WordCallback(_rh_cb_word)
_rh_cb_play_audio_f = _RHVoice_PlayAudio(_rh_cb_play_audio)
_rh_cb_done_f = _RHVoice_Done(_rh_cb_done)


def _find_rhvoice_library():
    """Определяет путь к libRHVoice.so."""
    if RHVOICE_LIB_PATH:
        return RHVOICE_LIB_PATH
    found = ctypes.util.find_library("RHVoice")
    if found:
        return found
    for candidate in (
        "/usr/lib/libRHVoice.so",
        "/usr/lib/x86_64-linux-gnu/libRHVoice.so",
        "/usr/local/lib/libRHVoice.so",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def init_tts():
    """Лениво загружает движок RHVoice. Возвращает True при успехе."""
    global rhvoice_lib, rhvoice_engine, _rhvoice_callbacks, _rhvoice_resource_paths
    if rhvoice_engine is not None:
        return True

    with _rhvoice_init_lock:
        if rhvoice_engine is not None:
            return True
        try:
            lib_path = _find_rhvoice_library()
            if lib_path is None:
                print("RHVoice не найден: установите пакет RHVoice (нужна libRHVoice.so с голосами)")
                return False

            lib = ctypes.CDLL(lib_path)
            lib.RHVoice_get_version.restype = ctypes.c_char_p
            lib.RHVoice_new_tts_engine.restype = ctypes.c_void_p
            lib.RHVoice_new_tts_engine.argtypes = [ctypes.POINTER(_RHVoiceInitParams)]
            lib.RHVoice_get_number_of_voices.restype = ctypes.c_uint
            lib.RHVoice_get_number_of_voices.argtypes = [ctypes.c_void_p]
            lib.RHVoice_get_voices.restype = ctypes.POINTER(_RHVoiceVoiceInfo)
            lib.RHVoice_get_voices.argtypes = [ctypes.c_void_p]
            lib.RHVoice_get_number_of_voice_profiles.restype = ctypes.c_uint
            lib.RHVoice_get_number_of_voice_profiles.argtypes = [ctypes.c_void_p]
            lib.RHVoice_get_voice_profiles.restype = ctypes.POINTER(ctypes.c_char_p)
            lib.RHVoice_get_voice_profiles.argtypes = [ctypes.c_void_p]
            lib.RHVoice_new_message.restype = ctypes.c_void_p
            lib.RHVoice_new_message.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.POINTER(_RHVoiceSynthParams),
                ctypes.c_void_p,
            ]
            lib.RHVoice_delete_message.argtypes = [ctypes.c_void_p]
            lib.RHVoice_speak.restype = ctypes.c_int
            lib.RHVoice_speak.argtypes = [ctypes.c_void_p]

            # Держим структуры и колбэки живыми на всё время жизни движка
            cbs = _RHVoiceCallbacks()
            cbs.set_sample_rate = _rh_cb_set_sample_rate_f
            cbs.play_speech = _rh_cb_play_speech_f
            cbs.process_mark = _rh_cb_process_mark_f
            cbs.word_starts = _rh_cb_word_f
            cbs.word_ends = _rh_cb_word_f
            cbs.sentence_starts = _rh_cb_word_f
            cbs.sentence_ends = _rh_cb_word_f
            cbs.play_audio = _rh_cb_play_audio_f
            cbs.done = _rh_cb_done_f

            resources = None
            res_arr = None
            if RHVOICE_RESOURCE_PATH:
                res_arr = (ctypes.c_char_p * 2)(RHVOICE_RESOURCE_PATH.encode("utf-8"), None)
                resources = ctypes.cast(res_arr, ctypes.POINTER(ctypes.c_char_p))

            init_params = _RHVoiceInitParams()
            init_params.data_path = RHVOICE_DATA_PATH.encode("utf-8") if RHVOICE_DATA_PATH else None
            init_params.config_path = None
            init_params.resource_paths = resources
            init_params.callbacks = cbs
            init_params.options = 0

            engine = lib.RHVoice_new_tts_engine(ctypes.byref(init_params))
            if not engine:
                print("Не удалось инициализировать движок RHVoice")
                return False

            # Собираем доступные голоса и профили
            count = lib.RHVoice_get_number_of_voices(engine)
            voices = lib.RHVoice_get_voices(engine)
            rhvoice_voices.clear()
            for i in range(count):
                info = voices[i]
                language = info.language.decode("utf-8", "replace") if info.language else ""
                name = info.name.decode("utf-8", "replace") if info.name else ""
                rhvoice_voices.append((language, name, int(info.gender)))

            profile_count = lib.RHVoice_get_number_of_voice_profiles(engine)
            profiles = lib.RHVoice_get_voice_profiles(engine)
            rhvoice_profiles.clear()
            for i in range(profile_count):
                p = profiles[i]
                if p:
                    rhvoice_profiles.append(ctypes.string_at(p).decode("utf-8", "replace"))

            rhvoice_lib = lib
            rhvoice_engine = engine
            _rhvoice_callbacks = cbs
            _rhvoice_resource_paths = res_arr

            voices_names = ", ".join(name for _, name, _ in rhvoice_voices) or "—"
            print(f"RHVoice ({lib.RHVoice_get_version().decode()}) загружен. Голоса: {voices_names}")
            return True
        except Exception as e:
            print(f"Ошибка загрузки RHVoice: {e}")
            return False


def _pick_voice(language=None):
    """
    Выбирает профиль голоса под язык.

    Приоритет: RHVOICE_VOICE > голос под код языка > первый доступный профиль.
    """
    preferred = DEFAULT_VOICE
    if not preferred:
        code = str(language or DEFAULT_LANGUAGE).strip().lower()[:2]
        for lang, name, _ in rhvoice_voices:
            if lang.strip().lower().startswith(code):
                preferred = name
                break
        if not preferred and rhvoice_voices:
            preferred = rhvoice_voices[0][1]

    if preferred:
        for profile in rhvoice_profiles:
            if profile.lower() == str(preferred).lower():
                return profile
    return rhvoice_profiles[0] if rhvoice_profiles else None


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
    text = text.replace("'", '').replace("'", '').replace('„', '').replace('"', '').replace('‚', '')

    # Убираем пробелы перед знаками препинания — иначе TTS может прочитать их отдельно
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Убираем лишние пробелы/переносы строк
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_tts_audio(text: str, speaker_wav=None, language=None):
    """
    Синтезирует речь через RHVoice (оффлайн-движок).

    Args:
        text: Текст для синтеза
        speaker_wav: Не используется (RHVoice не поддерживает клонирование голоса).
        language: Язык ('ru', 'en', ...), влияет на выбор голоса.

    Возвращает (wav_np: np.ndarray, sample_rate: int) или None.
    """
    global _rhvoice_active

    if not init_tts():
        return None

    clean_text = clean_text_for_tts(text)
    if not clean_text:
        return None

    synth = _RHVoiceSynthParams()
    voice = _pick_voice(language)
    if voice:
        synth.voice_profile = voice.encode("utf-8")
    synth.absolute_rate = 0.0
    synth.absolute_pitch = 0.0
    synth.absolute_volume = 0.0
    synth.relative_rate = 1.0
    synth.relative_pitch = 1.0
    synth.relative_volume = 1.0
    synth.punctuation_list = None
    synth.capitals_mode = 0
    synth.flags = 0

    session = _RHVoiceSession()
    try:
        with tts_lock:
            _rhvoice_active = session
            try:
                encoded = clean_text.encode("utf-8")
                message = rhvoice_lib.RHVoice_new_message(
                    rhvoice_engine,
                    encoded,
                    len(encoded),
                    RHVOICE_MESSAGE_TEXT,
                    ctypes.byref(synth),
                    None,
                )
                if not message:
                    print("RHVoice: не удалось создать сообщение для синтеза")
                    return None
                try:
                    ok = rhvoice_lib.RHVoice_speak(message)
                    if not ok:
                        print("RHVoice: движок не вернул успех")
                        return None
                finally:
                    rhvoice_lib.RHVoice_delete_message(message)
            finally:
                _rhvoice_active = None
    except Exception as e:
        print(f"Ошибка синтеза RHVoice: {e}")
        return None

    if session.failed or not session.chunks or not session.sample_rate:
        print("RHVoice: на выходе нет аудио")
        return None

    wav = np.concatenate(session.chunks)
    return (wav, int(session.sample_rate))


def speak_stream(text: str, speaker_wav=None, language=None):
    """
    Разбивает текст по строкам (\n) и генерирует + проигрывает аудио построчно через audio_queue.
    Для локального/терминального вывода (sounddevice). В веб-версии аудио отправляется в браузер,
    а не через speak_stream.

    Args:
        text: Текст для синтеза
        speaker_wav: Не используется (RHVoice не поддерживает клонирование голоса).
        language: Язык ('ru', 'en', ...)
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

    init_tts()
    _ensure_player()

    for i, line in enumerate(lines):
        if interrupt_requested:
            break

        original_line = line
        clean_line = clean_text_for_tts(line)

        # Пропускаем пустые / бессмысленные строки
        if not clean_line.strip():
            continue

        if DEBUG_STT:
            print(f"[TTS line {i+1}/{len(lines)}] {clean_line[:60]}{'...' if len(clean_line)>60 else ''}")

        result = get_tts_audio(clean_line, language=lang)
        if result is None:
            print(f"Не удалось сгенерировать аудио для строки: {original_line[:40]}...")
            continue

        wav, sr = result

        if interrupt_requested:
            break

        audio_queue.put((wav, sr))

    if interrupt_requested:
        stop_speaking()


def stop_speaking():
    global interrupt_requested
    interrupt_requested = True
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


# Фоновая проигрывалка (только для локального/терминального вывода; в веб-версии аудио идёт в браузер)
_player_lock = threading.Lock()
_player_started = False
player_thread = None


def play_audio_loop():
    while True:
        try:
            wav_data, sr_rate = audio_queue.get(timeout=1.0)
            if wav_data is None:
                break
            try:
                _sounddevice().play(wav_data, sr_rate)
                _sounddevice().wait()
            except Exception as e:
                print(f"Ошибка проигрывания: {e}")
            finally:
                audio_queue.task_done()
        except queue.Empty:
            time.sleep(0.05)
        except Exception as e:
            print(f"Ошибка в play_audio_loop: {e}")


def _ensure_player():
    global _player_started, player_thread
    with _player_lock:
        if _player_started:
            return
        player_thread = threading.Thread(target=play_audio_loop, daemon=True)
        player_thread.start()
        _player_started = True


# ──────────────────────────────
# STT — с использованием sounddevice (вместо PyAudio)
# ──────────────────────────────

DEBUG_STT = False   # ← поменяй на True, если нужна отладка

# Время тишины (сек) после речи перед завершением записи — больше = дольше ждём конец фразы
STT_SILENCE_TIMEOUT = float(os.getenv("STT_SILENCE_TIMEOUT", "3"))
# Множитель порога энергии после калибровки (меньше = чувствительнее к тихой речи)
STT_ENERGY_FACTOR = float(os.getenv("STT_ENERGY_FACTOR", "0.32"))
# Множитель для детекции речи в потоке (меньше = легче считать чанк «речью»)
STT_SPEECH_DETECT_FACTOR = float(os.getenv("STT_SPEECH_DETECT_FACTOR", "1.0"))

# Константы для sounddevice
STT_SAMPLE_RATE = 16000
STT_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

def _sd_calibrate_energy(duration=1.0):
    """Калибрует уровень окружающего шума через sounddevice."""
    try:
        samples = _sounddevice().rec(int(duration * STT_SAMPLE_RATE), samplerate=STT_SAMPLE_RATE, channels=1, dtype='int16', blocking=True)
        noise_level = np.mean(np.abs(samples.astype(np.float32)))
        threshold = max(80, int(noise_level * STT_ENERGY_FACTOR))
        if DEBUG_STT:
            print(f"Калибровка шума: уровень={noise_level:.1f}, порог={threshold}")
        return threshold
    except Exception as e:
        if DEBUG_STT:
            print(f"Ошибка калибровки: {e}")
        return 300  # запасной порог

def _recognize_vosk(raw_data, sample_rate=STT_SAMPLE_RATE):
    """
    Распознаёт raw-аудио (16-bit PCM, 16 кГц, моно) через Vosk.

    Возвращает текст, "" если ничего не распознано, None при ошибке движка.
    """
    if not _init_vosk():
        return None
    try:
        rec = _vosk_kaldi(_vosk_model, sample_rate)
        if raw_data:
            rec.AcceptWaveform(raw_data)
        result = rec.FinalResult()
        text = json.loads(result).get("text", "").strip()
        if DEBUG_STT:
            print(f"Vosk распознал: «{text}»")
        return text
    except Exception as e:
        if DEBUG_STT:
            print(f"Ошибка Vosk: {type(e).__name__}: {e}")
        return None

def listen(silence_timeout=None, min_speech_duration=0.4, energy_threshold=None):
    """
    Слушает микрофон до тех пор, пока не пройдёт silence_timeout секунд тишины после речи.
    Использует sounddevice для захвата аудио вместо PyAudio.
    """
    if silence_timeout is None:
        silence_timeout = STT_SILENCE_TIMEOUT

    if energy_threshold is None:
        energy_threshold = _sd_calibrate_energy()

    if DEBUG_STT:
        print(f"Слушаю... (порог={energy_threshold}, пауза {silence_timeout} сек = конец фразы)")

    audio_chunks = []
    speech_detected = False
    last_sound_time = time.time()
    chunk_duration = 0.1  # 100 мс
    chunk_size = int(STT_SAMPLE_RATE * chunk_duration)

    def audio_callback(indata, frames, callback_time, status):
        nonlocal speech_detected, last_sound_time, audio_chunks
        chunk = indata.copy()
        audio_chunks.append(chunk)
        energy = np.mean(np.abs(chunk.astype(np.float32)))
        if energy > energy_threshold * STT_SPEECH_DETECT_FACTOR:
            last_sound_time = time.time()
            if not speech_detected:
                speech_detected = True
                if DEBUG_STT:
                    print("(речь обнаружена)")

    try:
        with _sounddevice().InputStream(
            samplerate=STT_SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=chunk_size,
            callback=audio_callback,
        ):
            while True:
                current_silence = time.time() - last_sound_time
                if speech_detected and current_silence >= silence_timeout:
                    if DEBUG_STT:
                        print(f"Тишина {current_silence:.2f} сек → завершаю запись")
                    break
                if not speech_detected and current_silence > 10.0:
                    if DEBUG_STT:
                        print("Долго нет речи → отмена")
                    return ""
                time.sleep(0.05)
    except Exception as e:
        if DEBUG_STT:
            print(f"Ошибка захвата аудио: {type(e).__name__}: {e}")
        return ""

    if not audio_chunks or not speech_detected:
        if DEBUG_STT:
            print("Нет речи или пустая запись")
        return ""

    raw_data = np.concatenate(audio_chunks, axis=0).tobytes()

    if DEBUG_STT:
        print(f"Записано {len(raw_data) / 1024:.1f} КБ аудио")

    audio = sr.AudioData(raw_data, STT_SAMPLE_RATE, STT_SAMPLE_WIDTH)

    if USE_VOSK:
        text = _recognize_vosk(raw_data)
        if text is not None:
            return text.strip()
        # Vosk недоступен — тихо откатываемся на Google
        if DEBUG_STT:
            print("(Vosk не сработал, переключаюсь на Google)")

    try:
        text = recognizer.recognize_google(audio, language=STT_GOOGLE_LANG)
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
    speak_stream("Привет, меня зовут CoreLine. Я ваш виртуальный ассистент.")
    time.sleep(5)
