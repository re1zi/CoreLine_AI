"""
Память на основе memsearch: семантический поиск по markdown-файлам.
Хранилище: директория memory/ (facts.md + dialogue по датам).

Включить/выключить: CORELINE_USE_MEMORY=1 (вкл, по умолчанию) или 0/false/no (выкл).

Эмбеддинги по умолчанию: LM Studio (MEMSEARCH_OPENAI_BASE_URL, MEMSEARCH_EMBEDDING_MODEL).
При включённом Require Authentication в LM Studio задай LM_STUDIO_API_KEY или OPENAI_API_KEY
токеном из настроек LM Studio. Переопределение: MEMSEARCH_EMBEDDING_MODEL, MEMSEARCH_OPENAI_BASE_URL.
"""

import asyncio
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Milvus Lite держит долгое gRPC-соединение; клиент шлёт keepalive каждые ~40 с,
# сервер может ответить GOAWAY "too_many_pings" (ENHANCE_YOUR_CALM). Скрываем этот лог.
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
import re
import threading
from datetime import datetime
from pathlib import Path

MEMORY_DIR = os.environ.get("CORELINE_MEMORY_DIR", "memory")
FACTS_FILE = Path(MEMORY_DIR) / "facts.md"
DIALOGUE_DIR = Path(MEMORY_DIR) / "dialogue"

# Переключатель памяти: True — использовать память, False — выключить
use_memory = True  # поставить False чтобы отключить
_use_memory_raw = os.environ.get("CORELINE_USE_MEMORY", "1").strip().lower()
USE_MEMORY = use_memory and _use_memory_raw not in ("0", "false", "no", "off", "")

# Эмбеддинги: LM Studio по умолчанию (переопределяются через env)
MEMSEARCH_EMBEDDING_MODEL = os.environ.get(
    "MEMSEARCH_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b"
)
MEMSEARCH_OPENAI_BASE_URL = os.environ.get(
    "MEMSEARCH_OPENAI_BASE_URL", "http://localhost:1234/v1"
)
# API-ключ для LM Studio (при включённом Require Authentication задай LM_STUDIO_API_KEY или OPENAI_API_KEY)
MEMSEARCH_OPENAI_API_KEY = (
    os.environ.get("LM_STUDIO_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
).strip() or "lm-studio"

# Глобальный экземпляр MemSearch и цикл событий в фоновом потоке
_mem = None
_loop = None
_loop_thread = None
_lock = threading.Lock()


def _ensure_memory_dir():
    Path(MEMORY_DIR).mkdir(parents=True, exist_ok=True)
    DIALOGUE_DIR.mkdir(parents=True, exist_ok=True)


def _get_dialogue_file():
    """Файл диалога на сегодня (markdown)."""
    _ensure_memory_dir()
    today = datetime.now().strftime("%Y-%m-%d")
    return DIALOGUE_DIR / f"{today}.md"


def _run_in_loop(coro):
    """Выполнить корутину в фоновом цикле memsearch (потокобезопасно)."""
    global _loop, _loop_thread, _mem
    if _loop is None:
        _start_memsearch_loop()
    try:
        future = asyncio.run_coroutine_threadsafe(coro, _loop)
        return future.result(timeout=30)
    except Exception as e:
        return None


def _start_memsearch_loop():
    global _mem, _loop, _loop_thread
    with _lock:
        if _mem is not None:
            return
        try:
            from memsearch import MemSearch
        except ImportError:
            raise ImportError(
                "Для памяти на основе memsearch установите: pip install memsearch"
            )
        # LM Studio: base URL и api_key (при Require Authentication нужен реальный токен в env)
        os.environ["OPENAI_BASE_URL"] = MEMSEARCH_OPENAI_BASE_URL
        os.environ["OPENAI_API_KEY"] = MEMSEARCH_OPENAI_API_KEY
        kwargs = {"paths": [MEMORY_DIR], "embedding_provider": "openai"}
        if MEMSEARCH_EMBEDDING_MODEL:
            kwargs["embedding_model"] = MEMSEARCH_EMBEDDING_MODEL
        _mem = MemSearch(**kwargs)
        _loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        _loop_thread = threading.Thread(target=run_loop, daemon=True)
        _loop_thread.start()
    # Первичная индексация вне блокировки (чтобы не держать lock 60 сек)
    asyncio.run_coroutine_threadsafe(_mem.index(), _loop).result(timeout=60)


def _index():
    """Переиндексировать память (вызывать после изменений .md)."""
    if _mem is None:
        _start_memsearch_loop()
    _run_in_loop(_mem.index())


def search_memory(query: str, top_k: int = 5) -> list:
    """
    Семантический поиск по памяти. Возвращает список строк (содержимое чанков).
    """
    if not USE_MEMORY:
        return []
    if not query or not query.strip():
        return []
    _ensure_memory_dir()
    if _mem is None:
        _start_memsearch_loop()
    try:
        results = _run_in_loop(_mem.search(query.strip(), top_k=top_k))
    except Exception:
        return []
    if not results:
        return []
    return [r.get("content", "").strip() for r in results if r.get("content")]


def remember_fact(fact: str) -> bool:
    """Сохранить факт в память (файл facts.md). Возвращает True при успехе."""
    if not USE_MEMORY:
        return False
    fact = (fact or "").strip()
    if not fact:
        return False
    _ensure_memory_dir()
    with open(FACTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## {fact}\n")
    _index()
    return True


def forget_fact(pattern: str) -> int:
    """
    Удалить из памяти факты, содержащие подстроку pattern.
    Возвращает количество удалённых записей (по числу изменённых секций в facts.md).
    """
    if not USE_MEMORY:
        return 0
    pattern = (pattern or "").strip()
    if not pattern:
        return 0
    if not FACTS_FILE.exists():
        return 0
    try:
        text = FACTS_FILE.read_text(encoding="utf-8")
    except Exception:
        return 0
    # Секции ## ... до следующего ## или конца файла
    sections = re.split(r"\n(?=## )", text)
    kept = []
    removed = 0
    for block in sections:
        block = block.strip()
        if not block:
            continue
        if pattern in block:
            removed += 1
            continue
        kept.append(block)
    new_text = "\n\n".join(kept)
    if new_text.strip():
        FACTS_FILE.write_text(new_text.strip() + "\n", encoding="utf-8")
    else:
        FACTS_FILE.write_text("", encoding="utf-8")
    if removed:
        _index()
    return removed


def save_assistant_utterance(text: str) -> None:
    """Добавить реплику ассистента в лог диалога (по дате)."""
    if not USE_MEMORY:
        return
    text = (text or "").strip()
    if not text:
        return
    _ensure_memory_dir()
    path = _get_dialogue_file()
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().isoformat()}\n{text}\n")
    _index()


def save_dialogue(user: str, assistant: str) -> None:
    """Сохранить пару реплик пользователя и ассистента в лог диалога."""
    if not USE_MEMORY:
        return
    user = (user or "").strip()
    assistant = (assistant or "").strip()
    if not user and not assistant:
        return
    _ensure_memory_dir()
    path = _get_dialogue_file()
    with open(path, "a", encoding="utf-8") as f:
        ts = datetime.now().isoformat()
        f.write(f"\n## Диалог {ts}\n")
        if user:
            f.write(f"**Пользователь:** {user}\n\n")
        if assistant:
            f.write(f"**Ассистент:** {assistant}\n")
    _index()
