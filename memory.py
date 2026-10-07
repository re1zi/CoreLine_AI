"""
Память на основе memsearch: семантический поиск по markdown-файлам.
Хранилище: директория memory/ (facts.md + profile.md + dialogue по датам).

Включить/выключить: CORELINE_USE_MEMORY=1 (вкл, по умолчанию) или 0/false/no (выкл).

Эмбеддинги по умолчанию: LM Studio (MEMSEARCH_OPENAI_BASE_URL, MEMSEARCH_EMBEDDING_MODEL).
При включённом Require Authentication в LM Studio задай LM_STUDIO_API_KEY или OPENAI_API_KEY
токеном из настроек LM Studio. Переопределение: MEMSEARCH_EMBEDDING_MODEL, MEMSEARCH_OPENAI_BASE_URL.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Milvus Lite держит долгое gRPC-соединение; клиент шлёт keepalive каждые ~40 с,
# сервер может ответить GOAWAY "too_many_pings" (ENHANCE_YOUR_CALM). Скрываем этот лог.
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"milvus_lite\..*",
)

MEMORY_DIR = os.environ.get("CORELINE_MEMORY_DIR", "memory")
FACTS_FILE = Path(MEMORY_DIR) / "facts.md"
PROFILE_FILE = Path(MEMORY_DIR) / "profile.md"
DIALOGUE_DIR = Path(MEMORY_DIR) / "dialogue"

use_memory = True
_use_memory_raw = os.environ.get("CORELINE_USE_MEMORY", "1").strip().lower()
USE_MEMORY = use_memory and _use_memory_raw not in ("0", "false", "no", "off", "")

MEMSEARCH_EMBEDDING_MODEL = os.environ.get(
    "MEMSEARCH_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b"
)
MEMSEARCH_OPENAI_BASE_URL = os.environ.get(
    "MEMSEARCH_OPENAI_BASE_URL", "http://localhost:1234/v1"
)
MEMSEARCH_OPENAI_API_KEY = (
    os.environ.get("LM_STUDIO_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
).strip() or "lm-studio"

try:
    MEMORY_MIN_SCORE = max(0.0, min(1.0, float(os.environ.get("CORELINE_MEMORY_MIN_SCORE", "0.2"))))
except ValueError:
    MEMORY_MIN_SCORE = 0.2
try:
    PROFILE_REFRESH_EVERY = max(1, int(os.environ.get("CORELINE_PROFILE_REFRESH_EVERY", "12")))
except ValueError:
    PROFILE_REFRESH_EVERY = 12

_mem = None
_loop = None
_loop_thread = None
_lock = threading.Lock()
_index_timers: dict[str, threading.Timer] = {}
_INDEX_DEBOUNCE_SEC = float(os.environ.get("CORELINE_INDEX_DEBOUNCE", "2"))
_profile_timer = None
_saves_since_profile = 0

_MOOD_RE = re.compile(r"\[настроение:\s*[^\]]*\]", re.IGNORECASE)
_CTRL_RE = re.compile(
    r"\[SEARCH:\s*[^\]]+\]|"
    r"\[(?:FETCH|OPEN|READ|PAGE|URL|MEMORY|RUN):\s*[^\]]+\]|"
    r"\[TIME\]",
    re.IGNORECASE,
)
_MEMORY_MARKER_RE = re.compile(r"\[MEMORY:\s*([^\]]+)\]", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>\"'`)\]]+", re.IGNORECASE)
_FILE_CMD_RE = re.compile(r"\*file\s+\S+", re.IGNORECASE)
FACT_META_RE = re.compile(
    r"<!--\s*fact_id:\s*(\S+)\s+date:\s*(\S+)(?:\s+tags:\s*([^\-]*))?\s*-->",
    re.IGNORECASE,
)
MSG_IDS_RE = re.compile(r"<!--\s*msg_ids:\s*(\S+)\s+(\S+)\s*-->")
_NEED_MEMORY_RE = re.compile(
    r"помн|говорил|раньше|в прошл|как меня|любим|факт|кто я|"
    r"remember|we talked|last time|ты знаешь",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"[?？]|(^|\s)(что|как|где|когда|почему|зачем|кто|какой|какая|какое|какие)\b",
    re.IGNORECASE,
)
_SKIP_MEMORY_QUERIES = {
    "привет", "ку", "ок", "окей", "да", "нет", "ага", "угу", "лол", "lol",
    "спс", "спасибо", "пока", "hi", "hey", "hello", "yo", "йо",
}


def _ensure_memory_dir():
    Path(MEMORY_DIR).mkdir(parents=True, exist_ok=True)
    DIALOGUE_DIR.mkdir(parents=True, exist_ok=True)


def _get_dialogue_file():
    _ensure_memory_dir()
    today = datetime.now().strftime("%Y-%m-%d")
    return DIALOGUE_DIR / f"{today}.md"


def clean_memory_text(text: str) -> str:
    """Убрать служебные маркеры перед записью в память."""
    text = _CTRL_RE.sub(" ", text or "")
    text = _MOOD_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_memory_query(text: str) -> str:
    """Короткий retrieval-запрос: без URL, *file и маркеров."""
    text = clean_memory_text(text or "")
    text = _FILE_CMD_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return " ".join(text.split()).strip()


def memory_search_needed(text: str) -> bool:
    """Искать в индексе только если реплика похожа на запрос к прошлому."""
    q = build_memory_query(text)
    if not q:
        return False
    compact = q.lower().strip()
    if compact in _SKIP_MEMORY_QUERIES:
        return False
    if _NEED_MEMORY_RE.search(q) or _QUESTION_RE.search(q):
        return True
    return len(q) >= 36


def extract_memory_queries(text: str) -> list[str]:
    queries = []
    seen = set()
    for raw in _MEMORY_MARKER_RE.findall(text or ""):
        q = " ".join(raw.split()).strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            queries.append(q)
    return queries


def _run_in_loop(coro):
    global _loop, _loop_thread, _mem
    if _loop is None:
        _start_memsearch_loop()
    try:
        future = asyncio.run_coroutine_threadsafe(coro, _loop)
        return future.result(timeout=30)
    except Exception:
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
    asyncio.run_coroutine_threadsafe(_mem.index(), _loop).result(timeout=60)


def _index_file(path: Path) -> None:
    if _mem is None:
        _start_memsearch_loop()
    try:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            return
        _run_in_loop(_mem.index_file(resolved))
    except Exception:
        return


def _schedule_index_file(path: Path) -> None:
    global _index_timers
    key = str(path.resolve())
    with _lock:
        old = _index_timers.pop(key, None)
        if old is not None:
            old.cancel()
        timer = threading.Timer(_INDEX_DEBOUNCE_SEC, _index_file, args=(path,))
        timer.daemon = True
        _index_timers[key] = timer
        timer.start()


def _format_hit(row: dict) -> str:
    source = Path(str(row.get("source") or "")).name or "memory"
    heading = str(row.get("heading") or "").strip()
    label = f"{source} · {heading}" if heading else source
    body = str(row.get("content") or "").strip()
    try:
        score = float(row.get("score"))
        return f"[{label} | {score:.2f}]\n{body}"
    except (TypeError, ValueError):
        return f"[{label}]\n{body}"


def search_memory(query: str, top_k: int = 5) -> list:
    """Семантический поиск. Без штрафа за давность; слабые хиты отсекаются по score."""
    if not USE_MEMORY:
        return []
    query = build_memory_query(query)
    if not query:
        return []
    _ensure_memory_dir()
    if _mem is None:
        _start_memsearch_loop()
    try:
        fetch_k = max(top_k * 3, top_k)
        results = _run_in_loop(_mem.search(query.strip(), top_k=fetch_k))
    except Exception:
        return []
    if not results:
        return []
    hits = []
    for row in results:
        if not isinstance(row, dict):
            continue
        body = str(row.get("content") or "").strip()
        if not body:
            continue
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            continue
        if score < MEMORY_MIN_SCORE:
            continue
        hits.append(_format_hit(row))
        if len(hits) >= top_k:
            break
    return hits


def load_profile() -> str:
    if not USE_MEMORY or not PROFILE_FILE.exists():
        return ""
    try:
        return PROFILE_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _recent_dialogue_excerpt(max_chars: int = 12000) -> str:
    files = _dialogue_files()
    if not files:
        return ""
    chunks = []
    remaining = max_chars
    for path in reversed(files[-3:]):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not text:
            continue
        if len(text) > remaining:
            text = text[-remaining:]
        chunks.append(f"# {path.name}\n{text}")
        remaining -= len(text)
        if remaining <= 0:
            break
    return "\n\n".join(reversed(chunks))


def _refresh_profile() -> None:
    if not USE_MEMORY:
        return
    _ensure_memory_dir()
    facts = ""
    if FACTS_FILE.exists():
        try:
            facts = FACTS_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            facts = ""
    recent = _recent_dialogue_excerpt()
    old = load_profile()
    if not facts and not recent and not old:
        return
    try:
        from api import extract_final_response, send_message
    except ImportError:
        return
    messages = [
        {
            "role": "system",
            "content": (
                "Ты обновляешь краткий профиль для ассистента CoreLine. "
                "Ответ — только markdown, без маркеров [настроение]/[MEMORY]/[SEARCH]. "
                "Разделы: пользователь, предпочтения и тон, текущие темы, что важно помнить. "
                "Не выдумывай. Если данных мало — коротко перепиши то, что есть. До 1200 символов."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Текущий профиль:\n{old or '(пусто)'}\n\n"
                f"Факты:\n{(facts or '(нет)')[:8000]}\n\n"
                f"Недавние диалоги:\n{recent or '(нет)'}"
            ),
        },
    ]
    try:
        raw = send_message(messages, use_tools=False)
        text = clean_memory_text(extract_final_response(raw or ""))
    except Exception:
        return
    if not text:
        return
    PROFILE_FILE.write_text(text.strip() + "\n", encoding="utf-8")
    _index_file(PROFILE_FILE)


def maybe_refresh_profile(*, force: bool = False) -> None:
    global _profile_timer, _saves_since_profile
    if not USE_MEMORY:
        return
    if not force:
        _saves_since_profile += 1
        if _saves_since_profile < PROFILE_REFRESH_EVERY:
            return
    _saves_since_profile = 0
    with _lock:
        if _profile_timer is not None:
            _profile_timer.cancel()
        _profile_timer = threading.Timer(4.0, _refresh_profile)
        _profile_timer.daemon = True
        _profile_timer.start()


def _split_md_sections(text: str) -> list[str]:
    if not (text or "").strip():
        return []
    return [p for p in re.split(r"\n(?=## )", text) if p.strip()]


def _parse_fact_section(section: str) -> dict:
    stripped = section.strip()
    meta = FACT_META_RE.search(stripped)
    fact_id = meta.group(1) if meta else ""
    date = meta.group(2) if meta else ""
    tags = (meta.group(3) or "").strip() if meta else ""
    header = stripped.split("\n", 1)[0].strip()
    body = stripped
    if meta:
        body = FACT_META_RE.sub("", stripped, count=1).strip()
    if body.startswith("##"):
        rest = body.split("\n", 1)
        body = rest[1].strip() if len(rest) > 1 else ""
    if not fact_id and header.startswith("##"):
        parts = header[2:].strip().split()
        if parts and re.fullmatch(r"[0-9a-fA-F]{6,}", parts[-1]):
            fact_id = parts[-1]
    return {
        "id": fact_id,
        "date": date,
        "tags": tags,
        "header": header,
        "body": body,
        "raw": stripped,
    }


def _format_fact_section(fact_id: str, date: str, body: str, tags: str = "") -> str:
    tag_part = f" tags: {tags}" if tags else ""
    return (
        f"## {date} {fact_id}\n"
        f"<!-- fact_id: {fact_id} date: {date}{tag_part} -->\n"
        f"{body.strip()}\n"
    )


def _load_fact_sections() -> list[str]:
    if not FACTS_FILE.exists():
        return []
    try:
        text = FACTS_FILE.read_text(encoding="utf-8")
    except Exception:
        return []
    return _split_md_sections(text)


def remember_fact(fact: str) -> str | None:
    """Сохранить факт. Возвращает id или None."""
    if not USE_MEMORY:
        return None
    fact = clean_memory_text(fact or "")
    if not fact:
        return None
    _ensure_memory_dir()
    fact_id = uuid.uuid4().hex[:10]
    date = datetime.now().strftime("%Y-%m-%d")
    block = _format_fact_section(fact_id, date, fact)
    with open(FACTS_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + block)
    _index_file(FACTS_FILE)
    maybe_refresh_profile(force=True)
    return fact_id


@dataclass
class ForgetResult:
    removed: int = 0
    ambiguous: list[str] = field(default_factory=list)


def forget_fact(pattern: str) -> ForgetResult:
    """
    Удалить факт по id или по уникальному совпадению текста.
    Несколько совпадений — ничего не удаляет, возвращает список кандидатов.
    """
    if not USE_MEMORY:
        return ForgetResult()
    pattern = (pattern or "").strip()
    if not pattern:
        return ForgetResult()
    sections = _load_fact_sections()
    if not sections:
        return ForgetResult()
    parsed = [_parse_fact_section(s) for s in sections]
    needle = pattern.lower()
    matches = []
    for i, item in enumerate(parsed):
        if item["id"] and item["id"].lower() == needle:
            matches = [i]
            break
        blob = f"{item['header']}\n{item['body']}".lower()
        if needle in blob:
            matches.append(i)
    if not matches:
        return ForgetResult()
    if len(matches) > 1:
        previews = []
        for i in matches:
            item = parsed[i]
            fid = item["id"] or "?"
            snippet = _norm_memory_text(item["body"] or item["header"])[:80]
            previews.append(f"{fid}: {snippet}")
        return ForgetResult(ambiguous=previews)
    kept = [parsed[i]["raw"] for i in range(len(parsed)) if i not in matches]
    new_text = "\n\n".join(kept)
    if new_text.strip():
        FACTS_FILE.write_text(new_text.strip() + "\n", encoding="utf-8")
    else:
        FACTS_FILE.write_text("", encoding="utf-8")
    _index_file(FACTS_FILE)
    maybe_refresh_profile(force=True)
    return ForgetResult(removed=1)


def _norm_memory_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _dialogue_files() -> list[Path]:
    _ensure_memory_dir()
    if not DIALOGUE_DIR.exists():
        return []
    return sorted(DIALOGUE_DIR.glob("*.md"))


def _parse_dialogue_section(section: str) -> dict:
    ids = MSG_IDS_RE.search(section)
    user_id = asst_id = None
    if ids:
        user_id, asst_id = ids.group(1), ids.group(2)
        if user_id in ("-", "none", "null"):
            user_id = None
        if asst_id in ("-", "none", "null"):
            asst_id = None
    user_m = re.search(
        r"\*\*Пользователь:\*\*\s*(.*?)(?=\n\*\*Ассистент:\*\*|\Z)",
        section,
        re.S,
    )
    asst_m = re.search(r"\*\*Ассистент:\*\*\s*(.*)\Z", section, re.S)
    header = section.split("\n", 1)[0].strip()
    return {
        "header": header,
        "user_id": user_id,
        "assistant_id": asst_id,
        "user": (user_m.group(1).strip() if user_m else ""),
        "assistant": (asst_m.group(1).strip() if asst_m else ""),
        "raw": section,
    }


def _format_dialogue_section(header: str, user_id, assistant_id, user: str, assistant: str) -> str:
    if not (user or "").strip() and not (assistant or "").strip():
        return ""
    uid = user_id or "-"
    aid = assistant_id or "-"
    lines = [header, f"<!-- msg_ids: {uid} {aid} -->"]
    if (user or "").strip():
        lines.append(f"**Пользователь:** {user.strip()}")
        lines.append("")
    if (assistant or "").strip():
        lines.append(f"**Ассистент:** {assistant.strip()}")
    return "\n".join(lines).rstrip() + "\n"


def _write_dialogue_file(path: Path, sections: list[str]) -> None:
    kept = [s.strip() + "\n" for s in sections if s and s.strip()]
    path.write_text("\n".join(kept), encoding="utf-8")


def forget_dialogue_entries(entries: list) -> int:
    """
    Remove matching user/assistant utterances from dialogue markdown (long-term memory).
    Matches by message id when present, otherwise by normalized text.
    Returns the number of utterance parts removed.
    """
    if not USE_MEMORY or not entries:
        return 0
    user_ids = set()
    asst_ids = set()
    user_texts = set()
    asst_texts = set()
    for item in entries:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").lower()
        if role not in ("user", "assistant"):
            continue
        mid = item.get("id")
        text = _norm_memory_text(item.get("memory_text") or "")
        if not text:
            content = item.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text") or ""))
                    elif isinstance(part, str):
                        parts.append(part)
                text = _norm_memory_text(" ".join(parts))
            elif isinstance(content, str):
                text = _norm_memory_text(content)
        if role == "user":
            if mid:
                user_ids.add(str(mid))
            if text:
                user_texts.add(text)
        else:
            if mid:
                asst_ids.add(str(mid))
            if text:
                asst_texts.add(text)
    if not (user_ids or asst_ids or user_texts or asst_texts):
        return 0

    removed = 0
    changed_paths: list[Path] = []
    for path in _dialogue_files():
        try:
            original = path.read_text(encoding="utf-8")
        except Exception:
            continue
        sections = _split_md_sections(original)
        kept = []
        changed = False
        for section in sections:
            stripped = section.strip()
            is_pair = "**Пользователь:**" in stripped or "**Ассистент:**" in stripped or "Диалог " in stripped
            if is_pair:
                parsed = _parse_dialogue_section(stripped)
                drop_user = False
                drop_asst = False
                if parsed["user_id"] and parsed["user_id"] in user_ids:
                    drop_user = True
                if parsed["assistant_id"] and parsed["assistant_id"] in asst_ids:
                    drop_asst = True
                if parsed["user"] and _norm_memory_text(parsed["user"]) in user_texts:
                    drop_user = True
                if parsed["assistant"] and _norm_memory_text(parsed["assistant"]) in asst_texts:
                    drop_asst = True
                new_user = "" if drop_user else parsed["user"]
                new_asst = "" if drop_asst else parsed["assistant"]
                new_uid = None if drop_user else parsed["user_id"]
                new_aid = None if drop_asst else parsed["assistant_id"]
                if drop_user:
                    removed += 1
                    changed = True
                if drop_asst:
                    removed += 1
                    changed = True
                rebuilt = _format_dialogue_section(
                    parsed["header"], new_uid, new_aid, new_user, new_asst
                )
                if rebuilt:
                    kept.append(rebuilt)
                elif stripped:
                    changed = True
                continue
            lines = stripped.split("\n", 1)
            body = lines[1].strip() if len(lines) > 1 else ""
            if body and _norm_memory_text(body) in asst_texts:
                removed += 1
                changed = True
                continue
            kept.append(stripped + "\n")
        if changed:
            _write_dialogue_file(path, kept)
            changed_paths.append(path)
    for path in changed_paths:
        _schedule_index_file(path)
    return removed


def _replace_dialogue_assistant(user_id: str, user: str, assistant: str, assistant_id: str | None) -> Path | None:
    if not user_id:
        return None
    for path in _dialogue_files():
        try:
            original = path.read_text(encoding="utf-8")
        except Exception:
            continue
        sections = _split_md_sections(original)
        changed = False
        kept = []
        for section in sections:
            parsed = _parse_dialogue_section(section.strip())
            if parsed["user_id"] == user_id:
                rebuilt = _format_dialogue_section(
                    parsed["header"],
                    user_id,
                    assistant_id,
                    user or parsed["user"],
                    assistant,
                )
                if rebuilt:
                    kept.append(rebuilt)
                changed = True
            else:
                kept.append(section.strip() + "\n")
        if changed:
            _write_dialogue_file(path, kept)
            return path
    return None


def save_assistant_utterance(text: str) -> None:
    """Добавить реплику ассистента в лог диалога (по дате)."""
    if not USE_MEMORY:
        return
    text = clean_memory_text(text or "")
    if not text:
        return
    _ensure_memory_dir()
    path = _get_dialogue_file()
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().isoformat()}\n{text}\n")
    _schedule_index_file(path)
    maybe_refresh_profile()


def save_dialogue(
    user: str,
    assistant: str,
    user_id: str | None = None,
    assistant_id: str | None = None,
) -> None:
    """Сохранить пару реплик пользователя и ассистента в лог диалога."""
    if not USE_MEMORY:
        return
    user = clean_memory_text(user or "")
    assistant = clean_memory_text(assistant or "")
    if not user and not assistant:
        return
    replaced = _replace_dialogue_assistant(user_id, user, assistant, assistant_id) if user_id else None
    if replaced is not None:
        _schedule_index_file(replaced)
        maybe_refresh_profile()
        return
    _ensure_memory_dir()
    path = _get_dialogue_file()
    ts = datetime.now().isoformat()
    header = f"## Диалог {ts}"
    block = _format_dialogue_section(header, user_id, assistant_id, user, assistant)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + block)
    _schedule_index_file(path)
    maybe_refresh_profile()
