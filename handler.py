"""Shared chat turn handling for CLI, web, and TUI."""
from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from pathlib import Path
from queue import Queue
from typing import Any

from api import TEXT_STREAMING, extract_final_response, send_message, send_message_stream
from avatar import get_emotion_from_mood
from config import MAX_HISTORY_MESSAGES, WEB_SEARCH_SYSTEM_PROMPT, WEB_SEARCH_VIA_MCP
from web_search import (
    extract_fetch_urls,
    extract_search_queries,
    extract_urls,
    fetch_pages,
    has_fetch_intent,
    leftover_search_query,
    search_web,
)
from file_utils import (
    PROJECT_ROOT,
    build_content_parts,
    client_attachments_to_files,
    load_file_as_base64,
    parse_file_paths_from_input,
)
from memory import (
    build_memory_query,
    clean_memory_text,
    extract_memory_queries,
    forget_dialogue_entries,
    forget_fact,
    load_profile,
    memory_search_needed,
    remember_fact,
    save_dialogue,
    search_memory,
)
from searxng_ctl import start_searxng, stop_searxng

YES_ANSWERS = {"y", "yes", "д", "да", "*y"}
NO_ANSWERS = {"n", "no", "н", "нет", "*n"}
CONTROL_MARKERS_RE = re.compile(
    r"\[SEARCH:\s*[^\]]+\]|"
    r"\[(?:FETCH|OPEN|READ|PAGE|URL|MEMORY):\s*[^\]]+\]|"
    r"\[TIME\]|\[RUN:\s*[^\]]+\]",
    re.IGNORECASE,
)
HELP_TEXT = "\n".join([
    "*help — показать эту справку",
    "поиск <запрос> — поиск по памяти",
    "*remember: <факт> — сохранить важный факт в память (с id)",
    "*forget: <id или уникальная формулировка> — удалить факт",
    "*-m — отключить память на один запрос",
    "*won / *woff — интернет: запуск/остановка SearXNG",
    "посмотри ссылку <url> — открыть конкретную страницу",
    "*runon / *runoff — выполнение [RUN: команда] (без shell, без пайпов)",
    "*file <путь> — прикрепить файл (в вебе также кнопка скрепки)",
    "*voiceon / *voiceoff — голосовой вывод (RHVoice)",
    "*listenon / *listenoff — голосовой ввод",
    "*sleep / *спать — режим сна",
    "TUI: *d [N] удалить, *e [N] [текст] переписать, *r [N] перегенерировать",
])


def trim_history(history: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """Keep the leading system prompt and the newest messages. Mutates `history`."""
    if not history or len(history) <= max_messages:
        return history
    leading = []
    rest = list(history)
    if rest and rest[0].get("role") == "system":
        leading.append(rest.pop(0))
    keep = max(0, max_messages - len(leading))
    history[:] = leading + rest[-keep:]
    return history


def current_time_block() -> str:
    return f"Текущие дата и время: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"


def new_message_id() -> str:
    return str(uuid.uuid4())


def message_plain_text(msg: dict | None) -> str:
    if not msg:
        return ""
    stored = msg.get("memory_text")
    if stored:
        return str(stored).strip()
    content = msg.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and (part.get("type") == "text" or "text" in part):
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts).strip()
    return str(content or "").strip()


def find_history_index(history: list, msg_id: str | None) -> int | None:
    if not msg_id:
        return None
    for i, item in enumerate(history):
        if item.get("id") == msg_id and item.get("role") in ("user", "assistant"):
            return i
    return None


def drop_history_from(history: list, index: int, forget_memory: bool) -> list:
    """Remove history[index:] from the session and optionally from long-term memory."""
    if index < 0 or index >= len(history):
        return []
    dropped = list(history[index:])
    history[index:] = []
    to_forget = [m for m in dropped if m.get("role") in ("user", "assistant")]
    if forget_memory and to_forget:
        forget_dialogue_entries(to_forget)
    return to_forget


class SessionUI:
    """Async UI adapter. Override the methods the front-end needs."""

    file_restrict_root: Path | None = None
    stt_is_browser: bool = False

    async def message(self, role: str, content: str, msg_id: str | None = None) -> None:
        return None

    async def avatar(self, state: str) -> None:
        return None

    async def stream_start(self) -> None:
        return None

    async def stream_chunk(self, chunk: str) -> None:
        return None

    async def stream_end(self) -> None:
        return None

    async def confirm_commands(self, commands: list[str]) -> list[str]:
        return []

    async def speak(self, text: str) -> None:
        return None

    async def modes_changed(self, web_enabled: bool, run_enabled: bool) -> None:
        return None

    def voice_import_error(self) -> str | None:
        return None

    def init_tts(self) -> None:
        return None

    def init_stt(self) -> bool:
        return True

    def stop_speaking(self) -> None:
        return None


def _emotion_from_response(response: str, avatar: Any) -> str:
    mood_match = re.search(r"\[настроение:\s*([^\]]+)\]", response)
    if not mood_match:
        return "idle"
    mood_text = mood_match.group(1).strip()
    getter = getattr(avatar, "get_emotion_from_mood", None)
    if callable(getter):
        return getter(mood_text)
    return get_emotion_from_mood(mood_text)


async def generate_chat(
    messages: list,
    ui: SessionUI,
    use_tools: bool = False,
) -> str:
    """One LLM call; streams when TEXT_STREAMING is on and tools are off."""
    if TEXT_STREAMING and not use_tools:
        await ui.stream_start()
        chunk_queue: Queue = Queue()

        def stream_producer():
            try:
                for chunk, done in send_message_stream(messages, use_tools=False):
                    chunk_queue.put((chunk, done, None))
            except Exception as exc:  # noqa: BLE001
                chunk_queue.put((None, True, exc))

        threading.Thread(target=stream_producer, daemon=True).start()
        accumulated = ""
        loop = asyncio.get_running_loop()
        while True:
            chunk, done_or_err, err = await loop.run_in_executor(None, chunk_queue.get)
            if err is not None:
                await ui.stream_end()
                raise err
            if chunk:
                accumulated += chunk
                chunk_clean = re.sub(r"\[TIME\]", "", chunk, flags=re.IGNORECASE)
                if chunk_clean:
                    await ui.stream_chunk(chunk_clean)
            if done_or_err:
                break
        await ui.stream_end()
        return extract_final_response(accumulated)

    return await asyncio.to_thread(send_message, messages, use_tools)


async def process_turn(
    user: str,
    state: dict,
    ui: SessionUI,
    attachments: list | None = None,
) -> tuple[bool, str | None]:
    """Handle one user utterance. Returns (keep_running, assistant_text_or_None)."""
    history: list = state["history"]
    avatar = state.get("avatar")
    user_lower = user.lower().strip()

    if user_lower in ["exit", "quit", "*bye", "*пока"]:
        await ui.message("system", "До свидания!")
        return False, None

    if user_lower in ["*sleep", "*спать"]:
        await ui.avatar("sleeping")
        await ui.message("coreline", "*zzz...* (режим сна включён)")
        if state.get("voice_enabled"):
            await ui.speak("zzz...")
        return True, None

    if user.startswith("поиск"):
        parts = user.split(" ", 1)
        query = parts[1].strip() if len(parts) > 1 else ""
        await ui.avatar("idle")
        if not query:
            await ui.message("coreline", "Поиск: укажите запрос, например `поиск погода`")
        else:
            results = search_memory(query)
            if results:
                await ui.message("coreline", "Поиск:\n" + "\n".join(results))
            else:
                await ui.message("coreline", "Поиск: ничего не найдено")
        return True, None

    if user.startswith("*help"):
        await ui.avatar("idle")
        await ui.message("system", HELP_TEXT)
        return True, None

    if user_lower == "*voiceon":
        err = ui.voice_import_error()
        if err:
            await ui.message("system", f"Голосовой вывод недоступен: {err}")
            return True, None
        state["voice_enabled"] = True
        ui.init_tts()
        await ui.avatar("idle")
        await ui.message("system", "Голосовой вывод RHVoice включён.")
        return True, None

    if user_lower == "*voiceoff":
        state["voice_enabled"] = False
        ui.stop_speaking()
        await ui.avatar("idle")
        await ui.message("system", "Голосовой вывод выключен.")
        return True, None

    if user_lower == "*listenon":
        err = ui.voice_import_error()
        if err and not ui.stt_is_browser:
            await ui.message("system", f"Голосовой ввод недоступен: {err}")
            return True, None
        if not ui.stt_is_browser and not ui.init_stt():
            await ui.message("system", "Vosk не загрузился — голосовой ввод будет работать через Google.")
        state["listen_enabled"] = True
        await ui.avatar("idle")
        if ui.stt_is_browser:
            await ui.message("system", "Голосовой ввод доступен через кнопку микрофона в интерфейсе.")
        else:
            await ui.message("system", "Голосовой ввод включён. Говорите в микрофон — я буду слушать.")
        return True, None

    if user_lower == "*listenoff":
        state["listen_enabled"] = False
        await ui.avatar("idle")
        await ui.message("system", "Голосовой ввод выключен.")
        return True, None

    if user.startswith("*remember:"):
        fact = user[len("*remember:"):].strip()
        await ui.avatar("idle")
        fact_id = remember_fact(fact)
        if fact_id:
            await ui.message("system", f"Сохранено в память (id: {fact_id}).")
        else:
            await ui.message("system", "Нечего сохранять.")
        return True, None

    if user.startswith("*forget:"):
        pattern = user[len("*forget:"):].strip()
        await ui.avatar("idle")
        result = forget_fact(pattern)
        if result.ambiguous:
            await ui.message(
                "system",
                "Несколько совпадений — уточните id или формулировку:\n" + "\n".join(result.ambiguous),
            )
        elif result.removed:
            await ui.message("system", f"Удалено фактов: {result.removed}")
        else:
            await ui.message("system", "Ничего не удалено.")
        return True, None

    if user_lower == "*won":
        await ui.avatar("thinking")
        await ui.message("system", "Запускаю SearXNG…")
        ok, detail = await asyncio.to_thread(start_searxng)
        await ui.avatar("idle")
        if ok:
            state["web_enabled"] = True
            await ui.message("system", f"Веб-поиск включен. {detail}")
            await ui.modes_changed(True, state.get("run_enabled", False))
        else:
            state["web_enabled"] = False
            await ui.message("system", f"Веб-поиск не включён: {detail}")
            await ui.modes_changed(False, state.get("run_enabled", False))
        return True, None

    if user_lower == "*woff":
        state["web_enabled"] = False
        await ui.avatar("thinking")
        ok, detail = await asyncio.to_thread(stop_searxng)
        await ui.avatar("idle")
        if ok:
            await ui.message("system", f"Веб-поиск выключен. {detail}")
        else:
            await ui.message("system", f"Веб-поиск выключен в чате, но контейнер не остановился: {detail}")
        await ui.modes_changed(False, state.get("run_enabled", False))
        return True, None

    if user_lower == "*runon":
        state["run_enabled"] = True
        await ui.avatar("idle")
        await ui.message(
            "system",
            "Режим терминала включён. ИИ может выполнять команды по маркеру [RUN: команда].",
        )
        await ui.modes_changed(state.get("web_enabled", False), True)
        return True, None

    if user_lower == "*runoff":
        state["run_enabled"] = False
        await ui.avatar("idle")
        await ui.message("system", "Режим терминала выключен.")
        await ui.modes_changed(state.get("web_enabled", False), False)
        return True, None

    use_memory = True
    user_clean = user
    if "*-m" in user_clean:
        use_memory = False
        user_clean = user_clean.replace("*-m", "").strip()

    text_only, file_paths = parse_file_paths_from_input(user_clean)
    user_to_send = text_only if text_only else user_clean.strip()
    file_results = []
    failed_paths = []
    restrict = ui.file_restrict_root
    for fp in file_paths:
        loaded = load_file_as_base64(fp, restrict_to=restrict)
        if loaded:
            file_results.append(loaded)
        else:
            failed_paths.append(fp)
    file_results.extend(client_attachments_to_files(attachments))
    if failed_paths:
        await ui.avatar("idle")
        await ui.message("system", f"Не удалось загрузить файлы: {', '.join(failed_paths)}")
        return True, None
    if not user_to_send and not file_results:
        await ui.avatar("idle")
        await ui.message("system", "Укажите текст сообщения и/или пути к файлам (*file путь)")
        return True, None

    user_content = build_content_parts(user_to_send or "(прикреплённые файлы)", file_results)
    user_plain = user_to_send or user_clean
    user_id = state.pop("pending_user_id", None) or new_message_id()
    asst_id = new_message_id()
    state["pending_assistant_id"] = asst_id
    extra_user = {"role": "user", "content": user_content}
    response = await _generate_reply(
        state, ui, extra_user=extra_user, memory_query=user_clean, use_memory=use_memory
    )
    await _commit_assistant_turn(
        state,
        ui,
        response,
        user_plain=user_plain,
        user_id=user_id,
        asst_id=asst_id,
        extra_user=extra_user,
    )
    return True, response


async def _generate_reply(
    state: dict,
    ui: SessionUI,
    extra_user: dict | None,
    memory_query: str,
    use_memory: bool = True,
) -> str:
    """Generate an assistant reply. extra_user is omitted when the user turn is already in history."""
    history: list = state["history"]
    await ui.avatar("thinking")
    web_enabled = bool(state.get("web_enabled"))
    run_enabled = bool(state.get("run_enabled"))

    contextual_messages = list(history)
    contextual_messages.append({"role": "system", "content": current_time_block()})
    searched_memory_query = ""
    if use_memory:
        profile = load_profile()
        if profile:
            contextual_messages.append({
                "role": "system",
                "content": "Профиль (долгосрочная память, опирайся если уместно):\n" + profile,
            })
        contextual_messages.append({
            "role": "system",
            "content": (
                "Если нужны детали из прошлых разговоров или фактов, которых нет в профиле "
                "и текущем чате — добавь `[MEMORY: короткий запрос]`. "
                "Не пиши этот маркер в финальном ответе."
            ),
        })
        if memory_search_needed(memory_query):
            searched_memory_query = build_memory_query(memory_query)
            memory_snippets = search_memory(searched_memory_query)
            if memory_snippets:
                memory_block = (
                    "Контекст памяти (используй при ответе, если релевантно):\n"
                    + "\n".join(memory_snippets)
                )
                contextual_messages.append({"role": "system", "content": memory_block})

    leftover = leftover_search_query(memory_query)
    user_urls = extract_urls(memory_query)
    fetch_only = bool(user_urls) and (
        has_fetch_intent(memory_query) or (len(user_urls) == 1 and not leftover)
    )
    should_fetch_user_urls = bool(user_urls) and (fetch_only or web_enabled)

    if web_enabled:
        contextual_messages.append({
            "role": "system",
            "content": WEB_SEARCH_SYSTEM_PROMPT,
        })
    elif should_fetch_user_urls:
        contextual_messages.append({
            "role": "system",
            "content": (
                "Ниже текст запрошенной страницы. Ответь по нему. "
                "Если нужно открыть ещё одну ссылку, добавь `[FETCH: https://…]`."
            ),
        })

    if should_fetch_user_urls:
        shown = user_urls[0] if len(user_urls) == 1 else f"{user_urls[0]} (+{len(user_urls) - 1})"
        await ui.message("system", f"Открываю ссылку: {shown}")
        page_block = await asyncio.to_thread(fetch_pages, user_urls)
        contextual_messages.append({"role": "system", "content": page_block})

    if web_enabled and leftover and not fetch_only:
        await ui.message("system", f"Ищу в SearXNG: {leftover}")
        search_block = await asyncio.to_thread(search_web, leftover)
        contextual_messages.append({"role": "system", "content": search_block})

    if run_enabled:
        run_instructions = (
            "Тебе доступно выполнение команд в терминале. Добавь в ответ маркер `[RUN: команда]`, "
            "например `[RUN: ls -la]`. Система выполнит команду и вернёт вывод. Одна команда на маркер."
        )
        contextual_messages.append({"role": "system", "content": run_instructions})

    messages_to_send = list(contextual_messages)
    if extra_user is not None:
        messages_to_send.append(extra_user)
    use_tools = web_enabled and WEB_SEARCH_VIA_MCP
    response = await generate_chat(messages_to_send, ui, use_tools=use_tools)

    mem_follow = extract_memory_queries(response) if use_memory else []
    if searched_memory_query:
        already = searched_memory_query.lower()
        mem_follow = [q for q in mem_follow if q.lower() != already]
    if mem_follow:
        response_clean = CONTROL_MARKERS_RE.sub("", response).strip()
        follow_blocks = []
        for q in mem_follow[:2]:
            hits = search_memory(q)
            follow_blocks.append("\n".join(hits) if hits else f"По памяти ничего не найдено: {q}")
        history_with_user = list(history) + ([extra_user] if extra_user is not None else [])
        memory_follow = history_with_user + [
            {"role": "assistant", "content": response_clean},
            {"role": "system", "content": "\n\n".join(follow_blocks)},
            {
                "role": "user",
                "content": "Используй найденную память. Не пиши маркер [MEMORY].",
            },
        ]
        response = await generate_chat(memory_follow, ui, use_tools=False)

    extra_search = extract_search_queries(response) if web_enabled else []
    extra_search = [q for q in extra_search if q.lower() != leftover.lower()]
    extra_fetch = extract_fetch_urls(response)
    extra_fetch = [u for u in extra_fetch if u.lower() not in {x.lower() for x in user_urls}]
    allow_fetch_follow = web_enabled or should_fetch_user_urls
    if extra_search or (extra_fetch and allow_fetch_follow):
        response_clean = CONTROL_MARKERS_RE.sub("", response).strip()
        follow_blocks = []
        for q in extra_search[:2]:
            await ui.message("system", f"Уточняю поиск: {q}")
            follow_blocks.append(await asyncio.to_thread(search_web, q))
        if extra_fetch and allow_fetch_follow:
            await ui.message("system", f"Открываю ссылку: {extra_fetch[0]}")
            follow_blocks.append(await asyncio.to_thread(fetch_pages, extra_fetch[:2]))
        history_with_user = list(history) + ([extra_user] if extra_user is not None else [])
        search_follow = history_with_user + [
            {"role": "assistant", "content": response_clean},
            {"role": "system", "content": "\n\n".join(follow_blocks)},
            {
                "role": "user",
                "content": "Используй новые результаты и текст страниц. Не пиши маркеры [SEARCH]/[FETCH].",
            },
        ]
        response = await generate_chat(search_follow, ui, use_tools=False)

    run_matches = re.findall(r"\[RUN:\s*([^\]]+)\]", response, re.IGNORECASE)
    commands = [c.strip() for c in run_matches if c.strip()]
    if commands and run_enabled:
        response_clean = re.sub(r"\[RUN:\s*[^\]]+\]", "", response, flags=re.IGNORECASE).strip()
        approved = await ui.confirm_commands(commands)
        history_with_user = list(history) + ([extra_user] if extra_user is not None else [])
        if approved:
            run_outputs = []
            for cmd in commands:
                if cmd not in approved:
                    run_outputs.append(f"$ {cmd}\n(отменено пользователем)")
                    continue
                await ui.avatar("thinking")
                await ui.message("system", f"Выполняю: {cmd}")
                out = await asyncio.to_thread(run_shell_command, cmd)
                run_outputs.append(f"$ {cmd}\n{out}")
            run_block = "Вывод терминала:\n\n" + "\n---\n\n".join(run_outputs)
            run_context = history_with_user + [
                {"role": "assistant", "content": response_clean},
                {"role": "system", "content": run_block},
                {
                    "role": "user",
                    "content": "Используй вывод терминала, чтобы дополнить ответ. Не упоминай маркер [RUN], ответь естественно.",
                },
            ]
            response = await generate_chat(run_context, ui, use_tools=False)
        else:
            await ui.message("system", "Отменено.")
            run_context = history_with_user + [
                {"role": "assistant", "content": response_clean},
                {
                    "role": "user",
                    "content": "Пользователь отменил выполнение команд. Дай ответ без вывода терминала, не упоминая отмену.",
                },
            ]
            response = await generate_chat(run_context, ui, use_tools=False)

    return CONTROL_MARKERS_RE.sub("", response).strip()


async def _commit_assistant_turn(
    state: dict,
    ui: SessionUI,
    response: str,
    user_plain: str,
    user_id: str,
    asst_id: str,
    extra_user: dict | None,
) -> str:
    history: list = state["history"]
    avatar = state.get("avatar")
    emotion_state = _emotion_from_response(response, avatar)
    await ui.avatar(emotion_state)
    await ui.message("coreline", response, msg_id=asst_id)

    user_store = clean_memory_text(user_plain)
    asst_store = clean_memory_text(response)
    save_dialogue(user_store, asst_store, user_id=user_id, assistant_id=asst_id)
    if state.get("voice_enabled"):
        await ui.speak(response)

    if extra_user is not None:
        history.append({
            "role": "user",
            "content": extra_user.get("content"),
            "id": user_id,
            "memory_text": user_store,
        })
    history.append({
        "role": "assistant",
        "content": response,
        "id": asst_id,
        "memory_text": asst_store,
    })
    trim_history(history)
    state.pop("pending_assistant_id", None)
    return response


async def delete_chat_message(msg_id: str, state: dict, forget_memory: bool = True) -> str | None:
    """Drop a message and everything after it from session (and long-term memory)."""
    history: list = state["history"]
    idx = find_history_index(history, msg_id)
    if idx is None:
        return None
    drop_history_from(history, idx, forget_memory)
    return msg_id


async def rewrite_user_message(
    msg_id: str,
    new_text: str,
    state: dict,
    ui: SessionUI,
    attachments: list | None = None,
    forget_memory: bool = True,
) -> tuple[bool, str | None]:
    history: list = state["history"]
    idx = find_history_index(history, msg_id)
    if idx is None or history[idx].get("role") != "user":
        await ui.message("system", "Не удалось переписать сообщение.")
        return True, None
    drop_history_from(history, idx, forget_memory)
    if not state.get("pending_user_id"):
        state["pending_user_id"] = new_message_id()
    return await process_turn(new_text, state, ui, attachments)


async def regenerate_assistant(
    msg_id: str,
    state: dict,
    ui: SessionUI,
    forget_memory: bool = True,
) -> str | None:
    history: list = state["history"]
    idx = find_history_index(history, msg_id)
    if idx is None or history[idx].get("role") != "assistant":
        await ui.message("system", "Нечего перегенерировать.")
        return None
    prev = history[idx - 1] if idx > 0 else None
    if not prev or prev.get("role") != "user":
        await ui.message("system", "Нет пользовательского сообщения для ответа.")
        return None
    user_id = prev.get("id") or new_message_id()
    if not prev.get("id"):
        prev["id"] = user_id
    user_plain = message_plain_text(prev)
    drop_history_from(history, idx, forget_memory)
    asst_id = new_message_id()
    state["pending_assistant_id"] = asst_id
    response = await _generate_reply(
        state, ui, extra_user=None, memory_query=user_plain, use_memory=True
    )
    await _commit_assistant_turn(
        state,
        ui,
        response,
        user_plain=user_plain,
        user_id=user_id,
        asst_id=asst_id,
        extra_user=None,
    )
    return response


def run_sync(coro):
    """Run an async turn from a sync front-end (CLI / TUI bridge)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside a loop (should not happen for CLI); run in a worker thread.
    box: dict[str, Any] = {}

    def _target():
        box["result"] = asyncio.run(coro)

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    return box["result"]
