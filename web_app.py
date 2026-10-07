import asyncio
import base64
import io
import os
import secrets
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict

from avatar import HeadlessAvatar, is_allowed_avatar_file
from config import WEB_HOST, WEB_PORT
from file_utils import (
    PROJECT_ROOT,
    build_content_parts,
    client_attachments_to_files,
    load_file_as_base64,
    parse_file_paths_from_input,
)
from handler import (
    CONTROL_MARKERS_RE,
    NO_ANSWERS,
    SessionUI,
    YES_ANSWERS,
    delete_chat_message,
    find_history_index,
    generate_chat,
    new_message_id,
    process_turn,
    regenerate_assistant,
    rewrite_user_message,
    trim_history,
)
from voice import (
    clean_text_for_tts,
    get_tts_audio,
    init_tts,
    stop_speaking,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI()

AUTH_COOKIE_NAME = "coreline_auth"
SESSION_MAX_AGE = 60 * 60 * 12
AUTH_USERNAME = ""
AUTH_PASSWORD = ""
_sessions: dict[str, float] = {}
_sessions_lock = threading.Lock()

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

try:
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    USE_TEMPLATES = True
except ImportError:
    from jinja2 import Environment, FileSystemLoader
    jinja_env = Environment(loader=FileSystemLoader("templates"))
    USE_TEMPLATES = False

connections: Dict[WebSocket, Dict] = {}


def _ensure_login_credentials():
    """Require WEB_LOGIN_* from the environment. Create them in .env on first run."""
    global AUTH_USERNAME, AUTH_PASSWORD
    AUTH_USERNAME = os.getenv("WEB_LOGIN_USERNAME", "").strip()
    AUTH_PASSWORD = os.getenv("WEB_LOGIN_PASSWORD", "")
    if AUTH_USERNAME and AUTH_PASSWORD:
        return
    username = AUTH_USERNAME or "reizi"
    password = secrets.token_urlsafe(18)
    env_path = Path(".env")
    with open(env_path, "a", encoding="utf-8") as f:
        f.write(f"\nWEB_LOGIN_USERNAME={username}\nWEB_LOGIN_PASSWORD={password}\n")
    os.environ["WEB_LOGIN_USERNAME"] = username
    os.environ["WEB_LOGIN_PASSWORD"] = password
    AUTH_USERNAME = username
    AUTH_PASSWORD = password
    print("Created WEB_LOGIN_USERNAME / WEB_LOGIN_PASSWORD in .env (needed for the web UI).")


def _new_session() -> str:
    token = secrets.token_urlsafe(32)
    with _sessions_lock:
        _sessions[token] = time.time() + SESSION_MAX_AGE
    return token


def _valid_session(token: str | None) -> bool:
    if not token:
        return False
    now = time.time()
    with _sessions_lock:
        exp = _sessions.get(token)
        if exp is None or exp < now:
            _sessions.pop(token, None)
            return False
        return True


def _revoke_session(token: str | None) -> None:
    if not token:
        return
    with _sessions_lock:
        _sessions.pop(token, None)


def _cookie_kwargs(request: Request, token: str) -> dict:
    return {
        "key": AUTH_COOKIE_NAME,
        "value": token,
        "httponly": True,
        "samesite": "strict",
        "max_age": SESSION_MAX_AGE,
        "secure": request.url.scheme == "https",
    }


def load_system_prompt(prompt_name: str = "system.txt"):
    with open(f"prompts/{prompt_name}", "r", encoding="utf-8") as f:
        return f.read()


def _render(request: Request, name: str, context: dict):
    if USE_TEMPLATES:
        return templates.TemplateResponse(request, name, context)
    template = jinja_env.get_template(name)
    return HTMLResponse(content=template.render(**context))


async def send_message_to_client(websocket: WebSocket, message_type: str, data: dict):
    try:
        await websocket.send_json({
            "type": message_type,
            "data": data
        })
    except Exception as e:
        print(f"Ошибка отправки сообщения: {e}")


async def send_audio_to_client(websocket: WebSocket, text: str, voice_enabled: bool, speaker_wav=None):
    if not voice_enabled:
        return

    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return

    sent_any = False
    for line in lines:
        clean_line = clean_text_for_tts(line).strip()
        if not clean_line:
            continue
        audio_bytes = await asyncio.to_thread(generate_audio_for_web, clean_line, speaker_wav)
        if audio_bytes:
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            await send_message_to_client(websocket, "audio", {
                "data": audio_base64,
                "format": "wav"
            })
            sent_any = True

    if not sent_any:
        await send_message_to_client(websocket, "message", {
            "role": "system",
            "content": "Не удалось сгенерировать аудио."
        })


def generate_audio_for_web(text: str, speaker_wav=None, language=None):
    import wave

    result = get_tts_audio(text, speaker_wav=speaker_wav, language=language)
    if result is None:
        return None

    wav, sample_rate = result
    try:
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        if len(wav.shape) > 1:
            wav = wav[:, 0] if wav.shape[1] > 0 else wav.flatten()
        else:
            wav = wav.flatten()
        if wav.size == 0:
            return None
        if wav.max() > 1.0 or wav.min() < -1.0:
            wav = wav / np.max(np.abs(wav))
        wav_int16 = (wav * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(wav_int16.tobytes())
        wav_buffer.seek(0)
        return wav_buffer.read()
    except Exception as e:
        print(f"Ошибка генерации аудио: {e}")
        import traceback
        traceback.print_exc()
        return None


class WebUI(SessionUI):
    file_restrict_root = PROJECT_ROOT
    stt_is_browser = True

    def __init__(self, websocket: WebSocket, state: dict):
        self.websocket = websocket
        self.state = state

    async def message(self, role, content, msg_id=None):
        ws_role = "assistant" if role == "coreline" else role
        payload = {
            "role": ws_role,
            "content": content,
        }
        if msg_id:
            payload["id"] = msg_id
        elif ws_role == "assistant":
            pending = self.state.get("pending_assistant_id")
            if pending:
                payload["id"] = pending
        await send_message_to_client(self.websocket, "message", payload)

    async def avatar(self, state):
        await send_message_to_client(self.websocket, "avatar", {"state": state})

    async def stream_start(self):
        payload = {"role": "assistant"}
        pending = self.state.get("pending_assistant_id")
        if pending:
            payload["id"] = pending
        await send_message_to_client(self.websocket, "message_start", payload)

    async def stream_chunk(self, chunk):
        await send_message_to_client(self.websocket, "message_chunk", {"content": chunk})

    async def confirm_commands(self, commands):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.state["pending_confirm"] = fut
        await send_message_to_client(self.websocket, "run_confirm", {
            "commands": commands,
            "prompt": "Выполнить команды? Ответьте *y чтобы выполнить все, *n чтобы отменить.",
        })
        try:
            ok = await fut
            return list(commands) if ok else []
        except asyncio.CancelledError:
            return []
        finally:
            self.state.pop("pending_confirm", None)

    async def speak(self, text):
        await send_audio_to_client(self.websocket, text, True, speaker_wav=None)

    def init_tts(self):
        init_tts()

    def stop_speaking(self):
        stop_speaking()


@app.on_event("startup")
def _startup_auth():
    _ensure_login_credentials()


@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    return _render(request, "login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def post_login(request: Request, username: str = Form(...), password: str = Form(...)):
    user_ok = secrets.compare_digest(username, AUTH_USERNAME)
    pass_ok = secrets.compare_digest(password, AUTH_PASSWORD)
    if user_ok and pass_ok:
        token = _new_session()
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(**_cookie_kwargs(request, token))
        return response

    return _render(request, "login.html", {"request": request, "error": "Неверный логин или пароль"})


@app.get("/logout")
async def logout(request: Request):
    _revoke_session(request.cookies.get(AUTH_COOKIE_NAME))
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(AUTH_COOKIE_NAME)
    return response


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    if not _valid_session(request.cookies.get(AUTH_COOKIE_NAME)):
        return RedirectResponse(url="/login", status_code=302)
    return _render(request, "index.html", {"request": request})


@app.get("/rp", response_class=HTMLResponse)
async def get_rp(request: Request):
    if not _valid_session(request.cookies.get(AUTH_COOKIE_NAME)):
        return RedirectResponse(url="/login", status_code=302)
    return _render(request, "rp.html", {"request": request})


@app.get("/avatars/{filename}")
async def get_avatar(filename: str):
    safe = Path(filename).name
    if not is_allowed_avatar_file(safe):
        raise HTTPException(status_code=404, detail="Avatar not found")
    avatar_path = Path("avatars") / safe
    if not avatar_path.is_file():
        raise HTTPException(status_code=404, detail="Avatar not found")
    return FileResponse(avatar_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not _valid_session(websocket.cookies.get(AUTH_COOKIE_NAME)):
        await websocket.close(code=1008)
        return

    await websocket.accept()

    avatar = HeadlessAvatar(web=True)
    system_prompt = load_system_prompt("system.txt")
    state = {
        "avatar": avatar,
        "history": [{"role": "system", "content": system_prompt}],
        "web_enabled": False,
        "run_enabled": False,
        "voice_enabled": False,
        "listen_enabled": False,
    }
    connections[websocket] = state
    ui = WebUI(websocket, state)
    incoming: asyncio.Queue = asyncio.Queue()

    await ui.avatar("idle")
    await ui.message("system", "CoreLine готова к общению. Введите *help для справки.")

    async def reader():
        try:
            while True:
                data = await websocket.receive_json()
                fut = state.get("pending_confirm")
                if (
                    fut is not None
                    and not fut.done()
                    and data.get("type") == "message"
                ):
                    low = str(data.get("content", "")).strip().lower()
                    if low in YES_ANSWERS or low in NO_ANSWERS:
                        fut.set_result(low in YES_ANSWERS)
                        continue
                await incoming.put(data)
        except WebSocketDisconnect:
            fut = state.get("pending_confirm")
            if isinstance(fut, asyncio.Future) and not fut.done():
                fut.cancel()
            await incoming.put(None)

    async def processor():
        while True:
            data = await incoming.get()
            if data is None:
                break
            data_type = data.get("type")
            try:
                if data_type == "delete":
                    msg_id = str(data.get("id") or "")
                    truncated = await delete_chat_message(msg_id, state, forget_memory=True)
                    if truncated:
                        await send_message_to_client(websocket, "truncate", {"id": truncated})
                    else:
                        await ui.message("system", "Сообщение не найдено.")
                    continue

                if data_type == "rewrite":
                    msg_id = str(data.get("id") or "")
                    new_text = str(data.get("content") or "").strip()
                    attachments = data.get("attachments") or []
                    if not new_text and not attachments:
                        await ui.message("system", "Нечего сохранять.")
                        continue
                    idx = find_history_index(state["history"], msg_id)
                    if idx is None or state["history"][idx].get("role") != "user":
                        await ui.message("system", "Не удалось переписать сообщение.")
                        continue
                    await send_message_to_client(websocket, "truncate", {"id": msg_id})
                    user_id = new_message_id()
                    state["pending_user_id"] = user_id
                    display_content = new_text
                    if attachments:
                        display_content = (new_text + " " if new_text else "") + f"[📎 {len(attachments)} файл(ов)]"
                    await send_message_to_client(websocket, "message", {
                        "role": "user",
                        "content": display_content,
                        "id": user_id,
                    })
                    keep, _ = await rewrite_user_message(
                        msg_id, new_text, state, ui, attachments, forget_memory=True
                    )
                    if not keep:
                        await websocket.close()
                        break
                    continue

                if data_type == "regenerate":
                    msg_id = str(data.get("id") or "")
                    idx = find_history_index(state["history"], msg_id)
                    if idx is None or state["history"][idx].get("role") != "assistant":
                        await ui.message("system", "Нечего перегенерировать.")
                        continue
                    await send_message_to_client(websocket, "truncate", {"id": msg_id})
                    await regenerate_assistant(msg_id, state, ui, forget_memory=True)
                    continue

                if data_type != "message":
                    continue
                user_input = data.get("content", "").strip()
                attachments = data.get("attachments", [])
                if not user_input and not attachments:
                    continue
                display_content = user_input
                if attachments:
                    display_content = (user_input + " " if user_input else "") + f"[📎 {len(attachments)} файл(ов)]"
                payload = {
                    "role": "user",
                    "content": display_content,
                }
                looks_like_command = user_input.startswith("*") or user_input.startswith("поиск")
                if not looks_like_command:
                    user_id = new_message_id()
                    state["pending_user_id"] = user_id
                    payload["id"] = user_id
                await send_message_to_client(websocket, "message", payload)
                await ui.avatar("thinking")
                keep, _ = await process_turn(user_input, state, ui, attachments)
                if not keep:
                    await websocket.close()
                    break
            finally:
                await send_message_to_client(websocket, "turn_done", {})

    try:
        await asyncio.gather(reader(), processor())
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Ошибка WebSocket: {e}")
    finally:
        fut = state.get("pending_confirm")
        if isinstance(fut, asyncio.Future) and not fut.done():
            fut.cancel()
        connections.pop(websocket, None)


async def _rp_complete_turn(websocket, ui, state, user_input, attachments, user_id):
    text_only, file_paths = parse_file_paths_from_input(user_input)
    user_to_send = text_only if text_only else user_input.strip()
    file_results = []
    failed_paths = []
    for fp in file_paths:
        loaded = load_file_as_base64(fp, restrict_to=PROJECT_ROOT)
        if loaded:
            file_results.append(loaded)
        else:
            failed_paths.append(fp)
    file_results.extend(client_attachments_to_files(attachments))
    if failed_paths:
        await ui.message("system", f"Не удалось загрузить файлы: {', '.join(failed_paths)}")
        return
    if not user_to_send and not file_results:
        await ui.message("system", "Укажите текст сообщения и/или прикрепите файлы.")
        return

    user_content = build_content_parts(user_to_send or "(прикреплённые файлы)", file_results)
    asst_id = new_message_id()
    state["pending_assistant_id"] = asst_id
    messages_to_send = list(state["history"]) + [{"role": "user", "content": user_content}]
    response = await generate_chat(messages_to_send, ui, use_tools=False)
    response = CONTROL_MARKERS_RE.sub("", response).strip()
    await ui.message("coreline", response, msg_id=asst_id)
    state["history"].append({
        "role": "user",
        "content": user_content,
        "id": user_id,
        "memory_text": user_to_send or user_input,
    })
    state["history"].append({
        "role": "assistant",
        "content": response,
        "id": asst_id,
        "memory_text": response,
    })
    trim_history(state["history"])
    state.pop("pending_assistant_id", None)


@app.websocket("/ws_rp")
async def websocket_rp_endpoint(websocket: WebSocket):
    if not _valid_session(websocket.cookies.get(AUTH_COOKIE_NAME)):
        await websocket.close(code=1008)
        return

    await websocket.accept()

    system_prompt = load_system_prompt("system_rp.txt")
    state = {
        "history": [{"role": "system", "content": system_prompt}],
        "rp_mode": True,
    }
    connections[websocket] = state
    ui = WebUI(websocket, state)

    await ui.message("system", "RP-режим включён. Память отключена.")

    try:
        while True:
            data = await websocket.receive_json()
            data_type = data.get("type")

            if data_type == "init_history":
                incoming_history = data.get("history", [])
                safe_history = []
                for item in incoming_history:
                    role = item.get("role")
                    content = item.get("content")
                    if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                        entry = {"role": role, "content": content, "memory_text": content}
                        msg_id = item.get("id")
                        entry["id"] = str(msg_id) if msg_id else new_message_id()
                        safe_history.append(entry)
                state["history"] = [{"role": "system", "content": system_prompt}] + safe_history
                trim_history(state["history"])
                continue

            if data_type == "clear_history":
                state["history"] = [{"role": "system", "content": system_prompt}]
                await ui.message("system", "История RP-чата очищена.")
                continue

            if data_type == "delete":
                msg_id = str(data.get("id") or "")
                truncated = await delete_chat_message(msg_id, state, forget_memory=False)
                if truncated:
                    await send_message_to_client(websocket, "truncate", {"id": truncated})
                else:
                    await ui.message("system", "Сообщение не найдено.")
                await send_message_to_client(websocket, "turn_done", {})
                continue

            if data_type == "rewrite":
                msg_id = str(data.get("id") or "")
                new_text = str(data.get("content") or "").strip()
                attachments = data.get("attachments") or []
                if not new_text and not attachments:
                    await ui.message("system", "Нечего сохранять.")
                    await send_message_to_client(websocket, "turn_done", {})
                    continue
                idx = find_history_index(state["history"], msg_id)
                if idx is None or state["history"][idx].get("role") != "user":
                    await ui.message("system", "Не удалось переписать сообщение.")
                    await send_message_to_client(websocket, "turn_done", {})
                    continue
                await send_message_to_client(websocket, "truncate", {"id": msg_id})
                await delete_chat_message(msg_id, state, forget_memory=False)
                user_id = new_message_id()
                display_content = new_text
                if attachments:
                    display_content = (new_text + " " if new_text else "") + f"[📎 {len(attachments)} файл(ов)]"
                await send_message_to_client(websocket, "message", {
                    "role": "user",
                    "content": display_content,
                    "id": user_id,
                })
                await _rp_complete_turn(websocket, ui, state, new_text, attachments, user_id)
                await send_message_to_client(websocket, "turn_done", {})
                continue

            if data_type == "regenerate":
                msg_id = str(data.get("id") or "")
                idx = find_history_index(state["history"], msg_id)
                if idx is None or state["history"][idx].get("role") != "assistant":
                    await ui.message("system", "Нечего перегенерировать.")
                    await send_message_to_client(websocket, "turn_done", {})
                    continue
                prev = state["history"][idx - 1] if idx > 0 else None
                if not prev or prev.get("role") != "user":
                    await ui.message("system", "Нет пользовательского сообщения для ответа.")
                    await send_message_to_client(websocket, "turn_done", {})
                    continue
                await send_message_to_client(websocket, "truncate", {"id": msg_id})
                await delete_chat_message(msg_id, state, forget_memory=False)
                asst_id = new_message_id()
                state["pending_assistant_id"] = asst_id
                response = await generate_chat(list(state["history"]), ui, use_tools=False)
                response = CONTROL_MARKERS_RE.sub("", response).strip()
                await ui.message("coreline", response, msg_id=asst_id)
                state["history"].append({
                    "role": "assistant",
                    "content": response,
                    "id": asst_id,
                    "memory_text": response,
                })
                trim_history(state["history"])
                state.pop("pending_assistant_id", None)
                await send_message_to_client(websocket, "turn_done", {})
                continue

            if data_type != "message":
                continue

            user_input = data.get("content", "").strip()
            attachments = data.get("attachments", [])
            if not user_input and not attachments:
                continue

            display_content = user_input
            if attachments:
                display_content = (user_input + " " if user_input else "") + f"[📎 {len(attachments)} файл(ов)]"
            user_id = new_message_id()
            await send_message_to_client(websocket, "message", {
                "role": "user",
                "content": display_content,
                "id": user_id,
            })
            await _rp_complete_turn(websocket, ui, state, user_input, attachments, user_id)
            await send_message_to_client(websocket, "turn_done", {})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Ошибка RP WebSocket: {e}")
    finally:
        connections.pop(websocket, None)


if __name__ == "__main__":
    import uvicorn
    _ensure_login_credentials()
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)
