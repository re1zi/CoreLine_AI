import time
import re
import threading
import json
import asyncio
import io
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List
import os

from api import send_message
from avatar_web import AvatarTerminal
from file_utils import build_content_parts, load_file_as_base64, parse_file_paths_from_input
from memory import save_dialogue, search_memory, remember_fact, forget_fact
from run_shell import run_shell_command
from config import WEB_SEARCH_VIA_MCP
from voice import (
    speak_stream,
    stop_speaking,
    init_xtts,
    init_tts,
    get_tts_audio,
    clean_text_for_tts,
    listen,
)

app = FastAPI()

# Простая конфигурация логина
AUTH_USERNAME = os.getenv("WEB_LOGIN_USERNAME", "reizi")
AUTH_PASSWORD = os.getenv("WEB_LOGIN_PASSWORD", "5P5PUSvCmt5V")
AUTH_COOKIE_NAME = "coreline_auth"
AUTH_COOKIE_VALUE = os.getenv("WEB_LOGIN_TOKEN", "ok")

# Создаем директории для статики и шаблонов
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Используем Jinja2 для шаблонов
try:
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    USE_TEMPLATES = True
except ImportError:
    from jinja2 import Environment, FileSystemLoader
    jinja_env = Environment(loader=FileSystemLoader("templates"))
    USE_TEMPLATES = False

# Глобальное состояние для каждого подключения
connections: Dict[WebSocket, Dict] = {}

# Загружаем системный промпт
def load_system_prompt():
    with open("prompts/system.txt", "r", encoding="utf-8") as f:
        return f.read()

def get_avatar_image_path(state):
    """Возвращает путь к изображению аватара для веб-интерфейса"""
    avatar_map = {
        "idle": "idle.png",
        "thinking": "thinking.png",
        "speaking": "speaking.png",
        "sleeping": "sleep.png",
        "joy": "joy.png",
        "satisfaction": "satisfaction.png",
        "indifference": "indifference.png",
        "anger": "anger.png",
        "sadness": "sadness.png",
        "fear": "fear.png",
        "disgust": "disgust.png",
        "surprise": "surprise.png",
        "contempt": "contempt.png",
        "blush": "blush.png"
    }
    return f"/static/avatars/{avatar_map.get(state, 'idle.png')}"

async def send_message_to_client(websocket: WebSocket, message_type: str, data: dict):
    """Отправка сообщения клиенту"""
    try:
        await websocket.send_json({
            "type": message_type,
            "data": data
        })
    except Exception as e:
        print(f"Ошибка отправки сообщения: {e}")

async def send_audio_to_client(websocket: WebSocket, text: str, voice_enabled: bool):
    """Генерирует аудио по строкам (как speak_stream) и отправляет чанки клиенту через WebSocket."""
    if not voice_enabled:
        return

    # Та же логика разбиения по строкам, что и в speak_stream — быстрый отклик, первые фразы звучат раньше
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
        audio_bytes = generate_audio_for_web(clean_line)
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
    """Генерирует аудио (XTTS или Qwen3-TTS) и возвращает WAV байты для отправки в браузер"""
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

async def process_user_input(user: str, websocket: WebSocket, state: Dict, attachments: list | None = None):
    """Обработка ввода пользователя (адаптировано для веб)"""
    user_lower = user.lower().strip()
    avatar = state["avatar"]
    history = state["history"]
    web_enabled = state["web_enabled"]
    run_enabled = state["run_enabled"]
    voice_enabled = state["voice_enabled"]
    listen_enabled = state["listen_enabled"]

    if user_lower in ["exit", "quit", "*bye", "*пока"]:
        await send_message_to_client(websocket, "message", {
            "role": "system",
            "content": "До свидания!"
        })
        return False, None

    # Команда сна
    if user_lower in ["*sleep", "*спать"]:
        await send_message_to_client(websocket, "avatar", {"state": "sleeping"})
        await send_message_to_client(websocket, "message", {
            "role": "assistant",
            "content": "*zzz...* (режим сна включён)"
        })
        if voice_enabled:
            await send_audio_to_client(websocket, "zzz...", voice_enabled)
        return True, None

    # Поиск в памяти
    if user.startswith("поиск"):
        parts = user.split(" ", 1)
        query = parts[1].strip() if len(parts) > 1 else ""
        await send_message_to_client(websocket, "avatar", {"state": "idle"})
        if not query:
            await send_message_to_client(websocket, "message", {
                "role": "system",
                "content": "(укажите запрос, например: 'поиск погода')"
            })
        else:
            results = search_memory(query)
            if results:
                content = "Результаты поиска:\n" + "\n".join(f"  • {r}" for r in results)
            else:
                content = "(ничего не найдено)"
            await send_message_to_client(websocket, "message", {
                "role": "system",
                "content": content
            })
        return True, None

    # Справка
    if user.startswith("*help"):
        await send_message_to_client(websocket, "avatar", {"state": "idle"})
        help_text = """Доступные команды и флаги:
  *help                    — показать эту справку
  поиск <запрос>           — поиск по памяти (BM25)
  *remember: <факт>        — сохранить важный факт в память
  *forget: <подстрока>     — удалить строки памяти по подстроке
  *-m                      — отключить подмешивание памяти на один запрос
  *won                     — включить возможность использования интернета
  *woff                    — выключить возможность использования интернета
  *runon                   — включить выполнение команд в терминале (ИИ может использовать [RUN: команда])
  *runoff                  — выключить выполнение команд в терминале
  *voiceon                 — включить голосовой вывод (TTS)
  *voiceoff                — выключить голосовой вывод
  *listenon                — включить голосовой ввод (STT)
  *listenoff               — выключить голосовой ввод
  *sleep / *спать          — перейти в режим сна (визуально)
  📎 прикрепление файлов   — через кнопку скрепки в поле ввода"""
        await send_message_to_client(websocket, "message", {
            "role": "system",
            "content": help_text
        })
        return True, None

    # Управление TTS
    if user_lower == "*voiceon":
        state["voice_enabled"] = True
        init_xtts()
        await send_message_to_client(websocket, "avatar", {"state": "idle"})
        await send_message_to_client(websocket, "message", {
            "role": "system",
            "content": "Голосовой вывод XTTS включён."
        })
        return True, None
    
    if user_lower == "*voiceoff":
        state["voice_enabled"] = False
        stop_speaking()
        await send_message_to_client(websocket, "avatar", {"state": "idle"})
        await send_message_to_client(websocket, "message", {
            "role": "system",
            "content": "Голосовой вывод выключен."
        })
        return True, None

    # Управление STT (в веб-версии управляется через браузер, но оставляем команды для совместимости)
    if user_lower == "*listenon":
        state["listen_enabled"] = True
        await send_message_to_client(websocket, "avatar", {"state": "idle"})
        await send_message_to_client(websocket, "message", {
            "role": "system",
            "content": "Голосовой ввод доступен через кнопку микрофона в интерфейсе."
        })
        return True, None

    if user_lower == "*listenoff":
        state["listen_enabled"] = False
        await send_message_to_client(websocket, "avatar", {"state": "idle"})
        await send_message_to_client(websocket, "message", {
            "role": "system",
            "content": "Голосовой ввод отключен."
        })
        return True, None

    # Сохранение факта
    if user.startswith("*remember:"):
        fact = user[len("*remember:"):].strip()
        await send_message_to_client(websocket, "avatar", {"state": "idle"})
        if remember_fact(fact):
            await send_message_to_client(websocket, "message", {
                "role": "system",
                "content": "Сохранено в память."
            })
        else:
            await send_message_to_client(websocket, "message", {
                "role": "system",
                "content": "Нечего сохранять."
            })
        return True, None

    # Удаление из памяти
    if user.startswith("*forget:"):
        pattern = user[len("*forget:"):].strip()
        await send_message_to_client(websocket, "avatar", {"state": "idle"})
        removed = forget_fact(pattern)
        await send_message_to_client(websocket, "message", {
            "role": "system",
            "content": f"Удалено строк: {removed}"
        })
        return True, None

    # Обработка флагов *-m
    use_memory = True
    user_clean = user
    if "*-m" in user_clean:
        use_memory = False
        user_clean = user_clean.replace("*-m", "").strip()

    # Файлы: из вложений (веб) и/или *file пути (сервер)
    text_only, file_paths = parse_file_paths_from_input(user_clean)
    user_to_send = text_only if text_only else user_clean.strip()
    file_results = []
    for fp in file_paths:
        loaded = load_file_as_base64(fp)
        if loaded:
            file_results.append(loaded)
    for att in attachments or []:
        b64 = att.get("data")
        mime = att.get("type", "application/octet-stream")
        name = att.get("name", "file")
        if b64:
            file_results.append((b64, mime, name))
    if not user_to_send and not file_results:
        await send_message_to_client(websocket, "avatar", {"state": "idle"})
        await send_message_to_client(websocket, "message", {
            "role": "system",
            "content": "Укажите текст сообщения и/или прикрепите файлы."
        })
        return True, None
    user_content = build_content_parts(user_to_send or "(прикреплённые файлы)", file_results)

    await send_message_to_client(websocket, "avatar", {"state": "thinking"})

    contextual_messages = list(history)
    if use_memory:
        memory_snippets = search_memory(user_clean)
        if memory_snippets:
            memory_block = "Контекст памяти (используй при ответе, если релевантно):\n" + "\n".join(memory_snippets)
            contextual_messages.append({"role": "system", "content": memory_block})
    
    # *won / *woff — веб-поиск через MCP (инструменты в LM Studio)
    if web_enabled and WEB_SEARCH_VIA_MCP:
        contextual_messages.append({
            "role": "system",
            "content": "У тебя есть доступ к поиску в интернете через инструменты. Используй их для актуальной информации, новостей и данных."
        })

    if run_enabled:
        run_instructions = (
            "Тебе доступно выполнение команд в терминале. Добавь в ответ маркер `[RUN: команда]`, "
            "например `[RUN: ls -la]`. Система выполнит команду и вернёт вывод. Одна команда на маркер."
        )
        contextual_messages.append({"role": "system", "content": run_instructions})

    response = send_message(
        contextual_messages + [{"role": "user", "content": user_content}],
        use_tools=web_enabled,
    )
    
    # [TIME] обработка
    time_match = re.search(r'\[TIME\]', response, re.IGNORECASE)
    if time_match:
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        response_clean = re.sub(r'\[TIME\]', '', response, flags=re.IGNORECASE).strip()

        time_context = list(history)
        time_context.append({"role": "user", "content": user_content})
        time_context.append({"role": "assistant", "content": response_clean})
        time_context.append({"role": "system", "content": f"Текущее время и дата: {current_time_str}"})
        time_context.append({"role": "user", "content": "Используй точное текущее время из предоставленной информации, чтобы ответить естественно и точно. Не упоминай, что ты запрашивал или получил время — просто используй его."})

        response = send_message(time_context)
    
    # [RUN:] обработка — запрос подтверждения, выполнение по [y/n]
    run_matches = re.findall(r'\[RUN:\s*([^\]]+)\]', response, re.IGNORECASE)
    if run_matches and run_enabled:
        response_clean = re.sub(r'\[RUN:\s*[^\]]+\]', '', response, flags=re.IGNORECASE).strip()
        state["pending_run"] = {
            "response_clean": response_clean,
            "history": list(history),
            "user_content": user_content,
            "commands": [c.strip() for c in run_matches if c.strip()],
        }
        await send_message_to_client(websocket, "avatar", {"state": "idle"})
        await send_message_to_client(websocket, "run_confirm", {
            "commands": state["pending_run"]["commands"],
            "prompt": "Выполнить команды? Ответьте *y чтобы выполнить все, *n чтобы отменить."
        })
        return True, "__PENDING_RUN__"
    
    response = re.sub(r'\[SEARCH:\s*[^\]]+\]|\[TIME\]|\[RUN:\s*[^\]]+\]', '', response, flags=re.IGNORECASE).strip()
    
    mood_match = re.search(r'\[настроение:\s*([^\]]+)\]', response)
    if mood_match:
        mood_text = mood_match.group(1).strip()
        emotion_state = avatar.get_emotion_from_mood(mood_text)
        await send_message_to_client(websocket, "avatar", {"state": emotion_state})
    else:
        await send_message_to_client(websocket, "avatar", {"state": "idle"})

    await send_message_to_client(websocket, "message", {
        "role": "assistant",
        "content": response
    })
    save_dialogue(user_to_send or user_clean, response)

    if voice_enabled:
        await send_audio_to_client(websocket, response, voice_enabled)

    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": response})

    return True, response

@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    """Страница логина"""
    context = {"request": request, "error": None}
    if USE_TEMPLATES:
        return templates.TemplateResponse("login.html", context)
    else:
        template = jinja_env.get_template("login.html")
        return HTMLResponse(content=template.render(**context))


@app.post("/login", response_class=HTMLResponse)
async def post_login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Обработка формы логина"""
    if username == AUTH_USERNAME and password == AUTH_PASSWORD:
        response = RedirectResponse(url="/", status_code=302)
        # Простой куки для авторизации
        response.set_cookie(
            AUTH_COOKIE_NAME,
            AUTH_COOKIE_VALUE,
            httponly=True,
            max_age=60 * 60 * 12,  # 12 часов
        )
        return response

    # Неверный логин/пароль
    context = {"request": request, "error": "Неверный логин или пароль"}
    if USE_TEMPLATES:
        return templates.TemplateResponse("login.html", context)
    else:
        template = jinja_env.get_template("login.html")
        return HTMLResponse(content=template.render(**context))


@app.get("/logout")
async def logout(request: Request):
    """Выход (очистка куки)"""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(AUTH_COOKIE_NAME)
    return response


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Главная страница"""
    # Проверяем авторизацию по куки
    auth_token = request.cookies.get(AUTH_COOKIE_NAME)
    if auth_token != AUTH_COOKIE_VALUE:
        return RedirectResponse(url="/login", status_code=302)

    if USE_TEMPLATES:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        template = jinja_env.get_template("index.html")
        return HTMLResponse(content=template.render())

@app.get("/avatars/{filename}")
async def get_avatar(filename: str):
    """Отдача файлов аватаров"""
    avatar_path = f"avatars/{filename}"
    if os.path.exists(avatar_path):
        return FileResponse(avatar_path)
    return {"error": "Avatar not found"}, 404

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint для общения с CoreLine"""
    # Дополнительная проверка авторизации по куки
    auth_token = websocket.cookies.get(AUTH_COOKIE_NAME)
    if auth_token != AUTH_COOKIE_VALUE:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    
    # Инициализация состояния для этого подключения
    avatar = AvatarTerminal()
    system_prompt = load_system_prompt()
    state = {
        "avatar": avatar,
        "history": [{"role": "system", "content": system_prompt}],
        "web_enabled": False,
        "run_enabled": True,
        "voice_enabled": False,
        "listen_enabled": False
    }
    connections[websocket] = state
    
    # Отправляем начальное состояние
    await send_message_to_client(websocket, "avatar", {"state": "idle"})
    await send_message_to_client(websocket, "message", {
        "role": "system",
        "content": "CoreLine готов к общению. Введите *help для справки."
    })
    
    # В веб-версии голосовой ввод работает через браузер (Web Speech API)
    # Фоновая задача для терминального микрофона не нужна
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                user_input = data.get("content", "").strip()
                attachments = data.get("attachments", [])
                if not user_input and not attachments:
                    continue
                
                # Подтверждение выполнения команд [y/n]
                if state.get("pending_run"):
                    low = user_input.strip().lower()
                    if low in ("*y", "*n", "y", "n", "д", "да", "н", "нет"):
                        accepted = low in ("*y", "y", "д", "да")
                        pr = state.pop("pending_run")
                        avatar = state["avatar"]
                        if accepted:
                            run_outputs = []
                            for cmd in pr["commands"]:
                                await send_message_to_client(websocket, "avatar", {"state": "thinking"})
                                await send_message_to_client(websocket, "message", {
                                    "role": "system",
                                    "content": f"Выполняю: {cmd}"
                                })
                                out = run_shell_command(cmd)
                                run_outputs.append(f"$ {cmd}\n{out}")
                            run_block = "Вывод терминала:\n\n" + "\n---\n\n".join(run_outputs)
                            run_context = pr["history"] + [
                                {"role": "user", "content": pr["user_content"]},
                                {"role": "assistant", "content": pr["response_clean"]},
                                {"role": "system", "content": run_block},
                                {"role": "user", "content": "Используй вывод терминала, чтобы дополнить ответ. Не упоминай маркер [RUN], ответь естественно."}
                            ]
                            response = send_message(run_context)
                        else:
                            await send_message_to_client(websocket, "message", {
                                "role": "system",
                                "content": "Отменено."
                            })
                            run_context = pr["history"] + [
                                {"role": "user", "content": pr["user_content"]},
                                {"role": "assistant", "content": pr["response_clean"]},
                                {"role": "user", "content": "Пользователь отменил выполнение команд. Дай ответ без вывода терминала, не упоминая отмену."}
                            ]
                            response = send_message(run_context)
                        mood_match = re.search(r'\[настроение:\s*([^\]]+)\]', response)
                        if mood_match:
                            emotion_state = avatar.get_emotion_from_mood(mood_match.group(1).strip())
                            await send_message_to_client(websocket, "avatar", {"state": emotion_state})
                        else:
                            await send_message_to_client(websocket, "avatar", {"state": "idle"})
                        response = re.sub(r'\[настроение:\s*[^\]]+\]', '', response, flags=re.IGNORECASE).strip()
                        await send_message_to_client(websocket, "message", {"role": "assistant", "content": response})
                        uc = pr["user_content"]
                        user_str = uc if isinstance(uc, str) else ( " ".join(p.get("text", str(p)) for p in uc) if isinstance(uc, list) else str(uc) )
                        save_dialogue(user_str, response)
                        if state["voice_enabled"]:
                            await send_audio_to_client(websocket, response, state["voice_enabled"])
                        state["history"].append({"role": "user", "content": pr["user_content"]})
                        state["history"].append({"role": "assistant", "content": response})
                        continue
                
                # Отправляем сообщение пользователя (текст; вложения показываем как "[N файлов]")
                display_content = user_input
                if attachments:
                    display_content = (user_input + " " if user_input else "") + f"[📎 {len(attachments)} файл(ов)]"
                await send_message_to_client(websocket, "message", {
                    "role": "user",
                    "content": display_content
                })
                
                # Обрабатываем команды *won, *woff
                if user_input.lower() == "*runon":
                    state["run_enabled"] = True
                    await send_message_to_client(websocket, "avatar", {"state": "idle"})
                    await send_message_to_client(websocket, "message", {
                        "role": "system",
                        "content": "Режим терминала включён. ИИ может выполнять команды по маркеру [RUN: команда]."
                    })
                    continue
                
                if user_input.lower() == "*runoff":
                    state["run_enabled"] = False
                    await send_message_to_client(websocket, "avatar", {"state": "idle"})
                    await send_message_to_client(websocket, "message", {
                        "role": "system",
                        "content": "Режим терминала выключен."
                    })
                    continue
                
                if user_input.lower() == "*won":
                    state["web_enabled"] = True
                    await send_message_to_client(websocket, "avatar", {"state": "idle"})
                    await send_message_to_client(websocket, "message", {
                        "role": "system",
                        "content": "Веб-поиск включен. ИИ теперь может использовать интернет."
                    })
                    continue
                
                if user_input.lower() == "*woff":
                    state["web_enabled"] = False
                    await send_message_to_client(websocket, "avatar", {"state": "idle"})
                    await send_message_to_client(websocket, "message", {
                        "role": "system",
                        "content": "Веб-поиск выключен."
                    })
                    continue
                
                # Сразу показываем аватар "думает" до начала обработки
                await send_message_to_client(websocket, "avatar", {"state": "thinking"})
                
                # Обрабатываем ввод (с вложениями)
                should_continue, _ = await process_user_input(user_input, websocket, state, attachments)
                if not should_continue:
                    break
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Ошибка WebSocket: {e}")
    finally:
        if websocket in connections:
            del connections[websocket]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
