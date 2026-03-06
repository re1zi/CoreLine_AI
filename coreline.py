import time
import random
import select
import sys
import re
import threading
from api import send_message
from avatar import AvatarTerminal
from file_utils import (
    load_file_as_base64,
    build_content_parts,
    parse_file_paths_from_input,
)
from memory import save_dialogue, search_memory, remember_fact, forget_fact
from run_shell import run_shell_command
from config import WEB_SEARCH_VIA_MCP
from voice import (
    voice_enabled,
    speak_stream,
    stop_speaking,
    init_xtts,
    listen
)

voice_enabled = False   # TTS — голосовой вывод
listen_enabled = False  # STT — голосовой ввод

def get_multiline_input(prompt=""):
    """Неблокирующий мультиринговый ввод: возвращает None, если данных нет."""
    import sys
    import select
    
    stdin = sys.stdin
    try:
        ready, _, _ = select.select([stdin], [], [], 0)
    except (ValueError, OSError):
        ready = []
    if not ready:
        return None
    
    lines = []
    while True:
        try:
            line = input()
        except (UnicodeDecodeError, UnicodeError):
            try:
                raw_input = sys.stdin.buffer.readline()
                if not raw_input:
                    break
                line = None
                for encoding in ['utf-8', 'latin-1', 'cp1251', 'windows-1251', 'iso-8859-1']:
                    try:
                        line = raw_input.decode(encoding, errors='replace').rstrip('\n\r')
                        break
                    except Exception:
                        continue
                if line is None:
                    line = raw_input.decode('utf-8', errors='replace').rstrip('\n\r')
            except Exception:
                continue
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines) if lines else None

def process_user_input(user, avatar, history, web_enabled, run_enabled=False):
    global voice_enabled, listen_enabled

    """Обработка ввода пользователя"""
    user_lower = user.lower().strip()

    if user_lower in ["exit", "quit", "*bye", "*пока"]:
        return False, None

    # ── Команда сна (единственная для сна) ─────────
    if user_lower in ["*sleep", "*спать"]:
        avatar.show("sleeping")
        print("\nCoreLine: *zzz...* (режим сна включён)\n")
        if voice_enabled:
            threading.Thread(target=speak_stream, args=("zzz...",), daemon=True).start()
        return True, None
    # ───────────────────────────────────────────────

    # Поиск в памяти
    if user.startswith("поиск"):
        parts = user.split(" ", 1)
        query = parts[1].strip() if len(parts) > 1 else ""
        avatar.show("idle")
        print("CoreLine (поиск):")
        if not query:
            print("  (укажите запрос, например: 'поиск погода')")
        else:
            results = search_memory(query)
            if results:
                for r in results:
                    print("  " + r)
            else:
                print("  (ничего не найдено)")
        return True, None

    # Справка по командам
    if user.startswith("*help"):
        avatar.show("idle")
        print("Доступные команды и флаги:")
        print("  *help                    — показать эту справку")
        print("  поиск <запрос>           — поиск по памяти (BM25)")
        print("  *remember: <факт>        — сохранить важный факт в память")
        print("  *forget: <подстрока>     — удалить строки памяти по подстроке")
        print("  *-m                      — отключить подмешивание памяти на один запрос")
        print("  *won                     — включить возможность использования интернета")
        print("  *woff                    — выключить возможность использования интернета")
        print("  *runon                   — включить выполнение команд в терминале ([RUN: команда])")
        print("  *runoff                  — выключить выполнение команд в терминале")
        print("  *file <путь>             — прикрепить файл (изображение и др.) к сообщению")
        print("  *voiceon                 — включить голосовой вывод (TTS)")
        print("  *voiceoff                — выключить голосовой вывод")
        print("  *listenon                — включить голосовой ввод (STT)")
        print("  *listenoff               — выключить голосовой ввод")
        print("  *sleep / *спать          — перейти в режим сна (визуально)")
        return True, None

    # Управление TTS
    if user_lower == "*voiceon":
        voice_enabled = True
        init_xtts()
        avatar.show("idle")
        print("Голосовой вывод XTTS включён.")
        return True, None
    
    if user_lower == "*voiceoff":
        voice_enabled = False
        stop_speaking()
        avatar.show("idle")
        print("Голосовой вывод выключен.")
        return True, None

    # Управление STT
    if user_lower == "*listenon":
        listen_enabled = True
        avatar.show("idle")
        print("Голосовой ввод включён. Говорите в микрофон — я буду слушать.")
        return True, None

    if user_lower == "*listenoff":
        listen_enabled = False
        avatar.show("idle")
        print("Голосовой ввод выключен.")
        return True, None

    # Явное сохранение факта
    if user.startswith("*remember:"):
        fact = user[len("*remember:"):].strip()
        avatar.show("idle")
        if remember_fact(fact):
            print("Сохранено в память.")
        else:
            print("Нечего сохранять.")
        return True, None

    # Удаление из памяти
    if user.startswith("*forget:"):
        pattern = user[len("*forget:"):].strip()
        avatar.show("idle")
        removed = forget_fact(pattern)
        print(f"Удалено строк: {removed}")
        return True, None

    # Обработка флагов *-m
    use_memory = True
    user_clean = user
    if "*-m" in user_clean:
        use_memory = False
        user_clean = user_clean.replace("*-m", "").strip()

    # Парсинг *file для прикрепления файлов
    text_only, file_paths = parse_file_paths_from_input(user_clean)
    user_to_send = text_only if text_only else user_clean.strip()
    file_results = []
    failed_paths = []
    for fp in file_paths:
        loaded = load_file_as_base64(fp)
        if loaded:
            file_results.append(loaded)
        else:
            failed_paths.append(fp)
    if failed_paths:
        avatar.show("idle")
        print(f"Не удалось загрузить файлы: {', '.join(failed_paths)}")
        return True, None

    # Контент для API: текст + вложения или просто текст
    if not user_to_send and not file_results:
        avatar.show("idle")
        print("Укажите текст сообщения и/или пути к файлам (*file путь)")
        return True, None
    user_content = build_content_parts(user_to_send or "(прикреплённые файлы)", file_results)

    avatar.show("thinking")

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
    
    # [RUN:] обработка — выполнение команд в терминале (с подтверждением [y/n])
    run_matches = re.findall(r'\[RUN:\s*([^\]]+)\]', response, re.IGNORECASE)
    if run_matches and run_enabled:
        run_outputs = []
        for cmd in run_matches:
            cmd = cmd.strip()
            if not cmd:
                continue
            while True:
                try:
                    ans = input(f"Выполнить: {cmd} [y/n]? ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ans = "n"
                if ans in ("y", "yes", "д", "да"):
                    avatar.show("thinking")
                    print(f"\nCoreLine: Выполняю: {cmd}\n")
                    out = run_shell_command(cmd)
                    run_outputs.append(f"$ {cmd}\n{out}")
                    break
                elif ans in ("n", "no", "н", "нет"):
                    run_outputs.append(f"$ {cmd}\n(отменено пользователем)")
                    break
                print("Введите y или n.")
        if run_outputs:
            run_block = "Вывод терминала:\n\n" + "\n---\n\n".join(run_outputs)
            response_clean = re.sub(r'\[RUN:\s*[^\]]+\]', '', response, flags=re.IGNORECASE).strip()
            run_context = list(history)
            run_context.append({"role": "user", "content": user_content})
            run_context.append({"role": "assistant", "content": response_clean})
            run_context.append({"role": "system", "content": run_block})
            run_context.append({"role": "user", "content": "Используй вывод терминала, чтобы дополнить ответ. Не упоминай маркер [RUN], ответь естественно."})
            response = send_message(run_context)
    
    response = re.sub(r'\[SEARCH:\s*[^\]]+\]|\[TIME\]|\[RUN:\s*[^\]]+\]', '', response, flags=re.IGNORECASE).strip()
    
    mood_match = re.search(r'\[настроение:\s*([^\]]+)\]', response)
    if mood_match:
        mood_text = mood_match.group(1).strip()
        emotion_state = avatar.get_emotion_from_mood(mood_text)
    else:
        emotion_state = "idle"

    avatar.show_with_text(emotion_state, f"CoreLine: {response}")
    user_text = user_clean if isinstance(user_clean, str) else (user_content if isinstance(user_content, str) else str(user_content))
    save_dialogue(user_text, response)

    if voice_enabled:
        threading.Thread(target=speak_stream, args=(response,), daemon=True).start()

    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": response})

    return True, response

def main():
    global voice_enabled, listen_enabled

    avatar = AvatarTerminal()
    avatar.show("idle")

    with open("prompts/system.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    history = [{"role": "system", "content": system_prompt}]

    web_enabled = False
    run_enabled = True

    print("\nТы: ", end="", flush=True)

    # Фоновая функция для постоянного прослушивания микрофона
    def voice_input_loop():
        while True:
            if listen_enabled:
                spoken = listen()
                if spoken and spoken.strip():
                    avatar.show("thinking")
                    should_continue, _ = process_user_input(spoken, avatar, history, web_enabled, run_enabled)
                    print(f"\nГолосовой ввод: {spoken}")
                    if not should_continue:
                        sys.exit(0)
                    print("\nТы: ", end="", flush=True)
            time.sleep(0.3)

    listener_thread = threading.Thread(target=voice_input_loop, daemon=True)
    listener_thread.start()

    while True:
        user_input = get_multiline_input()
        
        if user_input is not None:
            # Обрабатываем команды *won, *woff
            if user_input.lower() == "*won":
                web_enabled = True
                avatar.show("idle")
                print("Веб-поиск включен. ИИ теперь может использовать интернет.")
                print("\nТы: ", end="", flush=True)
                continue
            
            if user_input.lower() == "*woff":
                web_enabled = False
                avatar.show("idle")
                print("Веб-поиск выключен.")
                print("\nТы: ", end="", flush=True)
                continue
            
            if user_input.lower() == "*runon":
                run_enabled = True
                avatar.show("idle")
                print("Режим терминала включён. ИИ может выполнять команды по маркеру [RUN: команда].")
                print("\nТы: ", end="", flush=True)
                continue
            
            if user_input.lower() == "*runoff":
                run_enabled = False
                avatar.show("idle")
                print("Режим терминала выключен.")
                print("\nТы: ", end="", flush=True)
                continue
                
            should_continue, response = process_user_input(user_input, avatar, history, web_enabled, run_enabled)
            if not should_continue:
                break
            
            print("\nТы: ", end="", flush=True)
            continue
        
        time.sleep(0.1)  # просто ждём ввода

if __name__ == "__main__":
    main()