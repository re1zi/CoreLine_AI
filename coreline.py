import time
import threading
from avatar import AvatarTerminal
from handler import NO_ANSWERS, SessionUI, YES_ANSWERS, process_turn, run_sync

try:
    from voice import (
        voice_enabled as _voice_enabled_default,
        speak_stream,
        stop_speaking,
        init_tts,
        init_stt,
        listen,
        use_voice_clone,
    )
    VOICE_IMPORT_ERROR = None
except Exception as voice_exc:  # noqa: BLE001 - optional voice deps in TUI mode
    _voice_enabled_default = False
    use_voice_clone = False
    VOICE_IMPORT_ERROR = str(voice_exc)

    def speak_stream(*args, **kwargs):
        return None

    def stop_speaking():
        return None

    def init_tts():
        return None

    def init_stt():
        return None

    def listen():
        return ""

voice_enabled = bool(_voice_enabled_default) and False   # TTS — голосовой вывод
listen_enabled = False  # STT — голосовой ввод

_turn_lock = threading.Lock()
_shutdown = threading.Event()


def _ui_message(tui, role, text, msg_id=None):
    if tui:
        tui.add_message(role, text, msg_id)
    elif text:
        prefix = "CoreLine" if role == "coreline" else ("Система" if role == "system" else "Ты")
        print(f"{prefix}: {text}")


def _ui_state(tui, avatar, state):
    avatar.show(state)
    if tui:
        tui.set_avatar_state(state)


def _ui_stream_start(tui):
    if tui:
        tui.start_stream()
    else:
        print("CoreLine: ", end="", flush=True)


def _ui_stream_chunk(tui, chunk):
    if tui:
        tui.append_stream(chunk)
    else:
        print(chunk, end="", flush=True)


def _ui_stream_end(tui):
    if tui:
        tui.end_stream()
    else:
        print()


class CliUI(SessionUI):
    def __init__(self, tui, avatar):
        self.tui = tui
        self._avatar = avatar

    async def message(self, role, content, msg_id=None):
        if not self.tui and role == "coreline":
            self._avatar.show_with_text(self._avatar.current_state, f"CoreLine: {content}")
            return
        _ui_message(self.tui, role, content, msg_id)

    async def avatar(self, state):
        _ui_state(self.tui, self._avatar, state)

    async def stream_start(self):
        _ui_stream_start(self.tui)

    async def stream_chunk(self, chunk):
        _ui_stream_chunk(self.tui, chunk)

    async def stream_end(self):
        _ui_stream_end(self.tui)

    async def confirm_commands(self, commands):
        import asyncio
        approved = []
        for cmd in commands:
            while True:
                if self.tui:
                    ans = await asyncio.to_thread(self.tui.ask_confirm, cmd)
                else:
                    try:
                        ans = await asyncio.to_thread(
                            lambda c=cmd: input(f"Выполнить: {c} [y/n]? ").strip().lower()
                        )
                    except (EOFError, KeyboardInterrupt):
                        ans = "n"
                ans = (ans or "").strip().lower()
                if ans in YES_ANSWERS:
                    approved.append(cmd)
                    break
                if ans in NO_ANSWERS:
                    break
                await self.message("system", "Введите y или n.")
        return approved

    async def speak(self, text):
        threading.Thread(target=speak_stream, args=(text,), daemon=True).start()

    async def modes_changed(self, web_enabled, run_enabled):
        if self.tui and hasattr(self.tui, "set_modes"):
            self.tui.set_modes(web_enabled, run_enabled)

    def voice_import_error(self):
        return VOICE_IMPORT_ERROR

    def init_tts(self):
        init_tts()

    def init_stt(self):
        return bool(init_stt())

    def stop_speaking(self):
        stop_speaking()


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


def process_user_input(user, avatar, history, web_enabled, run_enabled=False, tui=None, session=None):
    """Sync entry used by the CLI and the TUI bridge. `session` is mutated in place."""
    global voice_enabled, listen_enabled

    state = session if session is not None else {}
    state["avatar"] = avatar
    state["history"] = history
    state["web_enabled"] = web_enabled if session is None else state.get("web_enabled", web_enabled)
    state["run_enabled"] = run_enabled if session is None else state.get("run_enabled", run_enabled)
    state["voice_enabled"] = state.get("voice_enabled", voice_enabled)
    state["listen_enabled"] = state.get("listen_enabled", listen_enabled)

    ui = CliUI(tui, avatar)
    with _turn_lock:
        keep, response = run_sync(process_turn(user, state, ui))

    voice_enabled = bool(state.get("voice_enabled"))
    listen_enabled = bool(state.get("listen_enabled"))
    if session is not None:
        session["web_enabled"] = state.get("web_enabled", False)
        session["run_enabled"] = state.get("run_enabled", False)
        session["voice_enabled"] = voice_enabled
        session["listen_enabled"] = listen_enabled
    return keep, response


def main():
    global voice_enabled, listen_enabled

    avatar = AvatarTerminal()
    avatar.show("idle")

    with open("prompts/system.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    history = [{"role": "system", "content": system_prompt}]
    session = {
        "web_enabled": False,
        "run_enabled": False,
        "voice_enabled": voice_enabled,
        "listen_enabled": listen_enabled,
    }

    print("\nТы: ", end="", flush=True)

    def voice_input_loop():
        global listen_enabled
        while not _shutdown.is_set():
            if listen_enabled:
                spoken = listen()
                if spoken and spoken.strip():
                    avatar.show("thinking")
                    should_continue, _ = process_user_input(
                        spoken, avatar, history, session["web_enabled"], session["run_enabled"],
                        session=session,
                    )
                    print(f"\nГолосовой ввод: {spoken}")
                    if not should_continue:
                        _shutdown.set()
                        return
                    print("\nТы: ", end="", flush=True)
            time.sleep(0.3)

    listener_thread = threading.Thread(target=voice_input_loop, daemon=True)
    listener_thread.start()

    while not _shutdown.is_set():
        user_input = get_multiline_input()

        if user_input is not None:
            should_continue, _ = process_user_input(
                user_input, avatar, history, session["web_enabled"], session["run_enabled"],
                session=session,
            )
            if not should_continue:
                break
            print("\nТы: ", end="", flush=True)
            continue

        time.sleep(0.1)


if __name__ == "__main__":
    main()
