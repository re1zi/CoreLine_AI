import json
import sys
import threading

import coreline as coreline_mod
from avatar import HeadlessAvatar
from coreline import CliUI, process_user_input
from handler import (
    delete_chat_message,
    new_message_id,
    regenerate_assistant,
    rewrite_user_message,
    run_sync,
)


class BridgeTUI:
    def __init__(self, emit):
        self.emit = emit
        self._pending_confirm = None
        self._confirm_event = threading.Event()
        self.avatar_cell_width = 40
        self.avatar_cell_height = 20

    def set_modes(self, web_enabled, run_enabled):
        self.emit(
            {
                "type": "status",
                "web_enabled": web_enabled,
                "run_enabled": run_enabled,
            }
        )

    def set_avatar_state(self, state):
        art = None
        pixels = None
        avatar = getattr(self, "avatar", None)
        if avatar:
            art = avatar.render_ascii(state=state)
            # Half-block rendering uses 2 vertical pixels per terminal cell.
            pixels = avatar.render_pixels(
                state=state,
                width=int(self.avatar_cell_width),
                height=int(self.avatar_cell_height) * 2,
            )
        self.emit({"type": "avatar_state", "state": state, "art": art, "pixels": pixels})

    def set_avatar_config(self, cell_width, cell_height):
        try:
            cw = max(12, int(cell_width))
            ch = max(8, int(cell_height))
        except Exception:
            return
        self.avatar_cell_width = cw
        self.avatar_cell_height = ch

    def add_message(self, role, text, msg_id=None):
        payload = {"type": "message", "role": role, "text": text}
        if msg_id:
            payload["id"] = msg_id
        self.emit(payload)

    def start_stream(self):
        self.emit({"type": "stream_start"})

    def append_stream(self, chunk):
        self.emit({"type": "stream_chunk", "chunk": chunk})

    def end_stream(self):
        self.emit({"type": "stream_end"})

    def ask_confirm(self, cmd):
        self._pending_confirm = "n"
        self._confirm_event.clear()
        self.emit({"type": "confirm_request", "command": cmd})
        self._confirm_event.wait()
        return self._pending_confirm

    def resolve_confirm(self, answer):
        self._pending_confirm = (answer or "n").strip().lower()
        self._confirm_event.set()


def main():
    with open("prompts/system.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    history = [{"role": "system", "content": system_prompt}]
    avatar = HeadlessAvatar()
    bridge = None
    io_lock = threading.Lock()
    running = threading.Event()
    running.set()

    state = {
        "web_enabled": False,
        "run_enabled": False,
        "voice_enabled": False,
        "listen_enabled": False,
    }

    def emit(event):
        with io_lock:
            print(json.dumps(event, ensure_ascii=False), flush=True)

    bridge = BridgeTUI(emit=emit)
    bridge.avatar = avatar
    bridge.set_modes(state["web_enabled"], state["run_enabled"])
    bridge.set_avatar_state("idle")
    emit({"type": "ready"})

    # Background STT loop (mirrors coreline.py voice_input_loop but emits to TUI)
    def voice_input_loop():
        while running.is_set():
            try:
                if getattr(coreline_mod, "listen_enabled", False):
                    spoken = coreline_mod.listen()
                    if spoken and str(spoken).strip():
                        text = str(spoken).strip()
                        user_id = new_message_id()
                        state["pending_user_id"] = user_id
                        emit({"type": "message", "role": "user", "text": f"(voice) {text}", "id": user_id})
                        should_continue, _ = process_user_input(
                            text,
                            avatar,
                            history,
                            state["web_enabled"],
                            state["run_enabled"],
                            tui=bridge,
                            session=state,
                        )
                        emit({"type": "turn_done"})
                        if not should_continue:
                            running.clear()
                            break
            except Exception as exc:
                # Never crash the bridge on microphone/STT errors
                print(f"[bridge:stt] error: {exc}", file=sys.stderr, flush=True)
            # light polling
            threading.Event().wait(0.3)

    listener_thread = threading.Thread(target=voice_input_loop, daemon=True)
    listener_thread.start()

    for line in sys.stdin:
        raw = (line or "").strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            emit({"type": "error", "message": "invalid_json"})
            continue

        etype = event.get("type")
        if etype == "exit":
            running.clear()
            break

        if etype == "avatar_config":
            bridge.set_avatar_config(event.get("cell_width"), event.get("cell_height"))
            bridge.set_avatar_state(avatar.current_state)
            continue

        if etype == "confirm_response":
            bridge.resolve_confirm(event.get("answer", "n"))
            continue

        if etype in ("delete", "rewrite", "regenerate"):
            state["history"] = history
            state["avatar"] = avatar
            ui = CliUI(bridge, avatar)
            msg_id = str(event.get("id") or "")
            with coreline_mod._turn_lock:
                if etype == "delete":
                    truncated = run_sync(delete_chat_message(msg_id, state, True))
                    if truncated:
                        emit({"type": "truncate", "id": truncated})
                    else:
                        emit({"type": "message", "role": "system", "text": "Сообщение не найдено."})
                elif etype == "rewrite":
                    new_text = str(event.get("text") or event.get("content") or "").strip()
                    if not new_text:
                        emit({"type": "message", "role": "system", "text": "Нечего сохранять."})
                    else:
                        emit({"type": "truncate", "id": msg_id})
                        user_id = new_message_id()
                        state["pending_user_id"] = user_id
                        emit({"type": "message", "role": "user", "text": new_text, "id": user_id})
                        keep, _ = run_sync(
                            rewrite_user_message(msg_id, new_text, state, ui, None, True)
                        )
                        if not keep:
                            break
                else:
                    emit({"type": "truncate", "id": msg_id})
                    run_sync(regenerate_assistant(msg_id, state, ui, True))
            emit({"type": "turn_done"})
            continue

        if etype != "user_input":
            emit({"type": "error", "message": f"unsupported_event:{etype}"})
            continue

        user_input = str(event.get("text", "")).strip()
        if not user_input:
            continue

        looks_like_command = user_input.startswith("*") or user_input.startswith("поиск")
        payload = {"type": "message", "role": "user", "text": user_input}
        if not looks_like_command:
            user_id = new_message_id()
            state["pending_user_id"] = user_id
            payload["id"] = user_id
        emit(payload)

        should_continue, _ = process_user_input(
            user_input,
            avatar,
            history,
            state["web_enabled"],
            state["run_enabled"],
            tui=bridge,
            session=state,
        )
        emit({"type": "turn_done"})
        if not should_continue:
            break

    emit({"type": "shutdown"})


if __name__ == "__main__":
    main()
