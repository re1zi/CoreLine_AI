import os
import re
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Connect timeout, then read timeout (seconds). Streaming uses a longer read window.
_CONNECT_TIMEOUT = float(os.environ.get("LM_STUDIO_CONNECT_TIMEOUT", "10"))
_READ_TIMEOUT = float(os.environ.get("LM_STUDIO_TIMEOUT", "120"))
_STREAM_READ_TIMEOUT = float(os.environ.get("LM_STUDIO_STREAM_TIMEOUT", "300"))
HTTP_TIMEOUT = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
HTTP_STREAM_TIMEOUT = (_CONNECT_TIMEOUT, _STREAM_READ_TIMEOUT)

# OpenAI-совместимый endpoint (без tools)
API_URL = os.environ.get("LM_STUDIO_API_URL", "http://localhost:1234/v1/chat/completions")
# LM Studio Chat API с поддержкой tools/integrations (MCP)
CHAT_API_URL = os.environ.get("LM_STUDIO_CHAT_URL", "http://localhost:1234/api/v1/chat")

MODEL = os.environ.get("LM_STUDIO_MODEL", "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive")

# Токен для LM Studio (Require Authentication). Задай LM_STUDIO_API_KEY или OPENAI_API_KEY.
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY") or os.environ.get("OPENAI_API_KEY", "").strip()

# ID MCP-плагина в LM Studio: ключ из mcp.json → mcpServers. В API передаётся как mcp/<ключ>.
MCP_PLUGIN_ID = os.environ.get("MCP_PLUGIN_ID", "web-search").strip() or "web-search"
MCP_PLUGIN_ID_API = f"mcp/{MCP_PLUGIN_ID}" if not MCP_PLUGIN_ID.startswith("mcp/") else MCP_PLUGIN_ID

# Включить потоковую передачу текста (только для OpenAI-совместимого endpoint без tools).
TEXT_STREAMING = os.environ.get("TEXT_STREAMING", "").strip().lower() in ("1", "true", "yes")


def extract_final_response(content: str) -> str:
    """
    Извлекает финальный ответ из вывода thinking-модели.
    Рассуждения (<think>...</think>) остаются за кадром — в чат выводится только финальный ответ.
    """
    if not content:
        return ""

    # Удаляем блоки <think>...</think>` (Qwen3, DeepSeek R1 и подобные)
    result = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    # Удаляем оставшийся открытый <think> без закрывающего тега
    result = re.sub(r"<think>.*", "", result, flags=re.DOTALL | re.IGNORECASE)
    return result.strip()


class ThinkFilter:
    """Strips <think>...</think> from a streaming token stream, including split tags."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        out = []
        while self._buf:
            hay = self._buf.lower()
            if self._in_think:
                idx = hay.find(self._CLOSE)
                if idx == -1:
                    keep = len(self._CLOSE) - 1
                    if final:
                        self._buf = ""
                    elif len(self._buf) > keep:
                        self._buf = self._buf[-keep:]
                    break
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._in_think = False
                continue
            idx = hay.find(self._OPEN)
            if idx == -1:
                keep = _partial_prefix_len(hay, self._OPEN)
                if final or keep == 0:
                    out.append(self._buf)
                    self._buf = ""
                else:
                    out.append(self._buf[:-keep])
                    self._buf = self._buf[-keep:]
                break
            out.append(self._buf[:idx])
            self._buf = self._buf[idx + len(self._OPEN):]
            self._in_think = True
        return "".join(out)


def _partial_prefix_len(hay: str, token: str) -> int:
    max_keep = min(len(token) - 1, len(hay))
    for n in range(max_keep, 0, -1):
        if token.startswith(hay[-n:]):
            return n
    return 0


def _messages_to_lm_studio_input(messages):
    """Конвертация messages (OpenAI-формат) в input + system_prompt для /api/v1/chat."""
    system_parts = []
    turns = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                (p.get("text") or p.get("content") or "") for p in content if isinstance(p, dict)
            )
        if not (content and str(content).strip()):
            continue
        if role == "system":
            system_parts.append(str(content).strip())
        else:
            label = "User" if role == "user" else "Assistant"
            turns.append(f"{label}: {str(content).strip()}")
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    input_value = "\n\n".join(turns) if turns else ""
    return input_value, system_prompt


def _parse_sse_content(line: str):
    """Извлекает content из одной SSE-строки data: {...} (OpenAI stream)."""
    if not line.startswith("data: "):
        return None
    payload = line[6:].strip()
    if payload == "[DONE]":
        return None
    try:
        import json
        data = json.loads(payload)
        choices = data.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        return delta.get("content") or ""
    except Exception:
        return None


def send_message_stream(messages, use_tools=False):
    """
    Генератор: потоковая отдача ответа по чанкам (только для OpenAI-совместимого API без tools).
    При use_tools=True отдаёт один чанк с полным ответом (streaming для Chat API не реализован).
    Yields: (chunk: str, done: bool). done=True только на последнем чанке с полным текстом.
    """
    if use_tools:
        full = send_message(messages, use_tools=True)
        yield full, True
        return

    headers = {"Content-Type": "application/json"}
    if LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"

    url = API_URL
    body = {"model": MODEL, "messages": messages, "stream": True}

    response = requests.post(url, json=body, headers=headers, stream=True, timeout=HTTP_STREAM_TIMEOUT)
    if response.status_code >= 400:
        raise requests.exceptions.HTTPError(
            f"{response.status_code} {response.reason}: {response.text[:500]}", response=response
        )
    response.encoding = "utf-8"

    filt = ThinkFilter()
    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        part = _parse_sse_content(line)
        if part is not None:
            if part:
                visible = filt.feed(part)
                if visible:
                    yield visible, False
    tail = filt.flush()
    if tail:
        yield tail, False
    yield "", True


def send_message(messages, use_tools=False):
    """
    Отправка запроса в LM Studio.
    use_tools=True: использует /api/v1/chat с integrations (MCP-плагин).
    иначе — OpenAI-совместимый /v1/chat/completions.
    """
    headers = {"Content-Type": "application/json"}
    if LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"

    if use_tools:
        url = CHAT_API_URL
        input_value, system_prompt = _messages_to_lm_studio_input(messages)
        body = {
            "model": MODEL,
            "input": input_value,
            "temperature": 0.7,
            "max_output_tokens": 4096,
            "integrations": [
                {"type": "plugin", "id": MCP_PLUGIN_ID_API}
            ],
        }
        if system_prompt:
            body["system_prompt"] = system_prompt
        if not body["input"]:
            body["input"] = "(пусто)"
    else:
        url = API_URL
        body = {
            "model": MODEL,
            "messages": messages,
        }

    response = requests.post(url, json=body, headers=headers, timeout=HTTP_TIMEOUT)
    if response.status_code >= 400:
        raise requests.exceptions.HTTPError(
            f"{response.status_code} {response.reason}: {response.text[:500]}", response=response
        )
    data = response.json()

    if use_tools:
        # Ответ /api/v1/chat: data["output"] — массив { type: "message"|"tool_call"|"reasoning", content?: str }
        output = data.get("output") or []
        parts = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message" and item.get("content"):
                parts.append(item["content"])
        content = "\n\n".join(parts) if parts else ""
        return extract_final_response(content)

    # OpenAI-совместимый ответ
    if "message" in data:
        msg = data["message"]
    elif "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", data["choices"][0])
    else:
        msg = data
    content = msg.get("content") or ""
    if msg.get("reasoning_content") or msg.get("reasoning"):
        return content.strip()
    return extract_final_response(content)
