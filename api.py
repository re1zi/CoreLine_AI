import os
import re
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OpenAI-совместимый endpoint (без tools)
API_URL = os.environ.get("LM_STUDIO_API_URL", "http://localhost:1234/v1/chat/completions")
# LM Studio Chat API с поддержкой tools/integrations (MCP)
CHAT_API_URL = os.environ.get("LM_STUDIO_CHAT_URL", "http://localhost:1234/api/v1/chat")

MODEL = os.environ.get("LM_STUDIO_MODEL", "qwen_qwen3.5-35b-a3b")

# Токен для LM Studio (Require Authentication). Задай LM_STUDIO_API_KEY или OPENAI_API_KEY.
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY") or os.environ.get("OPENAI_API_KEY", "").strip()

# ID MCP-плагина в LM Studio: ключ из mcp.json → mcpServers. В API передаётся как mcp/<ключ>.
MCP_PLUGIN_ID = os.environ.get("MCP_PLUGIN_ID", "web-search").strip() or "web-search"
MCP_PLUGIN_ID_API = f"mcp/{MCP_PLUGIN_ID}" if not MCP_PLUGIN_ID.startswith("mcp/") else MCP_PLUGIN_ID


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

    response = requests.post(url, json=body, headers=headers)
    response.raise_for_status()
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
