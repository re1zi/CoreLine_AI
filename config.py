# config.py — настройки проекта CoreLine
# Переменные окружения для веб-поиска и MCP

import os

# Загружаем .env из корня проекта (LM_STUDIO_API_KEY и др.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -----------------------------------------------------------------------------
# LM Studio API (чат и эмбеддинги)
# -----------------------------------------------------------------------------
# Если в LM Studio включено Require Authentication, задай токен:
#   export LM_STUDIO_API_KEY="твой-токен-из-lm-studio"
# или
#   export OPENAI_API_KEY="твой-токен-из-lm-studio"
# Токен задаётся/смотрится в LM Studio в настройках сервера (Authentication).
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Веб-поиск (SearXNG)
# -----------------------------------------------------------------------------
# *won / *woff включают поиск. При SEARXNG_MANAGE=1 (по умолчанию) *won
# поднимает локальный контейнер SearXNG, *woff его останавливает.
#
# WEB_SEARCH_VIA_MCP=1 — старый путь: integrations mcp/web-search в LM Studio.
# -----------------------------------------------------------------------------

WEB_SEARCH_VIA_MCP = os.getenv("WEB_SEARCH_VIA_MCP", "0").lower() in ("1", "true", "yes")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888").strip()
SEARXNG_LANGUAGE = os.getenv("SEARXNG_LANGUAGE", "ru-RU").strip() or "auto"
try:
    SEARXNG_MAX_RESULTS = max(1, min(20, int(os.getenv("SEARXNG_MAX_RESULTS", "8"))))
except ValueError:
    SEARXNG_MAX_RESULTS = 8
try:
    SEARXNG_TIMEOUT = max(3, float(os.getenv("SEARXNG_TIMEOUT", "20")))
except ValueError:
    SEARXNG_TIMEOUT = 20.0
try:
    WEB_FETCH_MAX_PAGES = max(0, min(5, int(os.getenv("WEB_FETCH_MAX_PAGES", "3"))))
except ValueError:
    WEB_FETCH_MAX_PAGES = 3
try:
    WEB_FETCH_MAX_CHARS = max(500, int(os.getenv("WEB_FETCH_MAX_CHARS", "6000")))
except ValueError:
    WEB_FETCH_MAX_CHARS = 6000
try:
    WEB_FETCH_TIMEOUT = max(3, float(os.getenv("WEB_FETCH_TIMEOUT", "15")))
except ValueError:
    WEB_FETCH_TIMEOUT = 15.0
try:
    WEB_FETCH_MAX_BYTES = max(50_000, int(os.getenv("WEB_FETCH_MAX_BYTES", "1500000")))
except ValueError:
    WEB_FETCH_MAX_BYTES = 1_500_000

# *won / *woff поднимают и гасят docker compose в searxng/
SEARXNG_MANAGE = os.getenv("SEARXNG_MANAGE", "1").lower() in ("1", "true", "yes")
try:
    SEARXNG_START_TIMEOUT = max(5, float(os.getenv("SEARXNG_START_TIMEOUT", "60")))
except ValueError:
    SEARXNG_START_TIMEOUT = 60.0

# Keep the leading system prompt plus this many later messages (user/assistant turns).
try:
    MAX_HISTORY_MESSAGES = max(4, int(os.getenv("CORELINE_MAX_HISTORY", "40")))
except ValueError:
    MAX_HISTORY_MESSAGES = 40

# Web UI bind address. Default is localhost-only.
WEB_HOST = os.getenv("CORELINE_WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
try:
    WEB_PORT = int(os.getenv("CORELINE_WEB_PORT", "5000"))
except ValueError:
    WEB_PORT = 5000

# Текст системной подсказки при включённом *won (можно переопределить через env)
WEB_SEARCH_SYSTEM_PROMPT = os.getenv(
    "WEB_SEARCH_SYSTEM_PROMPT",
    "Тебе даны результаты SearXNG и текст открытых страниц. Опирайся на них. "
    "Если нужно уточнить поиск — `[SEARCH: запрос]`. "
    "Если нужно прочитать конкретную ссылку — `[FETCH: https://…]`."
).strip()
