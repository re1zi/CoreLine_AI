"""SearXNG search plus page fetch for *won and explicit links."""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

from config import (
    SEARXNG_LANGUAGE,
    SEARXNG_MAX_RESULTS,
    SEARXNG_TIMEOUT,
    SEARXNG_URL,
    WEB_FETCH_MAX_BYTES,
    WEB_FETCH_MAX_CHARS,
    WEB_FETCH_MAX_PAGES,
    WEB_FETCH_TIMEOUT,
)

_SEARCH_MARKER_RE = re.compile(r"\[SEARCH:\s*([^\]]+)\]", re.IGNORECASE)
_FETCH_MARKER_RE = re.compile(
    r"\[(?:FETCH|OPEN|READ|PAGE|URL):\s*([^\]]+)\]",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s<>\"'`)\]]+", re.IGNORECASE)
_FETCH_INTENT_RE = re.compile(
    r"(посмотри|посмотреть|посмотришь|открой|открыть|прочитай|прочитать|прочти|"
    r"смотри\s+ссылк|глянь|загляни|"
    r"look\s+at|open\s+(the\s+)?(link|url)|read\s+(the\s+)?(link|page|url)|fetch)",
    re.IGNORECASE,
)
_FILLER_RE = re.compile(
    r"\b(пожалуйста|плиз|please|ссылку|ссылка|ссылке|url|link|page|страницу|странице|"
    r"вот|эту|этот|этой|the|this|that)\b",
    re.IGNORECASE,
)
_SKIP_TAGS = {
    "script", "style", "noscript", "svg", "iframe", "canvas", "template",
    "nav", "footer", "header", "form", "button", "aside",
}
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 CoreLine/1.0"
)


def extract_search_queries(text: str) -> list[str]:
    queries = []
    seen = set()
    for raw in _SEARCH_MARKER_RE.findall(text or ""):
        q = " ".join(raw.split()).strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            queries.append(q)
    return queries


def extract_fetch_urls(text: str) -> list[str]:
    found = []
    seen = set()
    for raw in _FETCH_MARKER_RE.findall(text or ""):
        for url in extract_urls(raw):
            key = url.lower()
            if key not in seen:
                seen.add(key)
                found.append(url)
    return found


def extract_urls(text: str) -> list[str]:
    found = []
    seen = set()
    for raw in _URL_RE.findall(text or ""):
        url = raw.rstrip(".,;:!?)]}'\"")
        if not _allowed_url(url):
            continue
        key = url.lower()
        if key not in seen:
            seen.add(key)
            found.append(url)
    return found


def has_fetch_intent(text: str) -> bool:
    return bool(_FETCH_INTENT_RE.search(text or ""))


def leftover_search_query(text: str) -> str:
    """User text minus URLs and 'look at this link' filler."""
    leftover = _URL_RE.sub(" ", text or "")
    leftover = _FETCH_INTENT_RE.sub(" ", leftover)
    leftover = _FILLER_RE.sub(" ", leftover)
    leftover = re.sub(r"[«»\"'`]+", " ", leftover)
    return " ".join(leftover.split()).strip()


def _allowed_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip().lower()
    return bool(host)


def _clean(text: str, limit: int = 400) -> str:
    if not text:
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(text)))
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _search_url() -> str:
    return SEARXNG_URL.rstrip("/") + "/search"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._in_title = False
        self._prefer = 0
        self.title = ""
        self._body: list[str] = []
        self._main: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in ("article", "main"):
            self._prefer += 1
        if tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div", "section"):
            self._emit("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in ("article", "main") and self._prefer:
            self._prefer -= 1
        if tag in ("p", "li", "h1", "h2", "h3", "h4", "tr"):
            self._emit("\n")

    def handle_data(self, data):
        if self._skip:
            return
        chunk = " ".join((data or "").split())
        if not chunk:
            return
        if self._in_title:
            self.title += chunk + " "
            return
        self._emit(chunk + " ")

    def _emit(self, piece: str):
        dest = self._main if self._prefer else self._body
        dest.append(piece)

    def text(self) -> str:
        raw = "".join(self._main if len("".join(self._main)) > 200 else self._body)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def fetch_page(url: str) -> str:
    """Download one http(s) page and return readable text for the model."""
    url = (url or "").strip()
    if not _allowed_url(url):
        return f"Не могу открыть ссылку (нужен http/https): {url}"
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
                "Accept-Language": "ru,en;q=0.8",
            },
            timeout=WEB_FETCH_TIMEOUT,
            stream=True,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        return f"Не удалось открыть {url}: {exc}"

    final_url = response.url or url
    if response.status_code >= 400:
        response.close()
        return f"Страница {final_url} вернула {response.status_code}."

    content_type = (response.headers.get("Content-Type") or "").lower()
    chunks = []
    total = 0
    try:
        for piece in response.iter_content(chunk_size=65536):
            if not piece:
                continue
            chunks.append(piece)
            total += len(piece)
            if total >= WEB_FETCH_MAX_BYTES:
                break
    finally:
        response.close()
    raw = b"".join(chunks)
    encoding = response.encoding or "utf-8"
    try:
        body = raw.decode(encoding, errors="replace")
    except LookupError:
        body = raw.decode("utf-8", errors="replace")

    if "application/json" in content_type or body.lstrip()[:1] in "{[":
        if "html" not in content_type:
            return f"Содержимое {final_url} (JSON):\n{_truncate(body, WEB_FETCH_MAX_CHARS)}"

    if "text/plain" in content_type:
        return f"Текст {final_url}:\n{_truncate(body, WEB_FETCH_MAX_CHARS)}"

    if any(kind in content_type for kind in ("pdf", "image/", "audio/", "video/", "octet-stream")):
        return f"Ссылка {final_url} — не HTML ({content_type or 'неизвестный тип'}), текст страницы недоступен."

    parser = _HTMLTextExtractor()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        plain = _clean(body, WEB_FETCH_MAX_CHARS)
        return f"Страница {final_url}:\n{plain}"

    title = " ".join(parser.title.split())
    text = parser.text()
    if not text:
        text = _clean(body, WEB_FETCH_MAX_CHARS)
    heading = f"Страница «{title}»" if title else "Страница"
    return f"{heading}\n{final_url}\n\n{_truncate(text, WEB_FETCH_MAX_CHARS)}"


def fetch_pages(urls: list[str], limit: int | None = None) -> str:
    picked = []
    seen = set()
    cap = WEB_FETCH_MAX_PAGES if limit is None else limit
    for url in urls:
        if len(picked) >= cap:
            break
        if not _allowed_url(url):
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(url)
    if not picked:
        return "Нет ссылок для чтения."
    blocks = [fetch_page(url) for url in picked]
    return "Содержимое страниц:\n\n" + "\n\n---\n\n".join(blocks)


def search_web(query: str, fetch_pages_count: int | None = None) -> str:
    """Query SearXNG, then optionally read the top result pages."""
    query = " ".join((query or "").split()).strip()
    if not query:
        return "Веб-поиск: пустой запрос."
    listing, urls = _searxng_listing(query)
    count = WEB_FETCH_MAX_PAGES if fetch_pages_count is None else fetch_pages_count
    if count <= 0 or not urls:
        return listing
    pages = fetch_pages(urls, limit=count)
    return listing + "\n\n" + pages


def _searxng_listing(query: str) -> tuple[str, list[str]]:
    if not SEARXNG_URL:
        return "Веб-поиск: SEARXNG_URL не задан.", []

    params = {
        "q": query,
        "format": "json",
        "language": SEARXNG_LANGUAGE,
        "safesearch": 0,
        "pageno": 1,
    }
    headers = {"Accept": "application/json", "User-Agent": _UA}
    try:
        response = requests.get(
            _search_url(),
            params=params,
            headers=headers,
            timeout=SEARXNG_TIMEOUT,
        )
    except requests.RequestException as exc:
        return (
            f"Веб-поиск недоступен (SearXNG {SEARXNG_URL}): {exc}. "
            "Запусти контейнер: docker compose -f searxng/docker-compose.yml up -d",
            [],
        )

    if response.status_code == 403:
        return (
            "SearXNG отклонил format=json (обычно 403). "
            "В searxng/settings.yml в search.formats должен быть json, затем перезапусти контейнер.",
            [],
        )
    if response.status_code >= 400:
        return f"SearXNG вернул {response.status_code}: {response.text[:300]}", []

    try:
        data = response.json()
    except ValueError:
        return (
            "SearXNG отдал не JSON. Проверь, что инстанс локальный и json включён в search.formats.",
            [],
        )

    listing, urls = format_search_results(query, data)
    return listing, urls


def format_search_results(query: str, data: dict) -> tuple[str, list[str]]:
    lines = [f'Результаты SearXNG по запросу "{query}":']
    added = 0
    urls: list[str] = []

    for answer in data.get("answers") or []:
        text = answer if isinstance(answer, str) else (answer.get("answer") or answer.get("text") or "")
        text = _clean(text, 500)
        if text:
            lines.append(f"- Ответ: {text}")
            added += 1

    for box in data.get("infoboxes") or []:
        if not isinstance(box, dict):
            continue
        title = _clean(box.get("infobox") or box.get("id") or "", 120)
        content = _clean(box.get("content") or "", 500)
        url = (box.get("url") or "").strip()
        bit = " ".join(p for p in (title, content) if p)
        if bit:
            extra = f" ({url})" if url else ""
            lines.append(f"- Справка: {bit}{extra}")
            added += 1
        if _allowed_url(url):
            urls.append(url)

    result_n = 0
    for item in data.get("results") or []:
        if result_n >= SEARXNG_MAX_RESULTS:
            break
        if not isinstance(item, dict):
            continue
        title = _clean(item.get("title") or "без названия", 180)
        url = (item.get("url") or "").strip()
        snippet = _clean(item.get("content") or item.get("snippet") or "", 360)
        engine = _clean(item.get("engine") or "", 40)
        result_n += 1
        added += 1
        lines.append(f"{result_n}. {title}")
        if url:
            lines.append(f"   {url}")
            if _allowed_url(url):
                urls.append(url)
        if snippet:
            lines.append(f"   {snippet}")
        if engine:
            lines.append(f"   источник: {engine}")

    if added == 0:
        unresp = data.get("unresponsive_engines") or []
        extra = ""
        if unresp:
            names = []
            for eng in unresp:
                if isinstance(eng, (list, tuple)) and eng:
                    names.append(str(eng[0]))
                elif isinstance(eng, str):
                    names.append(eng)
            if names:
                extra = " Не ответили: " + ", ".join(names[:8]) + "."
        return f'SearXNG: по запросу "{query}" ничего не найдено.{extra}', []

    lines.append(
        "Ниже — текст нескольких верхних страниц. "
        "Если нужно открыть другую ссылку, добавь маркер `[FETCH: https://…]`."
    )
    return "\n".join(lines), urls
