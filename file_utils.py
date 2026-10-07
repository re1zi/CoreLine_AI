"""
Утилиты для работы с файлами в мультимодальных сообщениях.
Поддержка изображений и других файлов для моделей с vision.
"""
import base64
import mimetypes
import os
from pathlib import Path

# MIME-типы для изображений (поддерживаются большинством vision-моделей)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
# Дополнительные форматы (зависит от модели)
DOCUMENT_EXTENSIONS = {".pdf"}

# Максимальный размер файла в байтах (20 MB)
MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_ATTACHMENTS = 8

PROJECT_ROOT = Path(__file__).resolve().parent


def get_mime_type(filepath: str) -> str:
    """Определяет MIME-тип по расширению файла."""
    ext = Path(filepath).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".pdf": "application/pdf",
    }
    return mime_map.get(ext) or mimetypes.guess_type(filepath)[0] or "application/octet-stream"


def is_image(filepath: str) -> bool:
    """Проверяет, является ли файл изображением."""
    return Path(filepath).suffix.lower() in IMAGE_EXTENSIONS


def path_is_inside(path: Path, root: Path) -> bool:
    """True if path is the root or a file/dir under root (after resolve)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def load_file_as_base64(
    filepath: str,
    max_size: int = MAX_FILE_SIZE,
    restrict_to: Path | None = None,
) -> tuple[str, str, str] | None:
    """
    Загружает файл и возвращает (base64_str, mime_type, filename) или None при ошибке.

    restrict_to: if set, the resolved path must stay inside this directory.
    """
    expanded = os.path.expanduser(filepath.strip())
    path = Path(expanded).resolve()
    if restrict_to is not None and not path_is_inside(path, Path(restrict_to)):
        return None
    if not path.exists():
        return None
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
        if size > max_size:
            return None
        with open(path, "rb") as f:
            data = f.read()
        mime = get_mime_type(str(path))
        b64 = base64.b64encode(data).decode("ascii")
        return (b64, mime, path.name)
    except (OSError, IOError):
        return None


def client_attachments_to_files(
    attachments: list | None,
    max_size: int = MAX_FILE_SIZE,
    max_count: int = MAX_ATTACHMENTS,
) -> list[tuple[str, str, str]]:
    """Validate browser-uploaded attachments (base64) and return file_results tuples."""
    results: list[tuple[str, str, str]] = []
    for att in (attachments or [])[:max_count]:
        if not isinstance(att, dict):
            continue
        b64 = att.get("data")
        mime = str(att.get("type") or "application/octet-stream")
        name = Path(str(att.get("name") or "file")).name
        if not b64 or not isinstance(b64, str):
            continue
        if not (mime.startswith("image/") or mime == "application/pdf"):
            continue
        # base64 expands ~4/3; reject oversized payloads without decoding.
        if (len(b64) * 3) // 4 > max_size:
            continue
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception:
            continue
        if len(raw) > max_size:
            continue
        results.append((b64, mime, name))
    return results


def build_content_parts(text: str, file_results: list[tuple[str, str, str]]) -> list | str:
    """
    Строит content для сообщения user в формате OpenAI multimodal.
    file_results: список (base64_str, mime_type, filename).
    Возвращает список content parts или простую строку, если файлов нет.
    """
    if not file_results:
        return text.strip() if text.strip() else ""

    parts = []
    if text.strip():
        parts.append({"type": "text", "text": text.strip()})

    for b64, mime, _ in file_results:
        if mime.startswith("image/"):
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })
        elif mime == "application/pdf":
            # Некоторые API поддерживают PDF через image_url (частично)
            # или через отдельный тип. Используем image_url как fallback.
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })
        else:
            # Для прочих файлов — как текст в base64 (не все модели поддерживают)
            parts.append({
                "type": "text",
                "text": f"[Вложение: {mime}, base64 данные]\n(модель может не поддерживать этот тип файла)"
            })

    if not parts:
        return ""
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def parse_file_paths_from_input(user_input: str) -> tuple[str, list[str]]:
    """
    Извлекает пути к файлам из ввода вида *file путь *file "путь с пробелами" Текст.
    Возвращает (очищенный_текст, список_путей).
    """
    import re
    paths = []
    # Поддержка кавычек для путей с пробелами
    pattern = r'\*file\s+(?:"([^"]*)"|\'([^\']*)\'|([^\s*]+))'
    for m in re.finditer(pattern, user_input, re.IGNORECASE):
        p = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if p:
            paths.append(p)
    cleaned = re.sub(r"\*file\s+(?:\"[^\"]*\"|'[^']*'|[^\s*]+)", "", user_input, flags=re.IGNORECASE)
    return cleaned.strip(), paths
