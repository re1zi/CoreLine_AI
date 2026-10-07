"""Avatar mood map, CLI/TUI rendering, and web filename allowlist."""
from __future__ import annotations

import textwrap
from pathlib import Path

try:
    from PIL import Image
except Exception:  # noqa: BLE001 - optional for TUI/web without PIL
    Image = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except Exception:  # noqa: BLE001
    Console = None
    Panel = None
    Text = None

try:
    from rich.image import Image as RichImage  # type: ignore
except Exception:  # noqa: BLE001 - older rich has no rich.image
    RichImage = None

# Keep in sync with tui/src/index.ts pixel payload sizing.
AVATAR_PIXEL_WIDTH = 60
AVATAR_PIXEL_HEIGHT = 50

AVATAR_FILES = {
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
    "blush": "blush.png",
}

# Web static/avatars uses talking.png instead of speaking.png.
WEB_AVATAR_FILES = {
    **AVATAR_FILES,
    "speaking": "talking.png",
}

MOOD_TO_STATE = {
    "радость": "joy",
    "удовлетворение": "satisfaction",
    "безразличие": "indifference",
    "злость": "anger",
    "злость/недопонимание": "anger",
    "грусть": "sadness",
    "страх": "fear",
    "отвращение": "disgust",
    "удивление": "surprise",
    "презрение": "contempt",
    "смущение": "blush",
}

ASCII_FACES = {
    "idle": "(・‿・)",
    "thinking": "(¬‿¬)",
    "speaking": "(＾▽＾)",
    "sleeping": "(-_-) zZ",
    "joy": "ヽ(°〇°)ﾉ",
    "satisfaction": "(◡ ‿ ◡)",
    "indifference": "( ͡° ͜ʖ ͡°)",
    "anger": "(╬ಠ益ಠ)",
    "sadness": "(╥﹏╥)",
    "fear": "(° △ °)",
    "disgust": "(´Д`)",
    "surprise": "(⊙_⊙)",
    "contempt": "(¬_¬)",
    "blush": "(⁄ ⁄>⁄ ▽ ⁄<⁄ ⁄)",
}

ALLOWED_AVATAR_FILENAMES = frozenset(AVATAR_FILES.values()) | frozenset(WEB_AVATAR_FILES.values()) | {
    "error.png",
    "talking.png",
}

_console = Console() if Console is not None else None


def get_emotion_from_mood(mood_text: str) -> str:
    return MOOD_TO_STATE.get((mood_text or "").lower().strip(), "idle")


def avatar_filename(state: str, *, web: bool = False) -> str:
    table = WEB_AVATAR_FILES if web else AVATAR_FILES
    return table.get(state, "idle.png")


def is_allowed_avatar_file(filename: str) -> bool:
    return Path(filename).name in ALLOWED_AVATAR_FILENAMES


class Avatar:
    """Mood state plus optional terminal / pixel rendering."""

    def __init__(self, avatars_dir="avatars", *, headless=False, web=False):
        self.avatars_dir = avatars_dir
        self.headless = headless
        self.images = dict(WEB_AVATAR_FILES if web else AVATAR_FILES)
        self.current_state = "idle"
        self.scale = 0.7

    def get_emotion_from_mood(self, mood_text):
        return get_emotion_from_mood(mood_text)

    def _face(self) -> str:
        return ASCII_FACES.get(self.current_state, "(・‿・)")

    def _image_path(self) -> Path:
        name = self.images.get(self.current_state, "idle.png")
        return Path(self.avatars_dir) / name

    def _open_image(self):
        if Image is None:
            return None
        path = self._image_path()
        if not path.is_file():
            return None
        try:
            return Image.open(path)
        except Exception:
            return None

    def show(self, state=None):
        if state:
            self.current_state = state
        if self.headless or _console is None:
            return
        img = self._open_image()
        _console.clear()
        _console.print(Panel.fit("[bold cyan]CoreLine[/bold cyan]", style="magenta"))
        _console.print()
        if img is None:
            _console.print(f"[bold green]{self._face()}[/bold green]")
            return
        try:
            if RichImage is not None:
                w, h = img.size
                new_w = max(1, int(w * self.scale))
                new_h = max(1, int(h * self.scale))
                resized = img.resize((new_w, new_h), Image.LANCZOS)
                _console.print(RichImage.from_pil(resized))
            elif getattr(_console, "color_system", None) == "truecolor":
                self._print_truecolor(img)
            else:
                self._print_ascii(img)
        except Exception:
            _console.print(f"[bold green]{self._face()}[/bold green]")

    def show_with_text(self, state=None, text=""):
        if state:
            self.current_state = state
        if self.headless or _console is None or Text is None:
            return
        avatar_lines, avatar_width = self._get_avatar_lines()
        text_width = max(20, _console.width - avatar_width - 2)
        wrapped = []
        for para in (text or "").split("\n"):
            if para.strip():
                wrapped.extend(textwrap.wrap(para, width=text_width))
            else:
                wrapped.append("")
        max_lines = max(len(wrapped), len(avatar_lines))
        wrapped = wrapped + [""] * (max_lines - len(wrapped))
        avatar_lines = avatar_lines + [""] * (max_lines - len(avatar_lines))
        _console.clear()
        _console.print(Panel.fit("[bold cyan]CoreLine[/bold cyan]", style="magenta"))
        _console.print()
        for i in range(max_lines):
            avatar_part = avatar_lines[i]
            text_plain = Text(wrapped[i][:text_width])
            if avatar_part:
                _console.print(avatar_part + "  ", end="")
                _console.print(text_plain)
            else:
                _console.print(" " * (avatar_width + 2), end="")
                _console.print(text_plain)

    def _get_ascii_lines(self, img, width=None):
        shades = "@%#*+=-:. "
        if width is None:
            term_width = max(20, min(100, _console.width if _console else 80))
            term_width = max(10, int(term_width * self.scale))
        else:
            term_width = max(10, width)
        w, h = img.size
        aspect = h / max(1, w)
        new_w = term_width
        new_h = max(1, int(aspect * new_w * 0.5))
        gray = img.convert("L").resize((new_w, new_h))
        lines = []
        for y in range(new_h):
            row = []
            for x in range(new_w):
                v = gray.getpixel((x, y))
                row.append(shades[int(v / 255 * (len(shades) - 1))])
            lines.append("".join(row))
        return lines, new_w

    def _get_truecolor_lines(self, img):
        term_w = _console.width if _console else 80
        target_width = max(20, min(term_w - 4, 100))
        target_width = max(10, int(target_width * self.scale))
        w, h = img.size
        if w <= 0 or h <= 0:
            return self._get_ascii_lines(img)
        scale = target_width / w
        target_height = max(2, int(h * scale))
        if target_height % 2 == 1:
            target_height += 1
        resized = img.convert("RGBA").resize((target_width, target_height))
        lines = []
        for y in range(0, target_height, 2):
            segments = []
            for x in range(target_width):
                r1, g1, b1, a1 = resized.getpixel((x, y))
                r2, g2, b2, a2 = resized.getpixel((x, y + 1))
                r1 = int(r1 * a1 / 255)
                g1 = int(g1 * a1 / 255)
                b1 = int(b1 * a1 / 255)
                r2 = int(r2 * a2 / 255)
                g2 = int(g2 * a2 / 255)
                b2 = int(b2 * a2 / 255)
                segments.append(f"[#%02x%02x%02x on #%02x%02x%02x]▄[/]" % (r2, g2, b2, r1, g1, b1))
            lines.append("".join(segments))
        return lines, target_width

    def _get_avatar_lines(self, state=None):
        if state:
            self.current_state = state
        img = self._open_image()
        if img is None:
            face = self._face()
            return [f"[bold green]{face}[/bold green]"], len(face) + 10
        use_truecolor = getattr(_console, "color_system", None) == "truecolor"
        if RichImage is not None and not use_truecolor:
            w, h = img.size
            new_w = max(1, int(w * self.scale))
            new_h = max(1, int(h * self.scale))
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            return self._get_ascii_lines(resized)
        if use_truecolor:
            return self._get_truecolor_lines(img)
        return self._get_ascii_lines(img)

    def _print_ascii(self, img):
        lines, _ = self._get_ascii_lines(img)
        _console.print("\n".join(lines))

    def _print_truecolor(self, img):
        lines, _ = self._get_truecolor_lines(img)
        _console.print("\n".join(lines))

    def render_ascii(self, state=None, width=28):
        if state:
            self.current_state = state
        img = self._open_image()
        if img is None:
            return self._face()
        try:
            w, h = img.size
            if w <= 0 or h <= 0:
                return self._face()
            lines, _ = self._get_ascii_lines(img, width=width)
            return "\n".join(lines)
        except Exception:
            return self._face()

    def render_pixels(self, state=None, width=AVATAR_PIXEL_WIDTH, height=AVATAR_PIXEL_HEIGHT):
        if state:
            self.current_state = state
        img = self._open_image()
        if img is None or Image is None:
            return None
        try:
            src = img.convert("RGBA")
            if width <= 0 or height <= 0:
                return None
            sw, sh = src.size
            if sw <= 0 or sh <= 0:
                return None
            scale = min(width / sw, height / sh)
            nw = max(1, int(sw * scale))
            nh = max(1, int(sh * scale))
            resized = src.resize((nw, nh), Image.LANCZOS)
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ox = (width - nw) // 2
            oy = (height - nh) // 2
            canvas.paste(resized, (ox, oy), resized)
            pixels = []
            for y in range(height):
                row = []
                for x in range(width):
                    r, g, b, a = canvas.getpixel((x, y))
                    row.append([int(r), int(g), int(b), int(a)])
                pixels.append(row)
            return pixels
        except Exception:
            return None


class AvatarTerminal(Avatar):
    """CLI avatar (prints to the terminal)."""


class HeadlessAvatar(Avatar):
    """State + ASCII/pixels for the TUI/web; does not clear the terminal."""

    def __init__(self, avatars_dir="avatars", *, web=False):
        super().__init__(avatars_dir, headless=True, web=web)
