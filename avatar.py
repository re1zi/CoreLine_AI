import textwrap
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from PIL import Image

# rich.image может отсутствовать в установленной версии rich
try:
    from rich.image import Image as RichImage  # type: ignore
except Exception:  # noqa: BLE001 - совместимость со старыми версиями rich
    RichImage = None

console = Console()

class AvatarTerminal:
    def __init__(self, avatars_dir="avatars"):
        self.avatars_dir = avatars_dir
        self.images = {
            "idle": "idle.png",
            "thinking": "thinking.png",
            "speaking": "speaking.png",
            "sleeping": "sleep.png",
            # Эмоциональные состояния
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
        self.current_state = "idle"
        # Масштаб отображения аватара (1.0 = исходный размер)
        self.scale = 0.7

    def get_emotion_from_mood(self, mood_text):
        """Извлекает эмоцию из текста настроения и возвращает соответствующее состояние аватара."""
        mood_mapping = {
            "радость": "joy",
            "удовлетворение": "satisfaction",
            "безразличие": "indifference",
            "злость": "anger",
            "злость/недопонимание": "anger",  # обратная совместимость
            "грусть": "sadness",
            "страх": "fear",
            "отвращение": "disgust",
            "удивление": "surprise",
            "презрение": "contempt",
            "смущение": "blush"
        }
        return mood_mapping.get(mood_text.lower(), "idle")

    def show(self, state=None):
        """Отображение аватара в терминале"""
        if state:
            self.current_state = state
        path = f"{self.avatars_dir}/{self.images.get(self.current_state, 'idle.png')}"
        try:
            img = Image.open(path)
            console.clear()
            console.print(Panel.fit(f"[bold cyan]CoreLine[/bold cyan]", style="magenta"))
            console.print()
            if RichImage is not None:
                # Отображаем PNG в терминале через rich.image.Image (если доступно)
                # Масштабируем изображение перед выводом
                w, h = img.size
                new_w = max(1, int(w * self.scale))
                new_h = max(1, int(h * self.scale))
                resized = img.resize((new_w, new_h), Image.LANCZOS)
                rich_img = RichImage.from_pil(resized)
                console.print(rich_img)
            else:
                # Предпочитаем truecolor ANSI-рендер, если терминал поддерживает, иначе ASCII
                if getattr(console, "color_system", None) == "truecolor":
                    self._print_truecolor(img)
                else:
                    self._print_ascii(img)
        except Exception:
            # Если картинки нет, показываем ASCII-заглушку
            console.clear()
            console.print(Panel.fit(f"[bold cyan]CoreLine[/bold cyan]", style="magenta"))
            console.print()
            ascii_face = {
                "idle": "(・‿・)",
                "thinking": "(¬‿¬)",
                "speaking": "(＾▽＾)",
                "sleeping": "(-_-) zZ",
                # Эмоциональные ASCII-лица
                "joy": "ヽ(°〇°)ﾉ",
                "satisfaction": "(◡ ‿ ◡)",
                "indifference": "( ͡° ͜ʖ ͡°)",
                "anger": "(╬ಠ益ಠ)",
                "sadness": "(╥﹏╥)",
                "fear": "(° △ °)",
                "disgust": "(´Д`)",
                "surprise": "(⊙_⊙)",
                "contempt": "(¬_¬)",
                "blush": "(⁄ ⁄>⁄ ▽ ⁄<⁄ ⁄)"
            }
            console.print(f"[bold green]{ascii_face.get(self.current_state, '(・‿・)')}[/bold green]")

    def _get_ascii_lines(self, img):
        """Возвращает (список строк ASCII-арта, ширина в символах)."""
        shades = "@%#*+=-:. "
        term_width = max(20, min(100, console.width))
        term_width = max(10, int(term_width * self.scale))
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
        """Возвращает (список строк truecolor-арта, ширина в символах)."""
        target_width = max(20, min(console.width - 4, 100))
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
        """
        Возвращает (список строк аватара, ширина в символах).
        Для компоновки «текст слева | аватар справа».
        """
        if state:
            self.current_state = state
        path = f"{self.avatars_dir}/{self.images.get(self.current_state, 'idle.png')}"
        try:
            img = Image.open(path)
            use_truecolor = getattr(console, "color_system", None) == "truecolor"
            if RichImage is not None and not use_truecolor:
                w, h = img.size
                new_w = max(1, int(w * self.scale))
                new_h = max(1, int(h * self.scale))
                resized = img.resize((new_w, new_h), Image.LANCZOS)
                return self._get_ascii_lines(resized)
            if use_truecolor:
                return self._get_truecolor_lines(img)
            return self._get_ascii_lines(img)
        except Exception:
            ascii_face = {
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
                "blush": "(⁄ ⁄>⁄ ▽ ⁄<⁄ ⁄)"
            }
            face = ascii_face.get(self.current_state, "(・‿・)")
            return [f"[bold green]{face}[/bold green]"], len(face) + 10

    def show_with_text(self, state=None, text=""):
        """Показать аватар слева и текст ответа ИИ справа от него."""
        if state:
            self.current_state = state
        avatar_lines, avatar_width = self._get_avatar_lines()
        text_width = max(20, console.width - avatar_width - 2)
        # Сохраняем переносы строк: разбиваем по \n, каждую часть оборачиваем по ширине
        wrapped = []
        for para in (text or "").split("\n"):
            if para.strip():
                wrapped.extend(textwrap.wrap(para, width=text_width))
            else:
                wrapped.append("")
        max_lines = max(len(wrapped), len(avatar_lines))
        wrapped = wrapped + [""] * (max_lines - len(wrapped))
        avatar_lines = avatar_lines + [""] * (max_lines - len(avatar_lines))
        console.clear()
        console.print(Panel.fit(f"[bold cyan]CoreLine[/bold cyan]", style="magenta"))
        console.print()
        for i in range(max_lines):
            avatar_part = avatar_lines[i]
            text_part = wrapped[i][:text_width]
            # Текст как буквальная строка (без разбора Rich-разметки), чтобы [ ] не дублировались
            text_plain = Text(text_part)
            if avatar_part:
                console.print(avatar_part + "  ", end="")
                console.print(text_plain)
            else:
                console.print(" " * (avatar_width + 2), end="")
                console.print(text_plain)

    def _print_ascii(self, img):
        """Простой ASCII-арт из PIL Image, масштаб по ширине терминала."""
        # Градации от тёмного к светлому
        shades = "@%#*+=-:. "
        # Сохраняем пропорции: символы обычно выше, поэтому уменьшим высоту в 2 раза
        term_width = max(20, min(100, console.width))
        term_width = max(10, int(term_width * self.scale))
        # Конвертируем в L (grayscale) и ресайзим
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
        console.print("\n".join(lines))

    def _print_truecolor(self, img):
        """Рендер PNG через полублоки с ANSI truecolor.

        Используем символ '▄': верхний цвет = цвет нижнего пикселя, нижний цвет = цвет верхнего пикселя,
        так на один символ приходится 2 вертикальных пикселя.
        """
        # Целимся в ширину терминала, немного уменьшим чтобы влезало в панель
        target_width = max(20, min(console.width - 4, 100))
        target_width = max(10, int(target_width * self.scale))
        w, h = img.size
        if w <= 0 or h <= 0:
            return self._print_ascii(img)
        # Высоту делим на 2, так как два пикселя по вертикали на один символ
        scale = target_width / w
        target_height = max(2, int(h * scale))
        # Делаем высоту чётной для пар пикселей
        if target_height % 2 == 1:
            target_height += 1
        resized = img.convert("RGBA").resize((target_width, target_height))

        lines = []
        for y in range(0, target_height, 2):
            segments = []
            for x in range(target_width):
                r1, g1, b1, a1 = resized.getpixel((x, y))
                r2, g2, b2, a2 = resized.getpixel((x, y + 1))
                # Альфа-композит на чёрном фоне
                r1 = int(r1 * a1 / 255)
                g1 = int(g1 * a1 / 255)
                b1 = int(b1 * a1 / 255)
                r2 = int(r2 * a2 / 255)
                g2 = int(g2 * a2 / 255)
                b2 = int(b2 * a2 / 255)
                # Используем разметку rich для truecolor
                segments.append(f"[#%02x%02x%02x on #%02x%02x%02x]▄[/]" % (r2, g2, b2, r1, g1, b1))
            lines.append("".join(segments))
        console.print("\n".join(lines))

