# run_shell.py — выполнение команд терминала по запросу ИИ
import subprocess
import os
from typing import Optional

RUN_TIMEOUT = 15  # секунд
RUN_CWD = os.getcwd()  # рабочая директория (можно заменить на безопасную)


def run_shell_command(cmd: str, timeout: int = RUN_TIMEOUT, cwd: Optional[str] = None) -> str:
    """
    Выполняет одну команду в shell. Возвращает объединённый stdout+stderr или сообщение об ошибке.
    """
    cmd = cmd.strip()
    if not cmd:
        return "(пустая команда)"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=(cwd if cwd is not None else RUN_CWD),
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        parts = [out] if out else []
        if err:
            parts.append(f"stderr:\n{err}")
        if result.returncode != 0:
            parts.append(f"(код выхода: {result.returncode})")
        return "\n".join(parts) if parts else "(пустой вывод)"
    except subprocess.TimeoutExpired:
        return f"(таймаут {timeout} с)"
    except Exception as e:
        return f"(ошибка: {e})"
