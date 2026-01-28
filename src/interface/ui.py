# src/interface/ui.py
# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import itertools
import shutil
import sys
import threading
import time
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal
try:
	from typing_extensions import Self
except:
	from typing import Self


class MessageType(Enum):
    """Console messages with associated symbols."""
    SUCCESS = "\u2705"  # ✅
    ERROR = "\u274c"    # ❌
    WARNING = "\u26a0\ufe0f"   # ⚠️
    INFO = "\u2139\ufe0f"      # ℹ️
    QUESTION = "\u2753"  # ❓
    ROCKET = "\U0001f680"  # 🚀
    FIRE = "\U0001f525"    # 🔥
    DNA = "\U0001f9ec"     # 🧬

SPINNER_DOTS = ["\u25cb", "\u25d1", "\u25d0", "\u25e5", "\u25e4", "\u25e3", "\u25e2", "\u25e1", "\u25e0", "\u25ef"]
SPINNER_BRAILLE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_SIMPLE = ["|", "/", "-", "\\"]
SPINNER_ARROWS = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]

# ==============================================================================
# SPINNER - Loading Animation
# ==============================================================================

class Spinner:
    """Context Manager for console loading animations."""

    def __init__(
        self,
        message: str = "Processing...",
        style: Literal["dots", "braille", "simple", "arrows"] = "braille",
        color: bool = True
    ):
        self.message = message
        self.stop_running = False
        self.color = color and self._supports_color()

        spinner_map = {
            "dots": SPINNER_DOTS,
            "braille": SPINNER_BRAILLE,
            "simple": SPINNER_SIMPLE,
            "arrows": SPINNER_ARROWS,
        }
        self.spinner_chars = itertools.cycle(spinner_map.get(style, SPINNER_BRAILLE))
        self.thread: threading.Thread | None = None

    @staticmethod
    def _supports_color() -> bool:
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def _spin(self):
        while not self.stop_running:
            char = next(self.spinner_chars)
            if self.color:
                # Cyan color
                output = f"\r\033[36m{char}\033[0m {self.message}"
            else:
                output = f"\r{char} {self.message}"

            sys.stdout.write(output)
            sys.stdout.flush()
            time.sleep(0.1)

        # Clear line
        cols = shutil.get_terminal_size((80, 24)).columns
        sys.stdout.write("\r" + " " * cols + "\r")
        sys.stdout.flush()

    def __enter__(self) -> Self:
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_running = True
        if self.thread:
            self.thread.join(timeout=1.0)
        # If an exception occurred, let it propagate; otherwise suppress
        return False

# ==============================================================================
# PROGRESS BAR
# ==============================================================================

class ProgressBar:
    """Simple progress bar for console."""

    def __init__(self, total: int | float, desc: str = "", width: int = 40, xtra_info: Any | None = None):
        self.total = total
        self.current = 0
        self.desc = desc
        self.width = width
        self.start_time = time.time()
        self.xtra_info = xtra_info
        self.bar_color = "\033[94m" # Blue default

    def update(self, n: int | float = 1):
        self.current = min(self.current + n, self.total)
        if isinstance(n, float):
            self.current = round(self.current, 2)
        self._render()

    def _render(self):
        if self.total == 0:
            return

        percent = self.current / self.total
        filled = int(self.width * percent)
        # Unicode block chars for smoother bar
        bar = f"{self.bar_color}{'█' * filled}\033[0m{'░' * (self.width - filled)}"

        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0

        # Avoid division by zero
        if rate > 0:
            eta = (self.total - self.current) / rate
        else:
            eta = 0

        info_str = ""
        if isinstance(self.xtra_info, Mapping):
            info_str = " | " + " ".join(f"{k}={v}" for k,v in self.xtra_info.items())
        elif self.xtra_info:
            info_str = f" | {self.xtra_info}"

        sys.stdout.write(
            f"\r{self.desc} ▕{bar}▏ {self.current}/{self.total} "
            f"[{elapsed:.1f}s < {eta:.1f}s, {rate:.2f}it/s]{info_str}"
        )
        sys.stdout.flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.write("\n")
        sys.stdout.flush()

# ==============================================================================
# CONSOLE IO
# ==============================================================================

class ConsoleIO:
    """Static helper for Console Input/Output operations."""

    @staticmethod
    def print_header(title: str, width: int = 60):
        print(f"\n\033[1m{'=' * width}\033[0m")
        print(f"\033[1m {title.center(width - 2)} \033[0m")
        print(f"\033[1m{'=' * width}\033[0m")

    @staticmethod
    def print_success(msg: str):
        print(f"{MessageType.SUCCESS.value}  \033[92m{msg}\033[0m") # Green text

    @staticmethod
    def print_error(msg: str):
        print(f"{MessageType.ERROR.value}  \033[91m{msg}\033[0m", file=sys.stderr) # Red text

    @staticmethod
    def print_warning(msg: str):
        print(f"{MessageType.WARNING.value}  \033[93m{msg}\033[0m") # Yellow text

    @staticmethod
    def print_step(msg: str):
        """Standard step message."""
        print(f"{MessageType.ROCKET.value}  \033[1m{msg}\033[0m")

    @staticmethod
    def print_info(msg: str):
        print(f"{MessageType.INFO.value}  {msg}")

    @staticmethod
    def print_dna(msg: str):
        """Special format for genomic operations."""
        print(f"{MessageType.DNA.value}  \033[36m{msg}\033[0m") # Cyan

    @staticmethod
    def input_path(prompt: str, must_exist: bool = True) -> Path:
        while True:
            val = input(f"{MessageType.QUESTION.value} {prompt}: ").strip()
            if not val:
                ConsoleIO.print_error("Path required.")
                continue

            path = Path(val).expanduser().resolve()
            if must_exist and not path.exists():
                ConsoleIO.print_error(f"Path not found: {path}")
                continue
            return path
