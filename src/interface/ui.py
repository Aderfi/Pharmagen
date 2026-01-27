# src/interface/ui.py
# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import itertools
import shutil
import sys
import threading
import time
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class MessageType(Enum):
    """Console messages with associated symbols."""
    SUCCESS = "\u2705"  # ✅
    ERROR = "\u274c"    # ❌
    WARNING = "\u26a0\ufe0f"   # ⚠️
    INFO = "\u2139\ufe0f"      # ℹ️
    QUESTION = "\u2753"  # ❓
    ROCKET = "\U0001f680"  # 🚀
    FIRE = "\U0001f525"    # 🔥

SPINNER_DOTS = ["\u25cb", "\u25d1", "\u25d0", "\u25e5", "\u25e4", "\u25e3", "\u25e2", "\u25e1", "\u25e0", "\u25ef"]
SPINNER_BRAILLE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_SIMPLE = ["|", "/", "-", "\\"]
SPINNER_ARROWS = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]

# ==============================================================================
# SPINNER - Loading Animation
# ==============================================================================

class Spinner:
    """Context Manager for console loading animations.

    Usage:
    >>> with Spinner("Loading data..."):
    >>>     time.sleep(2)

        # Custom spinner style
    >>> with Spinner("Processing.. .", style="braille"):
    >>>     heavy_computation()
    """

    def __init__(
        self,
        message:  str = "Processing...",
        style: Literal["dots", "braille", "simple", "arrows"] = "braille",
        color: bool = True
    ):
        self.message = message
        self. stop_running = False
        self. color = color and self._supports_color()

        # Select spinner style
        spinner_map = {
            "dots":  SPINNER_DOTS,
            "braille": SPINNER_BRAILLE,
            "simple":  SPINNER_SIMPLE,
            "arrows": SPINNER_ARROWS,
        }
        self.spinner_chars = itertools.cycle(spinner_map.get(style, SPINNER_BRAILLE))
        self.thread: threading.Thread | None = None

    @staticmethod
    def _supports_color() -> bool:
        """Check if terminal supports ANSI colors."""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def _spin(self):
        """Animation loop running in separate thread."""
        while not self.stop_running:
            char = next(self.spinner_chars)
            if self.color:
                # Cyan color for spinner
                output = f"\r\033[36m{char}\033[0m {self.message}"
            else:
                output = f"\r{char} {self.message}"

            sys.stdout.write(output)
            sys.stdout.flush()
            time.sleep(0.1)

        # Clear line properly
        terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
        sys.stdout.write("\r" + " " * terminal_width + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_running = True
        if self.thread:
            self. thread.join(timeout=1.0)
        return False

# ==============================================================================
# PROGRESS BAR - For Iterables
# ==============================================================================

class ProgressBar:
    """
    Simple progress bar for console (alternative to tqdm for minimal dependencies).

    Usage:
    >>>    items = range(100)
    >>>    with ProgressBar(total=100, desc="Processing") as pbar:
    >>>        for item in items:
    >>>            # Do work
    >>>            pbar.update(1)
    """

    def __init__(self, total: int, desc: str = "", width: int = 40, xtra_info: Any | None = None):
        self.total = total
        self.current = 0
        self.desc = desc
        self.width = width
        self.start_time = time.time()
        self.xtra_info = xtra_info
        self.color = self._color_progressbar(default=True, choice='blue')

    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current = min(self.current + n, self.total)
        self._render()


    def _render(self):
        """Render the progress bar."""
        if self.total == 0:
            return
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = f"{self.color}█\033[0m" * filled + f"{self.color}░\033[0m" * (self.width - filled)

        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self. current) / rate if rate > 0 else 0

        sys.stdout.write(
            f"\r{self.desc} |{bar}| {self.current}/{self.total} "
            f"[{elapsed:.1f}s < {eta:.1f}s, {rate:.2f}it/s]")
        if isinstance(self.xtra_info, Mapping):
            sys.stdout.write(f"{' | '.join(f'{k}={v}' for k, v in (self.xtra_info or {}).items())}")
        elif isinstance(self.xtra_info, str):
            sys.stdout.write(f" | {self.xtra_info}")
        else:
            ""

        sys.stdout.flush()

    def _color_progressbar(self, default: bool = True, choice: str = 'blue'):
        """Add color to the progress bar"""
        if not default:
            user_choice = input("Progress bar color ([blue]/red/yellow/green/n): ").strip().lower()
            choice = user_choice if user_choice else choice

        color_map = {
            'blue': "\033[94m",
            'red': "\033[91m",
            'yellow': "\033[93m",
            'green':  "\033[92m",
            'n': "\033[0m",
            'no': "\033[0m",
        }

        return color_map.get(choice, "\033[0m")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.write("\n")
        sys.stdout.flush()

# ==============================================================================
# CONSOLE IO - Input/Output Utilities
# ==============================================================================

class ConsoleIO:
    """
    Static helper for Console Input/Output operations with validation.
    """
    # -------------------------------------------------------------------------
    # OUTPUT METHODS
    # -------------------------------------------------------------------------

    @staticmethod
    def print_header(title: str, char: str = "\u2500", width: int = 60):
        """Print a formatted header."""
        print("\n" + char * width)
        print(f" {title}")
        print(char * width)

    @staticmethod
    def print_success(msg: Any):
        """Print success message with ✅."""
        print(f"{MessageType.SUCCESS.value}  {msg}")

    @staticmethod
    def print_error(msg: Any):
        """Print error message with ❌."""
        print(f"{MessageType.ERROR.value}  {msg}", file=sys.stderr)

    @staticmethod
    def print_warning(msg: Any):
        """Print warning message with ⚠️."""
        print(f"{MessageType.WARNING.value}  {msg}")

    @staticmethod
    def print_info(msg: Any, char: Literal["ℹ️", "🚀", "🔥"] | None = None):
        """Print info message with. ℹ️ or custom char."""
        if char:
            print(f"{char} {msg}")
        else:
            print(f"{MessageType.INFO.value} {msg}")

    @staticmethod
    def print_divider(char: str = "-", width: int = 60):
        """Print a simple divider line."""
        print(char * width)

    # -------------------------------------------------------------------------
    # INPUT METHODS WITH VALIDATION
    # -------------------------------------------------------------------------

    @staticmethod
    def input_path(
        prompt:  str,
        default: Path | None = None,
        must_exist: bool = True,
        file_extensions: list[str] | None = None
    ) -> Path:
        """
        Prompt for a file/directory path with validation.

        Args:
            prompt: Message to display
            default: Default path if user presses Enter
            must_exist:  Whether path must exist
            file_extensions:  List of allowed extensions (e.g., ['.csv', '.tsv'])

        Returns:
            Validated Path object
        """
        while True:
            default_str = f" [{default}]" if default else ""
            path_str = input(f"{prompt}{default_str}: ").strip()

            # Use default if provided
            if not path_str and default:
                return default

            if not path_str:
                ConsoleIO.print_error("A path is required.")
                continue

            path = Path(path_str).expanduser().resolve()

            # Existence check
            if must_exist and not path.exists():
                ConsoleIO.print_error(f"Path does not exist: {path}")
                continue

            # Extension check
            if file_extensions and path.is_file():
                if path.suffix.lower() not in [ext.lower() for ext in file_extensions]:
                    ConsoleIO.print_error(
                        f"Invalid file type. Expected:  {', '.join(file_extensions)}"
                    )
                    continue
            return path

    @staticmethod
    def input_int(
        prompt: str,
        default: int | None = None,
        min_val: int | None = None,
        max_val: int | None = None
    ) -> int:
        """
        Prompt for an integer with validation.

        Args:
            prompt: Message to display
            default: Default value if user presses Enter
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated integer
        """
        while True:
            default_str = f" [default: {default}]" if default is not None else ""
            value_str = input(f"{prompt}{default_str}: ").strip()

            # Use default
            if not value_str and default is not None:
                return default

            # Parse integer
            try:
                value = int(value_str)
            except ValueError:
                ConsoleIO.print_error("Please enter a valid integer.")
                continue

            # Range validation
            if min_val is not None and value < min_val:
                ConsoleIO.print_error(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                ConsoleIO.print_error(f"Value must be <= {max_val}")
                continue

            return value

    @staticmethod
    def input_float(
        prompt: str,
        default: float | None = None,
        min_val:  float | None = None,
        max_val: float | None = None
    ) -> float:
        """Prompt for a float with validation."""
        while True:
            default_str = f" [default: {default}]" if default is not None else ""
            value_str = input(f"{prompt}{default_str}: ").strip()

            if not value_str and default is not None:
                return default

            try:
                value = float(value_str)
            except ValueError:
                ConsoleIO.print_error("Please enter a valid number.")
                continue

            if min_val is not None and value < min_val:
                ConsoleIO.print_error(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                ConsoleIO.print_error(f"Value must be <= {max_val}")
                continue

            return value

    @staticmethod
    def input_choice(
        prompt: str,
        choices: list[str],
        default: str | None = None,
        case_sensitive: bool = False
    ) -> str:
        """
        Prompt for a choice from a list. (Enumerated input)

        Args:
            prompt: Message to display
            choices: List of valid options
            default: Default choice if user presses Enter
            case_sensitive: Whether to match case-sensitively

        Returns:
            Selected choice
        """
        while True:
            choices_str = "/".join(choices)
            default_str = f" [{default}]" if default else ""

            for i, choice in enumerate(choices):
                print(f"{i+1}.  {choice}")

            value = input(f"Input the number of your choice. {default_str}: ").strip()
            index = int(value) - 1
            if 0 <= index < len(choices):
                with Spinner(f"Loading: {choices[index]}", style="braille").__enter__():
                    time.sleep(1)  # Simulate loading
                return choices[index]
            else:
                ConsoleIO.print_error(f"Invalid choice.  Options: {choices_str}")

    @staticmethod
    def confirm(prompt: str, default: bool = False) -> bool:
        """
        Ask for yes/no confirmation.

        Args:
            prompt: Question to ask
            default: Default answer if user presses Enter

        Returns:
            True for yes, False for no
        """
        default_str = "[Y/n]" if default else "[y/N]"
        while True:
            answer = input(f"{prompt} {default_str}: ").strip().lower()

            if not answer:
                return default

            if answer in ('y', 'yes', 'si', 'sí', 's', '+'):
                return True
            elif answer in ('n', 'no', '-'):
                return False
            else:
                ConsoleIO.print_error("Please answer 'y' or 'n'.")

    @staticmethod
    def clear_screen():
        """Clear the console screen (cross-platform)."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    """
    TEST CONSOLE IO MODULE
    """
    print("Testing Spinner...")
    with Spinner("Loading data.. .", style="braille"):
        time.sleep(2)
    ConsoleIO.print_success("Spinner test completed!")

    # Test Messages
    ConsoleIO.print_header("Console IO Tests")
    ConsoleIO.print_info("This is an info message")
    ConsoleIO.print_warning("This is a warning")
    ConsoleIO.print_error("This is an error")
    ConsoleIO.print_success("This is a success message")

    # Test Input Methods
    ConsoleIO.print_divider()

    # Integer input
    age = ConsoleIO.input_int("Enter age", default=25, min_val=0, max_val=120)
    print(f"You entered: {age}")

    # Choice input
    color = ConsoleIO.input_choice("Pick a color", ["red", "green", "blue"], default="blue")
    print(f"You chose: {color}")

    # Confirmation
    confirmed = ConsoleIO.confirm("Continue?", default=True)
    print(f"Confirmed: {confirmed}")

    # Progress bar
    print("\nTesting Progress Bar...")
    xtra_info = input("Enter extra info to display in progress bar: ").strip()
    with ProgressBar(total=50, desc="Processing", xtra_info=xtra_info) as pbar:
        for i in range(50):
            time.sleep(0.05)
            pbar.update(1)

    ConsoleIO.print_success("All tests completed!")
