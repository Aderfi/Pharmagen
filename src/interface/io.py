# Pharmagen - IO Utilities
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from src.cfg.manager import DIRS, METADATA, VERSION

LOGS_DIR = DIRS["logs"]


def save_json(data: dict[str, Any], path: str | Path):
    """Saves a dictionary to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Loads a JSON file into a dictionary."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def json_serial_adapter(obj: Any) -> Any:
    """
    JSON serialization adapter for non-standard types (Path, sets, etc).
    Usage: json.dump(..., default=json_serial_adapter)
    """
    if isinstance(obj, (Path)):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable by custom adapter")


def welcome_message():
    msg = f"""
    ============================================
            PHARMAGEN v{VERSION}
    ============================================
    Software para farmacogenética y deep learning.

    Logs: {LOGS_DIR}
    ============================================
    """
    print(msg)


def print_gnu_notice():
    """Imprime el aviso legal"""
    start_year = 2025
    current_year = datetime.now().year

    if current_year > start_year:
        year_str = f"{start_year}-{current_year}"
    else:
        year_str = str(start_year)

    author = "Adrim Hamed Outmani (@Aderfi)"
    program = METADATA.get("project_name", "Pharmagen")

    notice = f"""
    {program} Copyright (C) {year_str} {author}\n
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.
    """
    print(notice)


def print_warranty_details():
    """Texto completo para 'show w'."""
    print("\n" + "=" * 60)
    print("NO WARRANTY")
    print("=" * 60)
    print("""
    BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
    FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT WHEN
    OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
    PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
    OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS
    TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE
    PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
    REPAIR OR CORRECTION.
    """)
    input("\nPresione [Enter] para volver...")


def print_conditions_details():
    """Texto completo para 'show c'."""
    print("\n" + "=" * 60)
    print("REDISTRIBUTION CONDITIONS")
    print("=" * 60)
    print("""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
    """)
    input("\nPresione [Enter] para volver...")
