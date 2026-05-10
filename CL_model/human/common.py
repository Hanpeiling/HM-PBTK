from __future__ import annotations

from pathlib import Path
import json
from typing import Any

def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    required_files = ["human bin.xlsx", "GCN_Late.xlsx"]

    if data_dir is not None:
        folder = Path(data_dir).resolve()
        missing = [name for name in required_files if not (folder / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"The specified data directory does not contain: {missing}\n"
                f"Directory checked: {folder}"
            )
        return folder

    current_file = Path(__file__).resolve()

    candidates = [
        current_file.parent,                 # .../human
        current_file.parent.parent,          # .../CL上传
        Path.cwd(),                          # current working directory
    ]

    for folder in candidates:
        if all((folder / name).exists() for name in required_files):
            return folder

    raise FileNotFoundError(
        "Could not find the data folder automatically. "
        "Please run the script with --data-dir and point it to the folder "
        "that contains 'human bin.xlsx' and 'GCN_Late.xlsx'."
    )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_section(title: str) -> None:
    line = "=" * 100
    print("\n" + line)
    print(title)
    print(line)
