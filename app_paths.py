from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = PROJECT_ROOT


def _resolve_path(value: str | None, default: str) -> Path:
    raw = (value or default).strip()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _resolve_path_list(value: str | None, default: str) -> list[Path]:
    raw = value if value is not None else default
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in raw.split(","):
        candidate = _resolve_path(item, ".")
        if candidate in seen:
            continue
        seen.add(candidate)
        paths.append(candidate)
    return paths


DATA_ROOT = _resolve_path(os.getenv("DATA_ROOT"), "data")
LOG_DIR = _resolve_path(os.getenv("LOG_DIR"), "logs")
RAG_INDEX_PATH = _resolve_path(os.getenv("RAG_INDEX_PATH"), "rag_index/index.json")
UPLOADS_ROOT = _resolve_path(os.getenv("UPLOADS_ROOT"), ".")
DOCUMENT_ROOTS = _resolve_path_list(os.getenv("DOCUMENT_ROOTS"), ".")
