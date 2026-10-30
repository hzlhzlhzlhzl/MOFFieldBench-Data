from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # the established mofesp env intentionally has no PyYAML
    yaml = None


def load_config(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    # Resolved experiment configurations are JSON.  Accepting them directly
    # lets fair reruns preserve every per-task setting exactly.
    config = json.loads(text) if text.lstrip().startswith("{") else (
        yaml.safe_load(text) if yaml is not None else _parse_simple_yaml(text)
    )
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        value: Any = yaml.safe_load(raw_value) if yaml is not None else _parse_scalar(raw_value)
        cursor = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return result


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")
        key, raw_value = stripped.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        raw_value = raw_value.strip()
        if raw_value:
            current[key.strip()] = _parse_scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            current[key.strip()] = child
            stack.append((indent, child))
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
