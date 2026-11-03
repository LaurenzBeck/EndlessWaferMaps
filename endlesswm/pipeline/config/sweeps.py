"""Helpers for parsing sweep expressions in CLI and YAML presets."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation


def _split_sweep_tokens(text: str) -> list[str]:
    """Split sweep expressions by top-level commas only."""
    tokens: list[str] = []
    current: list[str] = []
    depth = 0

    for char in text:
        if char in "{(":
            depth += 1
        elif char in "})" and depth > 0:
            depth -= 1

        if char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
            continue

        current.append(char)

    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tokens


def _parse_decimal(value: str, label: str) -> Decimal:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {label} value '{value}'.") from exc


def _expand_range_parts(parts: list[str], token: str, label: str, syntax_hint: str) -> list[Decimal]:
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid {label} range '{token}'. Use {syntax_hint}.")

    start = _parse_decimal(parts[0], label)
    end = _parse_decimal(parts[-1], label)
    step = _parse_decimal(parts[1], label) if len(parts) == 3 else (Decimal("-1") if end < start else Decimal("1"))

    if step == 0 or (step > 0 and start > end) or (step < 0 and start < end):
        raise ValueError(f"Invalid {label} range '{token}'.")

    values: list[Decimal] = []
    current = start
    while (current <= end) if step > 0 else (current >= end):
        values.append(current)
        current += step
    return values


def parse_numeric_sweep(text: str | None, label: str) -> list[Decimal]:
    """Parse explicit lists and inclusive numeric ranges."""
    if not text or not (token_text := text.strip()):
        return []

    if token_text.startswith("{") and token_text.endswith("}"):
        token_text = token_text[1:-1]

    values: list[Decimal] = []
    for token in _split_sweep_tokens(token_text):
        if token.startswith("(") and token.endswith(")"):
            values.extend(_expand_range_parts(_split_sweep_tokens(token[1:-1]), token, label, "(start,end) or (start,step,end)"))
        elif ":" in token:
            values.extend(_expand_range_parts([p.strip() for p in token.split(":")], token, label, "start:end or start:step:end"))
        else:
            values.append(_parse_decimal(token, label))
    return values


def parse_string_sweep(arg: str | None, label: str, default_value: str | None = None) -> list[str]:
    """Parse explicit string lists from comma lists or brace-wrapped sets."""
    if not arg or not (token_text := arg.strip()):
        return [default_value] if default_value is not None else []

    if token_text.startswith("{") and token_text.endswith("}"):
        token_text = token_text[1:-1]

    values = [token for token in _split_sweep_tokens(token_text) if token]
    if not values:
        raise ValueError(f"No valid {label} values provided.")
    return values


def parse_int_sweep(arg: str | None, label: str, default_value: int | None = None) -> list[int]:
    """Parse integer sweep values from explicit lists and ranges."""
    if not arg:
        return [default_value] if default_value is not None else []

    values: list[int] = []
    for val in parse_numeric_sweep(arg, label):
        if val != val.to_integral_value():
            raise ValueError(f"Invalid {label} value '{val}'. Expected integers.")
        values.append(int(val))

    if not values:
        raise ValueError(f"No valid {label} values provided.")
    return values


def parse_float_sweep(arg: str | None, label: str, default_value: float | None = None) -> list[float]:
    """Parse float sweep values from explicit lists and ranges."""
    if not arg:
        return [default_value] if default_value is not None else []

    values = [float(val) for val in parse_numeric_sweep(arg, label)]
    if not values:
        raise ValueError(f"No valid {label} values provided.")
    return values
