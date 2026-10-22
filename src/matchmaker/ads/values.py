"""Runtime values of the resource-ad language.

Values are plain Python objects (bool, int, float, str, list) plus two
singletons: UNDEFINED (an attribute is missing, or depends on something
missing) and ERROR (the expression is ill-typed, e.g. ``"a" * 2``).
"""

from __future__ import annotations

from typing import Final


class _Sentinel:
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __bool__(self) -> bool:
        # Truthiness of UNDEFINED/ERROR is a classic source of bugs; force
        # callers to use is_true() so three-valued logic stays explicit.
        raise TypeError(f"{self._name} has no Python truth value; use is_true()")

    def __reduce__(self) -> str:
        return self._name


UNDEFINED: Final = _Sentinel("UNDEFINED")
ERROR: Final = _Sentinel("ERROR")

type Value = bool | int | float | str | list[Value] | _Sentinel


def is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_true(v: Value) -> bool:
    """True only for the boolean value true (matchmaking semantics)."""
    return v is True


def type_name(v: Value) -> str:
    if v is UNDEFINED:
        return "undefined"
    if v is ERROR:
        return "error"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "real"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "list"
    return type(v).__name__


def format_value(v: Value) -> str:
    """Render a value in resource-ad literal syntax."""
    if v is UNDEFINED:
        return "undefined"
    if v is ERROR:
        return "error"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return quote(v)
    if isinstance(v, list):
        return "{ " + ", ".join(format_value(x) for x in v) + " }" if v else "{ }"
    raise TypeError(f"not a resource-ad value: {v!r}")


def quote(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{escaped}"'
