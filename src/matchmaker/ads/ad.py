"""The ad: a case-insensitive mapping from attribute names to expressions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from .ast import AttrRef, Binary, Call, Expr, ListExpr, Literal, Subscript, Ternary, Unary
from .parser import parse_ad_attrs, parse_expr
from .printer import pretty_ad, unparse
from .values import ERROR, UNDEFINED, Value, _Sentinel

_EXPR_TYPES = (Literal, AttrRef, Unary, Binary, Ternary, Call, ListExpr, Subscript)


def to_expr(value: Any) -> Expr:
    """Lift a Python value into a literal expression tree."""
    if isinstance(value, _EXPR_TYPES):
        return value
    if isinstance(value, (_Sentinel, bool, int, float, str)):
        return Literal(value)
    if isinstance(value, (list, tuple)):
        return ListExpr(tuple(to_expr(v) for v in value))
    raise TypeError(f"cannot store {type(value).__name__} in an ad")


class Ad:
    """Attributes keyed case-insensitively; the original spelling is kept for printing."""

    __slots__ = ("_attrs",)

    def __init__(self, attrs: Mapping[str, Any] | None = None) -> None:
        self._attrs: dict[str, tuple[str, Expr]] = {}
        for k, v in (attrs or {}).items():
            self[k] = v

    @classmethod
    def parse(cls, text: str) -> Ad:
        ad = cls()
        for name, expr in parse_ad_attrs(text):
            ad[name] = expr
        return ad

    def set_expr(self, name: str, source: str) -> None:
        self[name] = parse_expr(source)

    def __setitem__(self, name: str, value: Any) -> None:
        self._attrs[name.lower()] = (name, to_expr(value))

    def __delitem__(self, name: str) -> None:
        del self._attrs[name.lower()]

    def lookup(self, name: str) -> Expr | None:
        entry = self._attrs.get(name.lower())
        return entry[1] if entry else None

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._attrs

    def __iter__(self) -> Iterator[str]:
        return (orig for orig, _ in self._attrs.values())

    def __len__(self) -> int:
        return len(self._attrs)

    def eval(self, name: str, target: Ad | None = None) -> Value:
        from .evaluator import evaluate_attr

        return evaluate_attr(self, name, target)

    def get(self, name: str, default: Value = UNDEFINED, target: Ad | None = None) -> Value:
        v = self.eval(name, target)
        return default if v is UNDEFINED or v is ERROR else v

    def copy(self) -> Ad:
        new = Ad()
        new._attrs = dict(self._attrs)
        return new

    def items(self) -> list[tuple[str, Expr]]:
        return list(self._attrs.values())

    def to_json(self) -> dict[str, str]:
        """Serialize every attribute as expression source (lossless)."""
        return {orig: unparse(expr) for orig, expr in self._attrs.values()}

    @classmethod
    def from_json(cls, data: Mapping[str, str]) -> Ad:
        ad = cls()
        for k, src in data.items():
            ad[k] = parse_expr(src)
        return ad

    def __str__(self) -> str:
        return pretty_ad(self.items())

    def __repr__(self) -> str:
        return f"Ad({self.to_json()!r})"
