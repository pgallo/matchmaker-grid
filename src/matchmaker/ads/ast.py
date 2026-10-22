"""Expression tree for the resource-ad language."""

from __future__ import annotations

from dataclasses import dataclass

from .values import Value


@dataclass(frozen=True, slots=True)
class Literal:
    value: Value


@dataclass(frozen=True, slots=True)
class AttrRef:
    name: str
    scope: str | None = None  # None, "MY" or "TARGET"


@dataclass(frozen=True, slots=True)
class Unary:
    op: str
    operand: Expr


@dataclass(frozen=True, slots=True)
class Binary:
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Ternary:
    cond: Expr
    then: Expr
    orelse: Expr


@dataclass(frozen=True, slots=True)
class Call:
    name: str
    args: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class ListExpr:
    items: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Subscript:
    base: Expr
    index: Expr


type Expr = Literal | AttrRef | Unary | Binary | Ternary | Call | ListExpr | Subscript


def conjuncts(expr: Expr) -> list[Expr]:
    """Flatten a top-level chain of ``&&`` into its clauses (used by diagnostics)."""
    if isinstance(expr, Binary) and expr.op == "&&":
        return conjuncts(expr.left) + conjuncts(expr.right)
    return [expr]
