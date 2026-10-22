"""Pretty printer: renders an expression tree with the minimum parentheses.

``unparse(parse_expr(s))`` is a canonical form; parsing it again yields an
identical tree, which the tests check as a round-trip property.
"""

from __future__ import annotations

from .ast import AttrRef, Binary, Call, Expr, ListExpr, Literal, Subscript, Ternary, Unary
from .parser import PRECEDENCE
from .values import format_value

_TERNARY = 0
_UNARY = 100
_POSTFIX = 101


def _prec(node: Expr) -> int:
    if isinstance(node, Ternary):
        return _TERNARY
    if isinstance(node, Binary):
        return PRECEDENCE[node.op]
    if isinstance(node, Unary):
        return _UNARY
    if isinstance(node, Literal) and isinstance(node.value, (int, float)) and node.value < 0:
        return _UNARY
    return _POSTFIX + 1


def unparse(node: Expr) -> str:
    match node:
        case Literal(value):
            return format_value(value)
        case AttrRef(name, scope):
            return f"{scope}.{name}" if scope else name
        case Unary(op, operand):
            return op + _wrap(operand, _UNARY)
        case Binary(op, left, right):
            p = PRECEDENCE[op]
            # Left associative: the right child needs parens at equal precedence.
            return f"{_wrap(left, p)} {op} {_wrap(right, p + 1)}"
        case Ternary(cond, then, orelse):
            return f"{_wrap(cond, _TERNARY + 1)} ? {unparse(then)} : {unparse(orelse)}"
        case Call(name, args):
            return f"{name}({', '.join(unparse(a) for a in args)})"
        case ListExpr(items):
            return "{ " + ", ".join(unparse(i) for i in items) + " }" if items else "{ }"
        case Subscript(base, index):
            return f"{_wrap(base, _POSTFIX)}[{unparse(index)}]"
    raise TypeError(f"unknown node {node!r}")


def _wrap(node: Expr, min_prec: int) -> str:
    text = unparse(node)
    return f"({text})" if _prec(node) < min_prec else text


def pretty_ad(attrs: list[tuple[str, Expr]], indent: str = "    ") -> str:
    """Multi-line ad rendering with aligned ``=`` signs."""
    if not attrs:
        return "[ ]"
    width = max(len(name) for name, _ in attrs)
    lines = [f"{indent}{name.ljust(width)} = {unparse(expr)};" for name, expr in attrs]
    return "[\n" + "\n".join(lines) + "\n]"
