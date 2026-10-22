"""Evaluator with three-valued logic plus ERROR.

Strictness rules (applied in this order for most operators):

1. ERROR in any evaluated operand gives ERROR.
2. UNDEFINED in any evaluated operand gives UNDEFINED.
3. A type mismatch gives ERROR.

The exceptions are the boolean connectives and the meta-equality operators.
``&&`` and ``||`` are non-strict where the result is already decided
(``false && UNDEFINED`` is false, ``true || UNDEFINED`` is true), which is what
lets a machine write ``Requirements = TARGET.Owner == "alice" || Load < 0.3``
and still match jobs that lack an ``Owner``. ``=?=`` and ``=!=`` never return
UNDEFINED: they compare type and value, so ``x =?= undefined`` tests presence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from functools import lru_cache

from .ad import Ad
from .ast import AttrRef, Binary, Call, Expr, ListExpr, Literal, Subscript, Ternary, Unary
from .values import ERROR, UNDEFINED, Value, format_value, is_number

_MAX_DEPTH = 64


class _Context:
    __slots__ = ("my", "stack", "target")

    def __init__(self, my: Ad | None, target: Ad | None, stack: tuple[tuple[int, str], ...]) -> None:
        self.my = my
        self.target = target
        self.stack = stack


def evaluate(expr: Expr, my: Ad | None = None, target: Ad | None = None) -> Value:
    return _eval(expr, _Context(my, target, ()))


def evaluate_attr(ad: Ad, name: str, target: Ad | None = None) -> Value:
    return _eval(AttrRef(name, "MY"), _Context(ad, target, ()))


# --------------------------------------------------------------------------
# core


def _eval(node: Expr, ctx: _Context) -> Value:
    match node:
        case Literal(value):
            return value
        case AttrRef(name, scope):
            return _lookup(name, scope, ctx)
        case Unary(op, operand):
            return _unary(op, _eval(operand, ctx))
        case Binary("&&", left, right):
            return _and(left, right, ctx)
        case Binary("||", left, right):
            return _or(left, right, ctx)
        case Binary(op, left, right):
            return _binary(op, _eval(left, ctx), _eval(right, ctx))
        case Ternary(cond, then, orelse):
            c = _as_condition(_eval(cond, ctx))
            if c is UNDEFINED or c is ERROR:
                return c
            return _eval(then if c else orelse, ctx)
        case Call(name, args):
            return _call(name, args, ctx)
        case ListExpr(items):
            return [_eval(i, ctx) for i in items]
        case Subscript(base, index):
            return _subscript(_eval(base, ctx), _eval(index, ctx))
    return ERROR


def _lookup(name: str, scope: str | None, ctx: _Context) -> Value:
    # Unscoped names resolve in MY first, then TARGET: the conventional rule
    # that lets "Memory" in a job's Requirements mean the machine's Memory.
    if scope == "MY" or scope is None:
        if ctx.my is not None and (expr := ctx.my.lookup(name)) is not None:
            return _eval_in(ctx.my, ctx.target, name, expr, ctx)
        if scope == "MY":
            return UNDEFINED
    if ctx.target is not None and (expr := ctx.target.lookup(name)) is not None:
        # The target's attribute is evaluated from the target's point of view.
        return _eval_in(ctx.target, ctx.my, name, expr, ctx)
    return UNDEFINED


def _eval_in(owner: Ad, other: Ad | None, name: str, expr: Expr, ctx: _Context) -> Value:
    key = (id(owner), name.lower())
    if key in ctx.stack or len(ctx.stack) >= _MAX_DEPTH:
        return ERROR  # circular attribute definition
    return _eval(expr, _Context(owner, other, (*ctx.stack, key)))


def _as_condition(v: Value) -> Value:
    if v is UNDEFINED or v is ERROR or isinstance(v, bool):
        return v
    if is_number(v):
        return v != 0
    return ERROR


def _and(left: Expr, right: Expr, ctx: _Context) -> Value:
    a = _as_condition(_eval(left, ctx))
    if a is ERROR or a is False:
        return a
    b = _as_condition(_eval(right, ctx))
    if b is ERROR or b is False:
        return b
    if a is UNDEFINED or b is UNDEFINED:
        return UNDEFINED
    return True


def _or(left: Expr, right: Expr, ctx: _Context) -> Value:
    a = _as_condition(_eval(left, ctx))
    if a is ERROR or a is True:
        return a
    b = _as_condition(_eval(right, ctx))
    if b is ERROR or b is True:
        return b
    if a is UNDEFINED or b is UNDEFINED:
        return UNDEFINED
    return False


def _unary(op: str, v: Value) -> Value:
    if v is ERROR or v is UNDEFINED:
        return v
    if op == "!":
        c = _as_condition(v)
        return c if c is ERROR else not c
    if is_number(v):
        return -v if op == "-" else v  # type: ignore[operator]
    return ERROR


def _identical(a: Value, b: Value) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b, strict=True))  # type: ignore[arg-type]
    return a is b or a == b


def _binary(op: str, a: Value, b: Value) -> Value:
    if op == "=?=":
        return _identical(a, b)
    if op == "=!=":
        return not _identical(a, b)
    if a is ERROR or b is ERROR:
        return ERROR
    if a is UNDEFINED or b is UNDEFINED:
        return UNDEFINED
    if op in ("==", "!=", "<", "<=", ">", ">="):
        return _compare(op, a, b)
    if not (is_number(a) and is_number(b)):
        return ERROR
    x, y = a, b
    match op:
        case "+":
            return x + y  # type: ignore[operator]
        case "-":
            return x - y  # type: ignore[operator]
        case "*":
            return x * y  # type: ignore[operator]
        case "/":
            if y == 0:
                return ERROR
            if isinstance(x, int) and isinstance(y, int):
                return int(x / y)  # truncate toward zero, like C
            return x / y  # type: ignore[operator]
        case "%":
            if y == 0:
                return ERROR
            if isinstance(x, int) and isinstance(y, int):
                return int(math.fmod(x, y))
            return math.fmod(x, y)  # type: ignore[arg-type]
    return ERROR


_CMP: dict[str, Callable[[object, object], bool]] = {
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    "<": lambda x, y: x < y,  # type: ignore[operator]
    "<=": lambda x, y: x <= y,  # type: ignore[operator]
    ">": lambda x, y: x > y,  # type: ignore[operator]
    ">=": lambda x, y: x >= y,  # type: ignore[operator]
}


def _compare(op: str, a: Value, b: Value) -> Value:
    # Booleans compare numerically with numbers; strings compare
    # case-insensitively (attribute values like "LINUX" vs "Linux").
    if isinstance(a, (bool, int, float)) and isinstance(b, (bool, int, float)):
        return _CMP[op](float(a), float(b))
    if isinstance(a, str) and isinstance(b, str):
        return _CMP[op](a.casefold(), b.casefold())
    return ERROR


def _subscript(base: Value, index: Value) -> Value:
    if base is ERROR or index is ERROR:
        return ERROR
    if base is UNDEFINED or index is UNDEFINED:
        return UNDEFINED
    if isinstance(base, list) and isinstance(index, int) and not isinstance(index, bool):
        return base[index] if 0 <= index < len(base) else ERROR
    return ERROR


# --------------------------------------------------------------------------
# builtins


def _call(name: str, args: tuple[Expr, ...], ctx: _Context) -> Value:
    # ifThenElse and the is* predicates must see raw UNDEFINED/ERROR, so they
    # are special forms rather than strict functions.
    if name == "ifthenelse":
        if len(args) != 3:
            return ERROR
        return _eval(Ternary(*args), ctx)
    values = [_eval(a, ctx) for a in args]
    if name == "isundefined":
        return values[0] is UNDEFINED if len(values) == 1 else ERROR
    if name == "iserror":
        return values[0] is ERROR if len(values) == 1 else ERROR
    fn = _BUILTINS.get(name)
    if fn is None:
        return ERROR
    return fn(values)


def _strict(values: list[Value]) -> Value | None:
    if any(v is ERROR for v in values):
        return ERROR
    if any(v is UNDEFINED for v in values):
        return UNDEFINED
    return None


def _member(values: list[Value]) -> Value:
    if len(values) != 2:
        return ERROR
    item, lst = values
    if item is ERROR or lst is ERROR:
        return ERROR
    if item is UNDEFINED or lst is UNDEFINED:
        return UNDEFINED
    if not isinstance(lst, list) or isinstance(item, list):
        return ERROR
    saw_undefined = False
    for elem in lst:
        r = _compare("==", item, elem) if elem is not UNDEFINED else UNDEFINED
        if r is True:
            return True
        saw_undefined = saw_undefined or r is UNDEFINED
    return UNDEFINED if saw_undefined else False


def _strcat(values: list[Value]) -> Value:
    if (s := _strict(values)) is not None:
        return s
    parts = []
    for v in values:
        if isinstance(v, list):
            return ERROR
        parts.append(v if isinstance(v, str) else format_value(v))
    return "".join(parts)  # type: ignore[arg-type]


def _size(values: list[Value]) -> Value:
    if len(values) != 1:
        return ERROR
    if (s := _strict(values)) is not None:
        return s
    v = values[0]
    return len(v) if isinstance(v, (str, list)) else ERROR


@lru_cache(maxsize=256)
def _compile(pattern: str, flags: int) -> re.Pattern[str] | None:
    try:
        return re.compile(pattern, flags)
    except re.error:
        return None


def _regexp(values: list[Value]) -> Value:
    if len(values) not in (2, 3):
        return ERROR
    if (s := _strict(values)) is not None:
        return s
    if not all(isinstance(v, str) for v in values):
        return ERROR
    pattern, target = values[0], values[1]
    options = values[2] if len(values) == 3 else ""
    flags = re.IGNORECASE if "i" in options else 0  # type: ignore[operator]
    rx = _compile(pattern, flags)  # type: ignore[arg-type]
    if rx is None:
        return ERROR
    return rx.search(target) is not None  # type: ignore[arg-type]


def _case(fn: Callable[[str], str]) -> Callable[[list[Value]], Value]:
    def impl(values: list[Value]) -> Value:
        if len(values) != 1:
            return ERROR
        if (s := _strict(values)) is not None:
            return s
        return fn(values[0]) if isinstance(values[0], str) else ERROR

    return impl


_BUILTINS: dict[str, Callable[[list[Value]], Value]] = {
    "member": _member,
    "strcat": _strcat,
    "size": _size,
    "regexp": _regexp,
    "tolower": _case(str.lower),
    "toupper": _case(str.upper),
}

BUILTIN_NAMES = sorted([*_BUILTINS, "ifthenelse", "isundefined", "iserror"])
