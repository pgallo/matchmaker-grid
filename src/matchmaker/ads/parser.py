"""Recursive-descent parser for resource-ad expressions and ads.

Grammar (lowest to highest precedence)::

    expr      := or ( "?" expr ":" expr )?          right associative
    or        := and ( "||" and )*
    and       := equality ( "&&" equality )*
    equality  := relation ( ("==" | "!=" | "=?=" | "=!=") relation )*
    relation  := additive ( ("<" | "<=" | ">" | ">=") additive )*
    additive  := term ( ("+" | "-") term )*
    term      := unary ( ("*" | "/" | "%") unary )*
    unary     := ("!" | "-" | "+") unary | postfix
    postfix   := primary ( "[" expr "]" )*
    primary   := INT | REAL | STRING | true | false | undefined | error
               | IDENT "(" args ")" | (MY|TARGET) "." IDENT | IDENT
               | "(" expr ")" | "{" args "}"

An ad is ``[ Name = expr ; Name = expr ; ... ]``.
"""

from __future__ import annotations

from .ast import AttrRef, Binary, Call, Expr, ListExpr, Literal, Subscript, Ternary, Unary
from .lexer import AdSyntaxError, Tok, Token, tokenize
from .values import ERROR, UNDEFINED

_KEYWORDS = {"true": True, "false": False, "undefined": UNDEFINED, "error": ERROR}
_SCOPES = {"my": "MY", "target": "TARGET"}

# Binary precedence levels, used both by the parser and the pretty printer.
BINARY_LEVELS: list[tuple[str, ...]] = [
    ("||",),
    ("&&",),
    ("==", "!=", "=?=", "=!="),
    ("<", "<=", ">", ">="),
    ("+", "-"),
    ("*", "/", "%"),
]
PRECEDENCE = {op: i + 1 for i, ops in enumerate(BINARY_LEVELS) for op in ops}


class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.tokens = tokenize(source)
        self.i = 0

    # -- token helpers ---------------------------------------------------
    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def at_op(self, *ops: str) -> bool:
        return self.tok.kind is Tok.OP and self.tok.text in ops

    def expect_op(self, op: str) -> Token:
        if not self.at_op(op):
            self.fail(f"expected {op!r}, found {self.tok.text or 'end of input'!r}")
        return self.advance()

    def fail(self, message: str) -> None:
        raise AdSyntaxError(message, self.source, self.tok.pos)

    # -- grammar ---------------------------------------------------------
    def expr(self) -> Expr:
        cond = self.binary(0)
        if self.at_op("?"):
            self.advance()
            then = self.expr()
            self.expect_op(":")
            orelse = self.expr()
            return Ternary(cond, then, orelse)
        return cond

    def binary(self, level: int) -> Expr:
        if level == len(BINARY_LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        while self.at_op(*BINARY_LEVELS[level]):
            op = self.advance().text
            right = self.binary(level + 1)
            left = Binary(op, left, right)
        return left

    def unary(self) -> Expr:
        if self.at_op("!", "-", "+"):
            op = self.advance().text
            operand = self.unary()
            # Fold negative numeric literals so "-3" prints and evaluates as a literal.
            if op == "-" and isinstance(operand, Literal) and type(operand.value) in (int, float):
                return Literal(-operand.value)  # type: ignore[operator]
            return Unary(op, operand)
        return self.postfix()

    def postfix(self) -> Expr:
        node = self.primary()
        while self.at_op("["):
            self.advance()
            index = self.expr()
            self.expect_op("]")
            node = Subscript(node, index)
        return node

    def primary(self) -> Expr:
        t = self.tok
        if t.kind in (Tok.INT, Tok.REAL, Tok.STRING):
            self.advance()
            return Literal(t.value)  # type: ignore[arg-type]
        if t.kind is Tok.IDENT:
            self.advance()
            lower = t.text.lower()
            if lower in _KEYWORDS:
                return Literal(_KEYWORDS[lower])
            if self.at_op("("):
                self.advance()
                return Call(lower, tuple(self.arguments(")")))
            if lower in _SCOPES and self.at_op("."):
                self.advance()
                name = self.tok
                if name.kind is not Tok.IDENT:
                    self.fail("expected attribute name after scope")
                self.advance()
                return AttrRef(name.text, _SCOPES[lower])
            return AttrRef(t.text)
        if self.at_op("("):
            self.advance()
            inner = self.expr()
            self.expect_op(")")
            return inner
        if self.at_op("{"):
            self.advance()
            return ListExpr(tuple(self.arguments("}")))
        self.fail(f"unexpected token {t.text or 'end of input'!r}")
        raise AssertionError("unreachable")

    def arguments(self, closer: str) -> list[Expr]:
        args: list[Expr] = []
        if self.at_op(closer):
            self.advance()
            return args
        while True:
            args.append(self.expr())
            if self.at_op(","):
                self.advance()
                continue
            self.expect_op(closer)
            return args

    def ad_body(self) -> list[tuple[str, Expr]]:
        self.expect_op("[")
        attrs: list[tuple[str, Expr]] = []
        while not self.at_op("]"):
            name = self.tok
            if name.kind is not Tok.IDENT:
                self.fail("expected attribute name")
            self.advance()
            self.expect_op("=")
            attrs.append((name.text, self.expr()))
            if self.at_op(";"):
                self.advance()
            elif not self.at_op("]"):
                self.fail("expected ';' or ']'")
        self.advance()
        return attrs

    def end(self) -> None:
        if self.tok.kind is not Tok.EOF:
            self.fail(f"unexpected trailing input {self.tok.text!r}")


def parse_expr(source: str) -> Expr:
    p = _Parser(source)
    node = p.expr()
    p.end()
    return node


def parse_ad_attrs(source: str) -> list[tuple[str, Expr]]:
    p = _Parser(source)
    attrs = p.ad_body()
    p.end()
    return attrs
