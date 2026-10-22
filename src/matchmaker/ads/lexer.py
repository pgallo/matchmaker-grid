"""Tokenizer for the resource-ad language."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Tok(Enum):
    INT = auto()
    REAL = auto()
    STRING = auto()
    IDENT = auto()
    OP = auto()
    EOF = auto()


@dataclass(frozen=True, slots=True)
class Token:
    kind: Tok
    text: str
    pos: int
    value: object = None


class AdSyntaxError(ValueError):
    def __init__(self, message: str, source: str, pos: int) -> None:
        caret = " " * pos + "^"
        super().__init__(f"{message} at offset {pos}\n  {source}\n  {caret}")
        self.pos = pos


# Longest operators first so "=?=" wins over "=" and "<=" over "<".
_OPERATORS = (
    "=?=",
    "=!=",
    "==",
    "!=",
    "<=",
    ">=",
    "&&",
    "||",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "%",
    "!",
    "?",
    ":",
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    ",",
    ".",
    ";",
    "=",
)
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "'": "'"}


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c.isspace():
            i += 1
            continue
        if c == "#" or source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        start = i
        if c.isdigit() or (c == "." and i + 1 < n and source[i + 1].isdigit()):
            i = _scan_number(source, i)
            text = source[start:i]
            if any(ch in text for ch in ".eE"):
                tokens.append(Token(Tok.REAL, text, start, float(text)))
            else:
                tokens.append(Token(Tok.INT, text, start, int(text)))
            continue
        if c.isalpha() or c == "_":
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            tokens.append(Token(Tok.IDENT, source[start:i], start))
            continue
        if c == '"':
            i += 1
            chars: list[str] = []
            while True:
                if i >= n:
                    raise AdSyntaxError("unterminated string", source, start)
                ch = source[i]
                if ch == '"':
                    i += 1
                    break
                if ch == "\\":
                    if i + 1 >= n or source[i + 1] not in _ESCAPES:
                        raise AdSyntaxError("bad escape sequence", source, i)
                    chars.append(_ESCAPES[source[i + 1]])
                    i += 2
                    continue
                chars.append(ch)
                i += 1
            tokens.append(Token(Tok.STRING, source[start:i], start, "".join(chars)))
            continue
        for op in _OPERATORS:
            if source.startswith(op, i):
                tokens.append(Token(Tok.OP, op, start))
                i += len(op)
                break
        else:
            raise AdSyntaxError(f"unexpected character {c!r}", source, i)
    tokens.append(Token(Tok.EOF, "", n))
    return tokens


def _scan_number(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i].isdigit():
        i += 1
    if i < n and s[i] == "." and i + 1 < n and s[i + 1].isdigit():
        i += 1
        while i < n and s[i].isdigit():
            i += 1
    if i < n and s[i] in "eE":
        j = i + 1
        if j < n and s[j] in "+-":
            j += 1
        if j < n and s[j].isdigit():
            i = j
            while i < n and s[i].isdigit():
                i += 1
    return i
