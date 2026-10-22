"""Tokenizer, parser, pretty printer and evaluator."""

from __future__ import annotations

import itertools

import pytest

from matchmaker.ads import (
    ERROR,
    UNDEFINED,
    Ad,
    AdSyntaxError,
    evaluate,
    parse_expr,
    tokenize,
    unparse,
)
from matchmaker.ads.ast import AttrRef, Binary, Literal, Ternary


def ev(src: str, my: Ad | None = None, target: Ad | None = None):
    return evaluate(parse_expr(src), my, target)


# -- tokenizer -----------------------------------------------------------


def test_tokenizer_longest_operator_and_literals():
    toks = [t.text for t in tokenize('a =?= b =!= 1.5e3 && "x\\"y" // comment')]
    assert toks == ["a", "=?=", "b", "=!=", "1.5e3", "&&", '"x\\"y"', ""]
    vals = [t.value for t in tokenize('42 .5 "a\\nb"')][:3]
    assert vals == [42, 0.5, "a\nb"]


def test_tokenizer_errors_point_at_offset():
    with pytest.raises(AdSyntaxError) as exc:
        tokenize('x == "open')
    assert exc.value.pos == 5
    with pytest.raises(AdSyntaxError):
        tokenize("x @ y")


# -- parser --------------------------------------------------------------


def test_precedence_and_associativity():
    tree = parse_expr("a || b && c == 1 + 2 * 3")
    assert isinstance(tree, Binary) and tree.op == "||"
    assert unparse(tree) == "a || b && c == 1 + 2 * 3"
    assert ev("10 - 4 - 3") == 3  # left associative
    assert ev("2 + 3 * 4") == 14
    assert ev("(2 + 3) * 4") == 20


def test_scoped_references_and_keywords_are_case_insensitive():
    assert parse_expr("my.Memory") == AttrRef("Memory", "MY")
    assert parse_expr("Target.Cpus") == AttrRef("Cpus", "TARGET")
    assert parse_expr("TRUE") == Literal(True)
    assert parse_expr("Undefined") == Literal(UNDEFINED)


def test_ternary_is_right_associative():
    tree = parse_expr("a ? 1 : b ? 2 : 3")
    assert isinstance(tree, Ternary) and isinstance(tree.orelse, Ternary)
    assert ev("false ? 1 : true ? 2 : 3") == 2


@pytest.mark.parametrize(
    "src",
    [
        "a +",
        "(a",
        "f(a,",
        "MY.",
        "a b",
        "{1, 2",
        "x ? y",
    ],
)
def test_syntax_errors(src):
    with pytest.raises(AdSyntaxError):
        parse_expr(src)


@pytest.mark.parametrize(
    "src",
    [
        'MY.Memory > 2048 && TARGET.OpSys == "LINUX"',
        "(a || b) && c",
        "a - (b - c)",
        "-(a + b) * 2",
        "!(a && b) || !c",
        'x =?= undefined ? "none" : strcat("v", x)',
        "(a ? b : c) ? d : e",
        '{ 1, 2.5, "s", { } }[0] + -3',
        'member(TARGET.Site, { "east", "west" })',
        'regexp("^w[0-9]+$", Name, "i")',
    ],
)
def test_pretty_printer_round_trips(src):
    tree = parse_expr(src)
    printed = unparse(tree)
    assert parse_expr(printed) == tree
    assert unparse(parse_expr(printed)) == printed


def test_pretty_printer_drops_redundant_parens():
    assert unparse(parse_expr("((a)) && ((b || c))")) == "a && (b || c)"
    assert unparse(parse_expr("(a * b) + c")) == "a * b + c"


def test_ad_parse_and_pretty():
    ad = Ad.parse('[ Name = "slot1"; Memory = 4096; Requirements = TARGET.RequestMemory <= MY.Memory ]')
    assert ad.eval("Memory") == 4096
    text = str(ad)
    assert "Requirements = TARGET.RequestMemory <= MY.Memory;" in text
    assert Ad.parse(text).to_json() == ad.to_json()
    assert Ad.from_json(ad.to_json()).to_json() == ad.to_json()


# -- evaluator: arithmetic and comparison --------------------------------


def test_arithmetic_types():
    assert ev("7 / 2") == 3
    assert ev("-7 / 2") == -3  # truncation toward zero
    assert ev("7.0 / 2") == 3.5
    assert ev("-7 % 3") == -1
    assert ev("1 / 0") is ERROR
    assert ev('"a" + 1') is ERROR
    assert ev("true + 1") is ERROR


def test_string_comparison_is_case_insensitive_but_meta_equality_is_not():
    assert ev('"LINUX" == "linux"') is True
    assert ev('"LINUX" =?= "linux"') is False
    assert ev('"abc" < "ABD"') is True
    assert ev('"a" == 1') is ERROR


def test_meta_equality_never_undefined():
    assert ev("undefined =?= undefined") is True
    assert ev("x =?= undefined") is True
    assert ev("x =!= undefined") is False
    assert ev("1 =?= 1.0") is False  # type matters
    assert ev("error =?= error") is True
    assert ev("{1, 2} =?= {1, 2}") is True


def test_unscoped_lookup_prefers_my_then_target():
    my = Ad({"Memory": 1, "Owner": "alice"})
    target = Ad({"Memory": 2, "Cpus": 4})
    assert ev("Memory", my, target) == 1
    assert ev("Cpus", my, target) == 4
    assert ev("TARGET.Memory", my, target) == 2
    assert ev("MY.Cpus", my, target) is UNDEFINED


def test_target_attributes_evaluate_in_their_own_scope():
    job = Ad({"RequestMemory": 512})
    machine = Ad({"Memory": 1024})
    machine.set_expr("Free", "MY.Memory - TARGET.RequestMemory")
    assert ev("TARGET.Free", job, machine) == 512


def test_circular_reference_is_error():
    ad = Ad()
    ad.set_expr("A", "B + 1")
    ad.set_expr("B", "A + 1")
    assert ad.eval("A") is ERROR


def test_lists_and_subscripts():
    assert ev("{1, 2, 3}[1]") == 2
    assert ev("{1, 2, 3}[5]") is ERROR
    assert ev("{1, x}[1]") is UNDEFINED


# -- three-valued logic truth tables --------------------------------------

T, F, U, E = True, False, UNDEFINED, ERROR
LIT = {T: "true", F: "false", U: "undefined", E: "error"}

AND_TABLE = {
    (T, T): T, (T, F): F, (T, U): U, (T, E): E,
    (F, T): F, (F, F): F, (F, U): F, (F, E): F,
    (U, T): U, (U, F): F, (U, U): U, (U, E): E,
    (E, T): E, (E, F): E, (E, U): E, (E, E): E,
}  # fmt: skip
OR_TABLE = {
    (T, T): T, (T, F): T, (T, U): T, (T, E): T,
    (F, T): T, (F, F): F, (F, U): U, (F, E): E,
    (U, T): T, (U, F): U, (U, U): U, (U, E): E,
    (E, T): E, (E, F): E, (E, U): E, (E, E): E,
}  # fmt: skip


@pytest.mark.parametrize(("a", "b"), list(itertools.product([T, F, U, E], repeat=2)))
def test_and_truth_table(a, b):
    assert ev(f"{LIT[a]} && {LIT[b]}") is AND_TABLE[(a, b)]


@pytest.mark.parametrize(("a", "b"), list(itertools.product([T, F, U, E], repeat=2)))
def test_or_truth_table(a, b):
    assert ev(f"{LIT[a]} || {LIT[b]}") is OR_TABLE[(a, b)]


@pytest.mark.parametrize(("a", "expected"), [(T, F), (F, T), (U, U), (E, E)])
def test_not_truth_table(a, expected):
    assert ev(f"!{LIT[a]}") is expected


def test_and_or_short_circuit_on_missing_attributes():
    # A missing attribute is UNDEFINED; a decided connective ignores it.
    assert ev("false && NoSuchAttr > 3") is False
    assert ev("true || NoSuchAttr > 3") is True
    assert ev("true && NoSuchAttr > 3") is UNDEFINED
    assert ev("false || NoSuchAttr > 3") is UNDEFINED


@pytest.mark.parametrize("op", ["+", "-", "*", "/", "<", "<=", "==", "!=", ">", ">="])
def test_strict_operators_propagate_undefined_and_error(op):
    assert ev(f"undefined {op} 1") is UNDEFINED
    assert ev(f"1 {op} undefined") is UNDEFINED
    assert ev(f"error {op} 1") is ERROR
    assert ev(f"error {op} undefined") is ERROR  # ERROR dominates
    assert ev(f"undefined {op} error") is ERROR


def test_ternary_with_undefined_condition():
    assert ev("undefined ? 1 : 2") is UNDEFINED
    assert ev("error ? 1 : 2") is ERROR
    assert ev('"s" ? 1 : 2') is ERROR
    assert ev("0 ? 1 : 2") == 2


def test_sentinels_refuse_python_truthiness():
    with pytest.raises(TypeError):
        bool(UNDEFINED)


# -- builtins ------------------------------------------------------------


def test_member():
    assert ev('member("east", {"west", "EAST"})') is True
    assert ev("member(4, {1, 2, 3})") is False
    assert ev("member(4, {1, undefined})") is UNDEFINED
    assert ev("member(1, {1, undefined})") is True
    assert ev("member(1, 2)") is ERROR
    assert ev("member(x, {1})") is UNDEFINED


def test_strcat_size_regexp():
    assert ev('strcat("slot", 1, "@", "w", 2)') == "slot1@w2"
    assert ev('strcat("a", undefined)') is UNDEFINED
    assert ev('size("hello")') == 5
    assert ev("size({1, 2, 3})") == 3
    assert ev("size(3)") is ERROR
    assert ev('regexp("^slot[0-9]+@", "slot12@w1")') is True
    assert ev('regexp("LINUX", "linux")') is False
    assert ev('regexp("LINUX", "linux", "i")') is True
    assert ev('regexp("(", "x")') is ERROR


def test_ifthenelse_and_isundefined_are_non_strict():
    assert ev("ifThenElse(true, 1, error)") == 1
    assert ev("ifThenElse(undefined, 1, 2)") is UNDEFINED
    assert ev("isUndefined(NoSuchAttr)") is True
    assert ev("isUndefined(1)") is False
    assert ev("isError(1 / 0)") is True
    assert ev("unknownFunction(1)") is ERROR
