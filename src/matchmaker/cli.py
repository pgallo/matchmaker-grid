"""Command-line entry points: ``matchmaker`` (expression tools) and ``matchmaker-grid`` (pool)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ads import Ad, AdSyntaxError, evaluate, format_value, parse_expr, tokenize, unparse
from .matchmaking import analyze, rank, symmetric_match


def _load_ad(text: str | None) -> Ad | None:
    if text is None:
        return None
    if text.startswith("@"):
        text = Path(text[1:]).read_text()
    return Ad.parse(text)


def matchmaker_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="matchmaker", description="Resource-ad expression tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="evaluate an expression")
    e.add_argument("expr")
    e.add_argument("--my", help="ad for MY, e.g. '[Memory = 4096]' or @file.ad")
    e.add_argument("--target", help="ad for TARGET")
    e.add_argument("--tokens", action="store_true", help="also print tokens")

    f = sub.add_parser("format", help="pretty-print an expression or an ad")
    f.add_argument("text")

    m = sub.add_parser("match", help="symmetric match of a job ad against machine ads")
    m.add_argument("job")
    m.add_argument("machines", nargs="+")

    a = p.parse_args(argv)
    try:
        if a.cmd == "eval":
            expr = parse_expr(a.expr)
            if a.tokens:
                print("tokens:", " ".join(t.text for t in tokenize(a.expr) if t.text))
            print("parsed:", unparse(expr))
            print("value: ", format_value(evaluate(expr, _load_ad(a.my), _load_ad(a.target))))
        elif a.cmd == "format":
            text = Path(a.text[1:]).read_text() if a.text.startswith("@") else a.text
            print(Ad.parse(text) if text.lstrip().startswith("[") else unparse(parse_expr(text)))
        elif a.cmd == "match":
            job = _load_ad(a.job)
            machines = [_load_ad(x) for x in a.machines]
            assert job is not None
            for mach in machines:
                assert mach is not None
                print(
                    f"{mach.get('Name', '?')!s:<16} match={symmetric_match(job, mach)!s:<5} "
                    f"job-rank={rank(job, mach):g} machine-rank={rank(mach, job):g}"
                )
            print(analyze(job, [x for x in machines if x is not None], str(job.get("Name", "job"))).render())
    except AdSyntaxError as exc:
        print(f"syntax error: {exc}", file=sys.stderr)
        return 2
    return 0


def grid_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="matchmaker-grid", description="Local high-throughput pool")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run the end-to-end demo", add_help=False)
    args, rest = p.parse_known_args(argv)
    if args.cmd == "demo":
        from .demo import main as demo_main

        return demo_main(rest)
    return 2


if __name__ == "__main__":
    sys.exit(matchmaker_main())
