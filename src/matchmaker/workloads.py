"""Job payloads run by startds: ``python -m matchmaker.workloads <kind> ...``.

Each payload prints one JSON line with its result on stdout, so the schedd
can capture it and DAG nodes can consume each other's output files.

* ``pi``       Monte Carlo estimate of pi over one shard of samples.
* ``longpi``   The same, in many chunks, with checkpoint/resume: progress is
               written to ``$MATCHMAKER_CHECKPOINT`` after every chunk and on
               SIGTERM, and picked up again on the next run.
* ``wc-map``   Word counts for one input shard, written to a JSON file.
* ``wc-reduce``Merge map outputs into a top-N list.
* ``flaky``    Fails until it has been attempted ``--fail-times`` times
               (attempt count kept in a marker file); used to show retries.
* ``call``     Import ``module:function`` and call it with JSON arguments,
               which is how Python callables are submitted as jobs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def _emit(result: dict[str, Any]) -> None:
    print(json.dumps(result, sort_keys=True), flush=True)


def monte_carlo_hits(samples: int, rng: random.Random) -> int:
    hits = 0
    rnd = rng.random
    for _ in range(samples):
        x, y = rnd(), rnd()
        if x * x + y * y <= 1.0:
            hits += 1
    return hits


def cmd_pi(a: argparse.Namespace) -> int:
    # Sample in fixed-size batches; with --duration keep going until the time
    # budget is spent, so shard runtime is stable across fast and slow hosts.
    rng = random.Random(a.seed)
    deadline = time.monotonic() + a.duration
    samples = hits = 0
    while True:
        hits += monte_carlo_hits(a.samples, rng)
        samples += a.samples
        if time.monotonic() >= deadline:
            break
    _emit({"kind": "pi", "seed": a.seed, "samples": samples, "hits": hits, "pi": 4 * hits / samples})
    return 0


def cmd_longpi(a: argparse.Namespace) -> int:
    ckpt = Path(os.environ.get("MATCHMAKER_CHECKPOINT", a.checkpoint or "longpi.ckpt"))
    state = {"chunk": 0, "hits": 0, "resumed_from": 0}
    if ckpt.exists():
        state = json.loads(ckpt.read_text())
        state["resumed_from"] = state["chunk"]

    def save() -> None:
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(ckpt)  # atomic: a kill mid-write never corrupts the checkpoint

    def on_term(_sig: int, _frame: object) -> None:
        save()
        sys.exit(143)

    signal.signal(signal.SIGTERM, on_term)
    while state["chunk"] < a.chunks:
        # Seed per chunk so the result is identical whether or not we resumed.
        rng = random.Random(a.seed * 1_000_003 + state["chunk"])
        state["hits"] += monte_carlo_hits(a.samples, rng)
        state["chunk"] += 1
        save()
        if a.pause:
            time.sleep(a.pause)
    total = a.chunks * a.samples
    _emit(
        {
            "kind": "longpi",
            "samples": total,
            "hits": state["hits"],
            "chunks": a.chunks,
            "resumed_from_chunk": state["resumed_from"],
            "pi": 4 * state["hits"] / total,
        }
    )
    return 0


_WORD = re.compile(r"[a-z']+")


def count_words(text: str) -> Counter[str]:
    return Counter(w.strip("'") for w in _WORD.findall(text.lower()) if w.strip("'"))


def cmd_wc_map(a: argparse.Namespace) -> int:
    counts = count_words(Path(a.input).read_text())
    Path(a.out).write_text(json.dumps(counts, sort_keys=True))
    _emit({"kind": "wc-map", "input": Path(a.input).name, "words": sum(counts.values()), "distinct": len(counts)})
    return 0


def cmd_wc_reduce(a: argparse.Namespace) -> int:
    total: Counter[str] = Counter()
    for path in a.inputs:
        total.update(json.loads(Path(path).read_text()))
    top = total.most_common(a.top)
    Path(a.out).write_text(json.dumps({"total": sum(total.values()), "top": top}))
    _emit({"kind": "wc-reduce", "total": sum(total.values()), "top": top})
    return 0


def cmd_flaky(a: argparse.Namespace) -> int:
    marker = Path(a.marker)
    attempts = int(marker.read_text()) + 1 if marker.exists() else 1
    marker.write_text(str(attempts))
    if attempts <= a.fail_times:
        print(f"flaky: attempt {attempts} failing on purpose", file=sys.stderr)
        _emit({"kind": "flaky", "attempt": attempts, "ok": False})
        return 3
    _emit({"kind": "flaky", "attempt": attempts, "ok": True})
    return 0


def cmd_call(a: argparse.Namespace) -> int:
    module, _, attr = a.target.partition(":")
    fn = importlib.import_module(module)
    for part in attr.split("."):
        fn = getattr(fn, part)
    result = fn(*json.loads(a.args), **json.loads(a.kwargs))
    _emit({"kind": "call", "target": a.target, "result": result})
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="matchmaker-workloads")
    sub = p.add_subparsers(dest="kind", required=True)

    s = sub.add_parser("pi")
    s.add_argument("--samples", type=int, default=200_000, help="samples per batch")
    s.add_argument("--duration", type=float, default=0.0, help="minimum seconds of sampling")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_pi)

    s = sub.add_parser("longpi")
    s.add_argument("--chunks", type=int, default=20)
    s.add_argument("--samples", type=int, default=50_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--pause", type=float, default=0.0)
    s.add_argument("--checkpoint")
    s.set_defaults(fn=cmd_longpi)

    s = sub.add_parser("wc-map")
    s.add_argument("--input", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_wc_map)

    s = sub.add_parser("wc-reduce")
    s.add_argument("--out", required=True)
    s.add_argument("--top", type=int, default=5)
    s.add_argument("inputs", nargs="+")
    s.set_defaults(fn=cmd_wc_reduce)

    s = sub.add_parser("flaky")
    s.add_argument("--marker", required=True)
    s.add_argument("--fail-times", type=int, default=1)
    s.set_defaults(fn=cmd_flaky)

    s = sub.add_parser("call")
    s.add_argument("target")
    s.add_argument("--args", default="[]")
    s.add_argument("--kwargs", default="{}")
    s.set_defaults(fn=cmd_call)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
