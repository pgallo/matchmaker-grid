"""Synthetic text corpus and small helpers used by the map-reduce DAG.

The text is generated from a fixed vocabulary with Zipf-like weights and a
fixed seed, so word counts are reproducible and testable.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

VOCAB = (
    "the of and to job machine slot match rank pool queue user share priority claim worker cycle ad "
    "requirement expression graph node retry checkpoint result shard reduce map scheduler cluster "
    "resource policy idle busy"
).split()


def make_text(seed: int, words: int) -> str:
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(len(VOCAB))]
    chosen = rng.choices(VOCAB, weights=weights, k=words)
    lines = [" ".join(chosen[i : i + 12]) for i in range(0, words, 12)]
    return "\n".join(lines) + "\n"


def write_shards(directory: str, shards: int, words: int = 4000) -> dict[str, object]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, shards + 1):
        (out / f"shard{i}.txt").write_text(make_text(seed=i, words=words))
    return {"shards": shards, "words_per_shard": words}


def summarize(path: str) -> dict[str, object]:
    data = json.loads(Path(path).read_text())
    return {"total": data["total"], "top3": [w for w, _ in data["top"][:3]]}


def main(argv: list[str]) -> int:
    """Tiny script entry for PRE/POST hooks: ``require FILE`` or ``verify FILE RETURN``."""
    cmd, *rest = argv
    if cmd == "require":
        return 0 if Path(rest[0]).exists() else 1
    if cmd == "verify":
        path, job_exit = rest[0], int(rest[1])
        if job_exit != 0 or not Path(path).exists():
            return 1
        data = json.loads(Path(path).read_text())
        return 0 if data.get("total", 0) > 0 and data.get("top") else 2
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
