"""The collector: a TCP directory of machine ads with soft-state expiry.

Startds keep a connection open and re-send their ad every ``advertise``
period. The ad is soft state: if no update arrives within ``ttl`` seconds the
machine is dropped. This is the same signal the schedd uses on the claim
connection, seen from the directory side.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from . import protocol
from .ads import Ad, is_true, parse_expr
from .ads.evaluator import evaluate


@dataclass
class _Entry:
    ad: Ad
    last_seen: float


class Collector:
    def __init__(self, ttl: float = 2.0, on_event: Callable[[str], None] | None = None) -> None:
        self.ttl = ttl
        self._ads: dict[str, _Entry] = {}
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self._on_event = on_event or (lambda _msg: None)
        self.address = ""

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        self._server = await asyncio.start_server(self._handle, host, port, limit=protocol.LIMIT)
        sock = self._server.sockets[0].getsockname()
        self.address = f"{sock[0]}:{sock[1]}"
        return self.address

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            for w in list(self._writers):
                w.close()
            await self._server.wait_closed()

    def machines(self, constraint: str | None = None) -> list[Ad]:
        self._expire()
        ads = [e.ad for e in self._ads.values()]
        if constraint:
            expr = parse_expr(constraint)
            ads = [a for a in ads if is_true(evaluate(expr, a))]
        return sorted(ads, key=lambda a: str(a.get("Name", "")))

    def _expire(self) -> None:
        now = time.monotonic()
        for name, e in list(self._ads.items()):
            if now - e.last_seen > self.ttl:
                del self._ads[name]
                self._on_event(f"collector: ad for {name} expired (no update for {self.ttl:g}s)")

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.add(writer)
        try:
            while (msg := await protocol.recv(reader)) is not None:
                kind = msg.get("type")
                if kind == "advertise":
                    ad = Ad.from_json(msg["ad"])
                    name = str(ad.get("Name"))
                    if name not in self._ads:
                        self._on_event(f"collector: new machine {name}")
                    self._ads[name] = _Entry(ad, time.monotonic())
                elif kind == "invalidate":
                    self._ads.pop(msg["name"], None)
                elif kind == "query":
                    ads = self.machines(msg.get("constraint"))
                    await protocol.send(writer, {"type": "ads", "ads": [a.to_json() for a in ads]})
        except (ConnectionError, ValueError):
            pass
        finally:
            self._writers.discard(writer)
            await protocol.close(writer)


async def query(address: str, constraint: str | None = None) -> list[Ad]:
    """Client side of the collector query protocol."""
    reader, writer = await protocol.connect(address)
    try:
        await protocol.send(writer, {"type": "query", "constraint": constraint})
        reply = await protocol.recv(reader, timeout=5.0)
        return [Ad.from_json(a) for a in (reply or {}).get("ads", [])]
    finally:
        await protocol.close(writer)
