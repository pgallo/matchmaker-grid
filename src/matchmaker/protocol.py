"""JSON-lines framing over asyncio streams: one JSON object per ``\\n``-terminated line."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

Message = dict[str, Any]
LIMIT = 4 * 1024 * 1024  # stdout of a job travels in one line


async def send(writer: asyncio.StreamWriter, msg: Message) -> None:
    writer.write(json.dumps(msg, separators=(",", ":")).encode() + b"\n")
    await writer.drain()


def send_nowait(writer: asyncio.StreamWriter, msg: Message) -> None:
    """Write without draining; safe from several tasks because each call is one write."""
    writer.write(json.dumps(msg, separators=(",", ":")).encode() + b"\n")


async def recv(reader: asyncio.StreamReader, timeout: float | None = None) -> Message | None:
    """Next message, or None on EOF. Raises TimeoutError if ``timeout`` elapses."""
    line = await asyncio.wait_for(reader.readline(), timeout) if timeout else await reader.readline()
    if not line:
        return None
    return json.loads(line)


async def connect(address: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    host, port = address.rsplit(":", 1)
    return await asyncio.open_connection(host, int(port), limit=LIMIT)


async def close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(ConnectionError, OSError):
        await writer.wait_closed()
