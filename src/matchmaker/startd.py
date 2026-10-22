"""The startd: a worker process that owns N static slots and runs jobs on them.

Run as ``python -m matchmaker.startd --name w1 --collector 127.0.0.1:9618 ...``.

* Advertises one ad per slot to the collector every ``--advertise`` seconds
  over a persistent connection (soft state: stop advertising and you vanish).
* Listens on its own TCP port for activations from the schedd. Each
  activation is one connection: ``activate`` -> ``accepted`` -> ``heartbeat``*
  -> ``done``. The schedd may send ``vacate`` at any time to preempt.
* Re-checks the symmetric match on activation, because the negotiator decided
  on an ad that may be a few hundred milliseconds old.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import platform
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

from . import protocol
from .ads import Ad, parse_expr
from .matchmaking import symmetric_match

MAX_OUTPUT = 64 * 1024


@dataclass
class SlotState:
    name: str
    ad: Ad
    job_id: str | None = None
    user: str | None = None
    proc: asyncio.subprocess.Process | None = None


class Startd:
    def __init__(
        self,
        name: str,
        collector: str,
        slots: int,
        memory: int,
        attrs: dict[str, str],
        advertise: float,
        heartbeat: float,
    ) -> None:
        self.name = name
        self.collector = collector
        self.advertise_period = advertise
        self.heartbeat_period = heartbeat
        self.address = ""
        self.slots: dict[str, SlotState] = {}
        for i in range(1, slots + 1):
            ad = Ad(
                {
                    "MyType": "Machine",
                    "Name": f"slot{i}@{name}",
                    "Machine": name,
                    "SlotId": i,
                    "Cpus": 1,
                    "TotalCpus": slots,
                    "Memory": memory // slots,
                    "OpSys": platform.system().upper() or "LINUX",
                    "Arch": platform.machine().upper() or "X86_64",
                    "StartdPid": os.getpid(),
                }
            )
            for k, src in attrs.items():
                ad[k] = parse_expr(src)
            self.slots[ad.get("Name")] = SlotState(str(ad.get("Name")), ad)  # type: ignore[index]

    def slot_ad(self, slot: SlotState) -> Ad:
        ad = slot.ad.copy()
        ad["MyAddress"] = self.address
        ad["State"] = "Claimed" if slot.job_id else "Unclaimed"
        ad["Activity"] = "Busy" if slot.job_id else "Idle"
        if slot.user:
            ad["RemoteUser"] = slot.user
            ad["JobId"] = slot.job_id
        ad["LoadAvg"] = round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else 0.0
        return ad

    async def run(self) -> None:
        server = await asyncio.start_server(self._activation, "127.0.0.1", 0, limit=protocol.LIMIT)
        host, port = server.sockets[0].getsockname()[:2]
        self.address = f"{host}:{port}"
        async with server:
            await self._advertise_forever()

    async def _advertise_forever(self) -> None:
        _, writer = await protocol.connect(self.collector)
        try:
            while True:
                for slot in self.slots.values():
                    protocol.send_nowait(writer, {"type": "advertise", "ad": self.slot_ad(slot).to_json()})
                await writer.drain()
                await asyncio.sleep(self.advertise_period)
        finally:
            await protocol.close(writer)

    async def _activation(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            msg = await protocol.recv(reader, timeout=5.0)
            if not msg or msg.get("type") != "activate":
                return
            slot = self.slots.get(msg["slot"])
            job = Ad.from_json(msg["ad"])
            reason = None
            if slot is None:
                reason = "no such slot"
            elif slot.job_id is not None:
                reason = f"slot busy with {slot.job_id}"
            elif not symmetric_match(job, self.slot_ad(slot)):
                reason = "match no longer holds against current machine ad"
            if reason or slot is None:
                await protocol.send(writer, {"type": "rejected", "reason": reason})
                return
            slot.job_id, slot.user = msg["job_id"], msg["user"]
            await protocol.send(writer, {"type": "accepted", "slot": slot.name})
            await self._execute(slot, msg, reader, writer)
        except (ConnectionError, TimeoutError, ValueError):
            pass
        finally:
            await protocol.close(writer)

    async def _execute(
        self,
        slot: SlotState,
        msg: dict[str, Any],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        args = [sys.executable if a == "python" else a for a in msg["args"]]
        env = {**os.environ, **msg.get("env", {}), "MATCHMAKER_SLOT": slot.name}
        job_id = slot.job_id
        started = time.monotonic()
        vacated = False
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=msg.get("cwd") or None,
            )
        except OSError as exc:
            self._release(slot, job_id)
            await protocol.send(
                writer, {"type": "done", "job_id": job_id, "exit_code": 127, "stdout": "", "stderr": str(exc)}
            )
            return
        slot.proc = proc
        communicate = asyncio.create_task(proc.communicate())
        control = asyncio.create_task(protocol.recv(reader))
        try:
            while not communicate.done():
                done, _ = await asyncio.wait({communicate, control}, timeout=self.heartbeat_period)
                if control in done:
                    cmd = control.result()
                    if cmd is None or cmd.get("type") == "vacate":
                        # Schedd asked us to stop (or went away): terminate politely,
                        # give the job a moment to flush a checkpoint, then kill.
                        vacated = True
                        with contextlib.suppress(ProcessLookupError):
                            proc.terminate()
                        try:
                            await asyncio.wait_for(asyncio.shield(communicate), 1.0)
                        except TimeoutError:
                            with contextlib.suppress(ProcessLookupError):
                                proc.kill()
                        break
                    control = asyncio.create_task(protocol.recv(reader))
                if not communicate.done():
                    await protocol.send(writer, {"type": "heartbeat", "job_id": job_id})
            out, err = await communicate
            result = {
                "type": "done",
                "job_id": job_id,
                "exit_code": proc.returncode,
                "vacated": vacated,
                "stdout": out.decode(errors="replace")[-MAX_OUTPUT:],
                "stderr": err.decode(errors="replace")[-MAX_OUTPUT:],
                "runtime": round(time.monotonic() - started, 3),
            }
            # Free the slot before reporting, so a follow-up activation that the
            # negotiator issues as soon as it hears "done" finds the slot idle.
            self._release(slot, job_id)
            await protocol.send(writer, result)
        finally:
            control.cancel()
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            self._release(slot, job_id)

    @staticmethod
    def _release(slot: SlotState, job_id: str | None) -> None:
        if slot.job_id == job_id:
            slot.job_id = slot.user = slot.proc = None


def _parse_attr(text: str) -> tuple[str, str]:
    name, _, expr = text.partition("=")
    if not name.strip() or not expr.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=EXPR, got {text!r}")
    return name.strip(), expr.strip()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="matchmaker-startd")
    p.add_argument("--name", required=True)
    p.add_argument("--collector", required=True)
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--memory", type=int, default=4096)
    p.add_argument("--attr", type=_parse_attr, action="append", default=[])
    p.add_argument("--advertise", type=float, default=0.25)
    p.add_argument("--heartbeat", type=float, default=0.2)
    a = p.parse_args(argv)
    startd = Startd(a.name, a.collector, a.slots, a.memory, dict(a.attr), a.advertise, a.heartbeat)
    loop = asyncio.new_event_loop()
    task = loop.create_task(startd.run())
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    with contextlib.suppress(asyncio.CancelledError, ConnectionError):
        loop.run_until_complete(task)


if __name__ == "__main__":
    main()
