"""A local pool: in-process collector, schedd and negotiator, startd subprocesses.

The negotiator talks to the collector over TCP like any other client, and the
schedd talks to startds over TCP; only the orchestration lives in one process.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import collector as collector_mod
from .collector import Collector
from .dag import Node
from .fairshare import Accountant
from .negotiator import Decision, Negotiator, NegotiatorConfig, Slot
from .schedd import Job, JobStatus, Schedd


@dataclass
class WorkerSpec:
    name: str
    slots: int = 2
    memory: int = 4096
    attrs: dict[str, str] = field(default_factory=dict)


class Pool:
    def __init__(
        self,
        workdir: Path,
        workers: list[WorkerSpec],
        *,
        half_life: float = 4.0,
        cycle: float = 0.25,
        quotas: dict[str, float] | None = None,
        factors: dict[str, float] | None = None,
        max_preemptions: int = 2,
        heartbeat_timeout: float = 1.5,
        ttl: float = 2.0,
        log: Callable[[str], None] = print,
        on_decision: Callable[[float, Decision], None] | None = None,
    ) -> None:
        self.workdir = workdir
        self.specs = {w.name: w for w in workers}
        self.log = log
        self.t0 = time.monotonic()
        self.collector = Collector(ttl=ttl, on_event=self._event)
        self.schedd = Schedd(workdir, heartbeat_timeout=heartbeat_timeout, suspect_for=ttl + 1.0, on_event=self._event)
        self.accountant = Accountant(half_life)
        for user, f in (factors or {}).items():
            self.accountant.set_factor(user, f)
        self.negotiator = Negotiator(
            self.accountant, NegotiatorConfig(quotas=dict(quotas or {}), max_preemptions_per_cycle=max_preemptions)
        )
        self.cycle_period = cycle
        self.procs: dict[str, asyncio.subprocess.Process] = {}
        self.decisions: list[tuple[float, Decision]] = []
        self.on_decision = on_decision
        self._loop_task: asyncio.Task[None] | None = None
        self.machine_count = 0

    def now(self) -> float:
        return time.monotonic() - self.t0

    def _event(self, msg: str) -> None:
        self.log(f"[{self.now():5.1f}s] {msg}")

    # -- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        await self.collector.start()
        for spec in self.specs.values():
            await self.start_worker(spec.name)
        expected = sum(s.slots for s in self.specs.values())
        await self.wait_for_slots(expected)
        self._loop_task = asyncio.create_task(self._negotiate_forever())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
        await self.schedd.shutdown()
        for name in list(self.procs):
            await self.kill_worker(name, quiet=True)
        await self.collector.stop()

    async def start_worker(self, name: str) -> None:
        spec = self.specs[name]
        src_root = str(Path(__file__).resolve().parent.parent)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src_root, os.environ.get("PYTHONPATH")]))}
        args = [
            sys.executable,
            "-m",
            "matchmaker.startd",
            "--name",
            spec.name,
            "--collector",
            self.collector.address,
            "--slots",
            str(spec.slots),
            "--memory",
            str(spec.memory),
        ]
        for k, v in spec.attrs.items():
            args += ["--attr", f"{k}={v}"]
        # New session: the worker and every job it starts share a process group,
        # so killing the group simulates the whole machine going away.
        self.procs[name] = await asyncio.create_subprocess_exec(*args, env=env, start_new_session=True)

    async def kill_worker(self, name: str, *, quiet: bool = False) -> None:
        proc = self.procs.pop(name, None)
        if proc is None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()
        if not quiet:
            self._event(f"pool: worker {name} killed")

    def freeze_worker(self, name: str) -> None:
        """SIGSTOP the worker's process group: sockets stay open but heartbeats stop."""
        os.killpg(self.procs[name].pid, signal.SIGSTOP)
        self._event(f"pool: worker {name} frozen (SIGSTOP), heartbeats stop")

    async def wait_for_slots(self, n: int, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while len(self.collector.machines()) < n:
            if time.monotonic() > deadline:
                raise TimeoutError(f"only {len(self.collector.machines())}/{n} slots advertised")
            await asyncio.sleep(0.05)

    # -- negotiation -----------------------------------------------------
    async def negotiate_once(self) -> list[Decision]:
        ads = await collector_mod.query(self.collector.address)
        slots: list[Slot] = []
        by_name = {}
        for ad in ads:
            machine = str(ad.get("Machine", ""))
            if self.schedd.is_suspect(machine):
                continue
            name = str(ad.get("Name"))
            job_id = self.schedd.claims.get(name)
            job = self.schedd.jobs[job_id] if job_id else None
            ckpt = job is not None and job.ad.get("WantCheckpoint", False) is True
            slots.append(Slot(name, ad, job_id, job.owner if job else None, ckpt))
            by_name[name] = ad
        self.machine_count = len(slots)
        decisions = self.negotiator.cycle(self.now(), self.schedd.idle_jobs(), slots)
        for d in decisions:
            ad = by_name[d.slot]
            self.decisions.append((self.now(), d))
            if self.on_decision:
                self.on_decision(self.now(), d)
            self.schedd.start(d, str(ad.get("MyAddress")), str(ad.get("Machine")))
        return decisions

    async def _negotiate_forever(self) -> None:
        while True:
            try:
                await self.negotiate_once()
            except (ConnectionError, OSError, TimeoutError) as exc:
                self._event(f"negotiator: cycle skipped ({exc.__class__.__name__})")
            await asyncio.sleep(self.cycle_period)

    # -- helpers ---------------------------------------------------------
    async def wait_jobs(self, jobs: list[Job], timeout: float) -> None:
        await asyncio.wait_for(asyncio.gather(*(j.done.wait() for j in jobs)), timeout)

    def dag_job_runner(self, owner: str) -> Callable[[Node], asyncio.Future[int]]:
        async def run(node: Node) -> int:
            job = self.schedd.submit(owner, node.command, attrs={"DagNode": node.name})
            await job.done.wait()
            if job.status is JobStatus.COMPLETED:
                return 0
            return job.exit_code if job.exit_code is not None else 1

        return run  # type: ignore[return-value]

    def running_by_user(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for job_id in self.schedd.claims.values():
            owner = self.schedd.jobs[job_id].owner
            out[owner] = out.get(owner, 0) + 1
        return out
