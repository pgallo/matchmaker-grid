"""The schedd: the job queue and the submit-side half of every claim.

A job moves through IDLE -> RUNNING -> {COMPLETED, FAILED}. Three things
send a RUNNING job back to IDLE without charging a retry:

* the worker is lost (no heartbeat for ``heartbeat_timeout`` seconds, or the
  activation connection drops without a ``done`` message);
* the claim is preempted by the negotiator (``vacate``);
* the startd rejects the activation because its ad changed.

A non-zero exit code is a real failure: it is retried up to ``max_retries``
times, then the job is FAILED. Checkpointing jobs get a stable checkpoint
path in their environment so a rerun picks up where the last one stopped.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from . import protocol
from .ads import Ad, parse_expr, unparse
from .ads.ast import Binary
from .negotiator import Decision, IdleJob


class JobStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REMOVED = "removed"


TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.REMOVED}


@dataclass
class Job:
    job_id: str
    owner: str
    ad: Ad
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    max_retries: int = 0
    status: JobStatus = JobStatus.IDLE
    failures: int = 0
    runs: int = 0
    lost: int = 0
    preempted: int = 0
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    slot: str | None = None
    history: list[str] = field(default_factory=list)
    submitted_at: float = field(default_factory=time.monotonic)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def finish(self, status: JobStatus) -> None:
        self.status = status
        self.done.set()


class Schedd:
    def __init__(
        self,
        workdir: Path,
        heartbeat_timeout: float = 1.5,
        suspect_for: float = 3.0,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.workdir = workdir
        self.heartbeat_timeout = heartbeat_timeout
        self.suspect_for = suspect_for
        self.jobs: dict[str, Job] = {}
        self.claims: dict[str, str] = {}  # slot name -> job id
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._suspect: dict[str, float] = {}  # machine -> until
        self._next = 1
        self._emit = on_event or (lambda _m: None)

    # -- submission ------------------------------------------------------
    def submit(
        self,
        owner: str,
        args: list[str],
        *,
        requirements: str | None = None,
        rank: str | None = None,
        request_cpus: int = 1,
        request_memory: int = 256,
        max_retries: int = 0,
        checkpoint: bool = False,
        attrs: dict[str, object] | None = None,
        env: dict[str, str] | None = None,
    ) -> Job:
        job_id = f"{self._next}.0"
        self._next += 1
        ad = Ad(
            {
                "MyType": "Job",
                "Owner": owner,
                "ClusterId": int(job_id.split(".")[0]),
                "RequestCpus": request_cpus,
                "RequestMemory": request_memory,
                "Cmd": args[0] if args else "",
                **(attrs or {}),
            }
        )
        base = parse_expr("TARGET.Cpus >= MY.RequestCpus && TARGET.Memory >= MY.RequestMemory")
        ad["Requirements"] = Binary("&&", base, parse_expr(requirements)) if requirements else base
        if rank:
            ad["Rank"] = parse_expr(rank)
        job_env = dict(env or {})
        if checkpoint:
            ad["WantCheckpoint"] = True
            job_env["MATCHMAKER_CHECKPOINT"] = str(self.workdir / f"job{job_id}.ckpt")
        job = Job(job_id, owner, ad, args, job_env, max_retries)
        self.jobs[job_id] = job
        return job

    def submit_callable(self, owner: str, target: str, *fn_args: object, **kw: object) -> Job:
        """Submit ``module:function(*fn_args)`` as a job (runs via the workloads shim)."""
        import json

        args = ["python", "-m", "matchmaker.workloads", "call", target, "--args", json.dumps(list(fn_args))]
        return self.submit(owner, args, **kw)  # type: ignore[arg-type]

    def remove(self, job_id: str, reason: str) -> None:
        job = self.jobs[job_id]
        if job.status is JobStatus.IDLE:
            job.history.append(f"removed: {reason}")
            job.finish(JobStatus.REMOVED)

    # -- views for the negotiator ---------------------------------------
    def idle_jobs(self) -> list[IdleJob]:
        return [
            IdleJob(j.job_id, j.owner, j.ad)
            for j in sorted(self.jobs.values(), key=lambda j: int(j.job_id.split(".")[0]))
            if j.status is JobStatus.IDLE
        ]

    def is_suspect(self, machine: str) -> bool:
        until = self._suspect.get(machine)
        return until is not None and time.monotonic() < until

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in JobStatus}
        for j in self.jobs.values():
            out[j.status.value] += 1
        return out

    def all_done(self) -> bool:
        return all(j.status in TERMINAL for j in self.jobs.values())

    # -- claims ----------------------------------------------------------
    def start(self, decision: Decision, address: str, machine: str) -> None:
        job = self.jobs[decision.job_id]
        if job.status is not JobStatus.IDLE:
            return
        job.status = JobStatus.RUNNING
        job.slot = decision.slot
        self.claims[decision.slot] = job.job_id
        victim = self._tasks.get(decision.preempts) if decision.preempts else None
        if decision.preempts:
            self.vacate(decision.preempts)
        self._tasks[job.job_id] = asyncio.create_task(self._run(job, decision.slot, address, machine, victim))

    def vacate(self, job_id: str) -> None:
        writer = self._writers.get(job_id)
        if writer is not None and not writer.is_closing():
            protocol.send_nowait(writer, {"type": "vacate", "job_id": job_id})

    async def _run(self, job: Job, slot: str, address: str, machine: str, wait_for: asyncio.Task[None] | None) -> None:
        if wait_for is not None:
            # The slot is still busy with the job being evicted; it frees up once
            # the startd confirms the vacate with a "done" message.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(wait_for), 5.0)
        job.runs += 1
        writer: asyncio.StreamWriter | None = None
        outcome = "lost"
        try:
            reader, writer = await protocol.connect(address)
            await protocol.send(
                writer,
                {
                    "type": "activate",
                    "slot": slot,
                    "job_id": job.job_id,
                    "user": job.owner,
                    "ad": job.ad.to_json(),
                    "args": job.args,
                    "env": job.env,
                    "cwd": str(self.workdir),
                },
            )
            reply = await protocol.recv(reader, timeout=self.heartbeat_timeout)
            if reply is None or reply.get("type") != "accepted":
                reason = (reply or {}).get("reason", "no reply")
                job.history.append(f"activation on {slot} rejected: {reason}")
                job.status = JobStatus.IDLE
                outcome = "rejected"
                return
            self._writers[job.job_id] = writer
            job.history.append(f"running on {slot}")
            while True:
                msg = await protocol.recv(reader, timeout=self.heartbeat_timeout)
                if msg is None:
                    raise ConnectionError("connection closed before job finished")
                if msg.get("type") == "done":
                    outcome = "done"
                    self._finish(job, slot, msg)
                    return
        except TimeoutError:
            self._lost(job, slot, machine, f"no heartbeat for {self.heartbeat_timeout:g}s")
        except (ConnectionError, OSError) as exc:
            self._lost(job, slot, machine, f"connection lost ({exc.__class__.__name__})")
        finally:
            if self.claims.get(slot) == job.job_id:
                del self.claims[slot]
            self._writers.pop(job.job_id, None)
            if writer is not None:
                if outcome == "lost":
                    writer.transport.abort()
                else:
                    await protocol.close(writer)

    def _lost(self, job: Job, slot: str, machine: str, why: str) -> None:
        job.lost += 1
        job.status = JobStatus.IDLE
        job.history.append(f"lost on {slot}: {why}")
        self._suspect[machine] = time.monotonic() + self.suspect_for
        self._emit(f"schedd: job {job.job_id} ({job.owner}) lost on {slot}: {why}; rescheduling")

    def _finish(self, job: Job, slot: str, msg: dict[str, object]) -> None:
        code = int(msg.get("exit_code") or 0)
        job.stdout = str(msg.get("stdout", ""))
        job.stderr = str(msg.get("stderr", ""))
        if msg.get("vacated"):
            job.preempted += 1
            job.status = JobStatus.IDLE
            job.history.append(f"vacated from {slot} (preempted)")
            self._emit(f"schedd: job {job.job_id} ({job.owner}) vacated from {slot}, back to idle")
            return
        job.exit_code = code
        if code == 0:
            job.history.append(f"completed on {slot}")
            job.finish(JobStatus.COMPLETED)
            return
        job.failures += 1
        if job.failures <= job.max_retries:
            job.status = JobStatus.IDLE
            job.history.append(f"exit {code} on {slot}; retry {job.failures}/{job.max_retries}")
            self._emit(f"schedd: job {job.job_id} ({job.owner}) exited {code}; retry {job.failures}/{job.max_retries}")
        else:
            job.history.append(f"exit {code} on {slot}; no retries left")
            self._emit(f"schedd: job {job.job_id} ({job.owner}) failed with exit {code}")
            job.finish(JobStatus.FAILED)

    async def shutdown(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        for t in self._tasks.values():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

    def describe(self, job_id: str) -> str:
        j = self.jobs[job_id]
        return f"{j.job_id} {j.owner} {j.status.value} req={unparse(j.ad.lookup('Requirements'))}"  # type: ignore[arg-type]
