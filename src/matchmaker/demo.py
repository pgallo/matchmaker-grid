"""End-to-end demo: a 4-worker local pool, ~40 jobs from 3 users, and a map-reduce DAG.

Timeline:

1. Start the collector, negotiator and 4 startd processes (8 slots in all).
2. alice submits a burst (long checkpointing jobs plus pi shards) and fills the pool.
3. bob arrives (quota 4 cores, GPU jobs, a flaky job with retries, and one job
   no machine can run, which is diagnosed and removed).
4. carol arrives with a better priority; machines whose policy allows it
   preempt alice's claims. alice's checkpointing jobs resume where they stopped.
5. The worker running a checkpointing job is frozen: heartbeats stop, the
   schedd declares the job lost and it is rescheduled elsewhere, then resumes.
6. carol runs a map-reduce DAG. Its last node is gated by a PRE script that
   fails, so the DAG writes a rescue file; the demo fixes the gate and resumes
   the DAG from the rescue file, re-running only the missing node.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .ads import Ad
from .dag import Dag, DagRunner, latest_rescue, run_local_script
from .matchmaking import analyze
from .negotiator import Decision
from .pool import Pool, WorkerSpec
from .schedd import Job, JobStatus

PREEMPT_POLICY = "MY.RemoteUserPrio > TARGET.SubmitterUserPrio * 1.2"


@dataclass
class Scale:
    pi_seconds: float
    long_chunks: int
    long_pause: float
    words: int
    timeout: float


SCALES = {
    "full": Scale(pi_seconds=2.0, long_chunks=24, long_pause=0.3, words=6000, timeout=180.0),
    "quick": Scale(pi_seconds=1.0, long_chunks=16, long_pause=0.25, words=1500, timeout=90.0),
}


def workers() -> list[WorkerSpec]:
    return [
        WorkerSpec("w1", 2, 8192, {"Site": '"east"', "HasScratch": "true", "PreemptionRequirements": PREEMPT_POLICY}),
        WorkerSpec("w2", 2, 8192, {"Site": '"east"', "HasScratch": "true", "PreemptionRequirements": PREEMPT_POLICY}),
        WorkerSpec(
            "w3",
            2,
            4096,
            {
                "Site": '"west"',
                "Gpus": "1",
                "GpuMemory": "16384",
                # GPU box: only takes small-memory jobs, prefers GPU work, never preempts.
                "Requirements": "TARGET.RequestMemory <= 1024",
                "Rank": "TARGET.RequestGpus =?= 1 ? 10 : 0",
            },
        ),
        WorkerSpec(
            "w4",
            2,
            2048,
            {
                "Site": '"west"',
                "HasScratch": "false",
                # carol's group owns this box: her jobs rank highest and may preempt anyone.
                "Rank": 'TARGET.Owner == "carol" ? 10 : 0',
                "PreemptionRequirements": f'TARGET.Owner == "carol" || {PREEMPT_POLICY}',
            },
        ),
    ]


class Demo:
    def __init__(self, workdir: Path, scale: Scale, verbose: bool = True) -> None:
        self.workdir = workdir
        self.scale = scale
        self.verbose = verbose
        self.lines: list[str] = []
        self.pool = Pool(
            workdir,
            workers(),
            half_life=4.0,
            cycle=0.25,
            quotas={"bob": 4},
            heartbeat_timeout=1.2,
            ttl=1.5,
            log=self.say,
            on_decision=self.on_decision,
        )
        self.matches = 0
        self.preemptions = 0

    def say(self, msg: str = "") -> None:
        self.lines.append(msg)
        if self.verbose:
            print(msg, flush=True)

    def header(self, title: str) -> None:
        self.say("")
        self.say(f"== {title} ".ljust(78, "="))

    def on_decision(self, t: float, d: Decision) -> None:
        self.matches += 1
        if d.preempts:
            self.preemptions += 1
        self.say(f"[{t:5.1f}s] {d.describe()}")

    # -- submissions ----------------------------------------------------
    def pi(self, owner: str, seed: int, **kw: object) -> Job:
        args = [
            "python",
            "-m",
            "matchmaker.workloads",
            "pi",
            "--samples",
            "20000",
            "--duration",
            str(self.scale.pi_seconds),
            "--seed",
            str(seed),
        ]
        return self.pool.schedd.submit(owner, args, **kw)  # type: ignore[arg-type]

    def longpi(self, owner: str, seed: int) -> Job:
        args = [
            "python",
            "-m",
            "matchmaker.workloads",
            "longpi",
            "--chunks",
            str(self.scale.long_chunks),
            "--samples",
            "20000",
            "--seed",
            str(seed),
            "--pause",
            str(self.scale.long_pause),
        ]
        # Long jobs prefer the big-memory east machines.
        return self.pool.schedd.submit(owner, args, checkpoint=True, rank='TARGET.Site == "east" ? 1 : 0')

    def fairshare_table(self) -> None:
        running = self.pool.running_by_user()
        snap = self.pool.accountant.snapshot()
        idle: dict[str, int] = {}
        for j in self.pool.schedd.jobs.values():
            if j.status is JobStatus.IDLE:
                idle[j.owner] = idle.get(j.owner, 0) + 1
        self.say(f"[{self.pool.now():5.1f}s] fair-share   user   running  idle  real-prio  eff-prio")
        for user, (_cores, real, eff) in snap.items():
            self.say(f"{'':21}{user:<6} {running.get(user, 0):>7} {idle.get(user, 0):>5} {real:>10.2f} {eff:>9.2f}")

    async def fairshare_monitor(self, period: float) -> None:
        while True:
            await asyncio.sleep(period)
            self.fairshare_table()

    # -- scenario ---------------------------------------------------------
    async def run(self) -> dict[str, object]:
        started = time.monotonic()
        self.header("starting pool: collector + negotiator + 4 startd processes")
        await self.pool.start()
        for ad in self.pool.collector.machines():
            self.say(
                f"  {ad.get('Name')!s:<10} cpus={ad.get('Cpus')} mem={ad.get('Memory')}MB "
                f"site={ad.get('Site')} gpus={ad.get('Gpus', 0)} preempt={'PreemptionRequirements' in ad}"
            )
        monitor = asyncio.create_task(self.fairshare_monitor(2.0))
        try:
            summary = await self._scenario()
        finally:
            monitor.cancel()
            await self.pool.stop()
        summary["elapsed"] = round(time.monotonic() - started, 1)
        self.header("summary")
        self.say(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    async def _scenario(self) -> dict[str, object]:
        sch = self.pool.schedd
        self.header("alice submits 4 checkpointing jobs + 12 pi shards")
        alice = [self.longpi("alice", seed=i) for i in range(4)]
        alice += [self.pi("alice", seed=100 + i) for i in range(12)]
        await asyncio.sleep(1.5)

        self.header("bob submits 10 pi shards, 2 GPU jobs, a flaky job and an impossible job")
        bob = [self.pi("bob", seed=200 + i) for i in range(10)]
        bob += [
            self.pi("bob", seed=300 + i, requirements="TARGET.Gpus >= 1", attrs={"RequestGpus": 1}) for i in range(2)
        ]
        flaky = sch.submit(
            "bob",
            [
                "python",
                "-m",
                "matchmaker.workloads",
                "flaky",
                "--marker",
                str(self.workdir / "flaky.attempts"),
                "--fail-times",
                "2",
            ],
            max_retries=3,
        )
        impossible = self.pi("bob", seed=999, requirements="TARGET.Gpus >= 2 && TARGET.HasScratch", request_memory=3000)
        bob += [flaky, impossible]
        await asyncio.sleep(0.5)
        machines = await self._machines()
        self.say(analyze(impossible.ad, machines, impossible.job_id).render())
        sch.remove(impossible.job_id, "no machine can satisfy it (see analysis)")
        self.say(f"  removed job {impossible.job_id}")

        await asyncio.sleep(1.0)
        self.header("carol arrives with 10 pi shards and a better priority")
        carol = [self.pi("carol", seed=400 + i) for i in range(10)]

        self.header("fault injection: freeze the worker running a checkpointing job")
        victim = await self._wait_checkpointing_job(alice[:4])
        frozen_machine = None
        if victim is not None:
            job, machine, chunk = victim
            frozen_machine = machine
            self.say(f"  job {job.job_id} is on {job.slot} with checkpoint at chunk {chunk}")
            self.pool.freeze_worker(machine)
            await self._wait(lambda: job.lost > 0, 10.0)
            await self.pool.kill_worker(machine)
            await asyncio.sleep(0.3)
            await self.pool.start_worker(machine)
            self.say(f"  worker {machine} restarted and will re-advertise")

        dag_ok, rescue_used = await self._dag()

        everything = alice + bob + carol
        await self.pool.wait_jobs(everything, self.scale.timeout)
        self.fairshare_table()
        return self._report(alice, bob, carol, flaky, frozen_machine, dag_ok, rescue_used)

    async def _machines(self) -> list[Ad]:
        from .collector import query

        return await query(self.pool.collector.address)

    async def _wait(self, cond, timeout: float) -> None:  # type: ignore[no-untyped-def]
        deadline = time.monotonic() + timeout
        while not cond():
            if time.monotonic() > deadline:
                raise TimeoutError("condition not reached")
            await asyncio.sleep(0.05)

    async def _wait_checkpointing_job(self, jobs: list[Job]) -> tuple[Job, str, int] | None:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            for job in jobs:
                ckpt = Path(job.env["MATCHMAKER_CHECKPOINT"])
                if job.status is JobStatus.RUNNING and job.slot and ckpt.exists():
                    try:
                        chunk = json.loads(ckpt.read_text())["chunk"]
                    except (ValueError, KeyError):
                        continue
                    if 2 <= chunk <= self.scale.long_chunks - 6:
                        return job, job.slot.split("@", 1)[1], chunk
            await asyncio.sleep(0.05)
        self.say("  (no checkpointing job was mid-run; skipping fault injection)")
        return None

    async def _dag(self) -> tuple[bool, bool]:
        self.header("carol submits a map-reduce DAG")
        d = self.workdir / "dag"
        d.mkdir(exist_ok=True)
        wl = "python -m matchmaker.workloads"
        shards = 4
        lines = [
            f"JOB split {wl} call matchmaker.corpus:write_shards --args '[\"{d}\", {shards}, {self.scale.words}]'",
        ]
        for i in range(1, shards + 1):
            lines.append(f"JOB map{i} {wl} wc-map --input {d}/shard{i}.txt --out {d}/map{i}.json")
        maps = " ".join(f"{d}/map{i}.json" for i in range(1, shards + 1))
        lines += [
            f"JOB reduce {wl} wc-reduce --out {d}/counts.json --top 5 {maps}",
            f"JOB report {wl} call matchmaker.corpus:summarize --args '[\"{d}/counts.json\"]'",
            "PARENT split CHILD " + " ".join(f"map{i}" for i in range(1, shards + 1)),
            "PARENT " + " ".join(f"map{i}" for i in range(1, shards + 1)) + " CHILD reduce",
            "PARENT reduce CHILD report",
            "RETRY map2 2",
            f"SCRIPT POST reduce python -m matchmaker.corpus verify {d}/counts.json $RETURN",
            # The report is gated on an approval flag that does not exist yet.
            f"SCRIPT PRE report python -m matchmaker.corpus require {d}/approved.flag",
            "RETRY report 1",
        ]
        dag_file = d / "wordcount.dag"
        dag_file.write_text("\n".join(lines) + "\n")
        dag = Dag.load(dag_file)
        self.say(f"  {len(dag.nodes)} nodes, order: {' -> '.join(dag.topological_order())}")

        async def script(args: list[str]) -> int:
            return await run_local_script(args, cwd=self.workdir)

        runner = DagRunner(dag, self.pool.dag_job_runner("carol"), script, on_event=self._dag_event)
        ok = await asyncio.wait_for(runner.run(), self.scale.timeout)
        rescue_used = False
        if not ok:
            rescue = dag.write_rescue()
            self.say(f"  DAG incomplete; rescue file {rescue.name}:")
            for line in rescue.read_text().splitlines():
                self.say(f"    {line}")
            (d / "approved.flag").write_text("ok\n")
            self.say("  approval flag created; resuming from the rescue file")
            resumed = Dag.load(dag_file, latest_rescue(dag_file))
            runner = DagRunner(resumed, self.pool.dag_job_runner("carol"), script, on_event=self._dag_event)
            ok = await asyncio.wait_for(runner.run(), self.scale.timeout)
            rescue_used = True
            self.say(f"  resumed run executed: {', '.join(runner.completion_order)}")
        counts = json.loads((d / "counts.json").read_text())
        self.say(f"  word count: {counts['total']} words, top 5: {counts['top']}")
        self.say(f"  DAG {'completed' if ok else 'FAILED'}")
        return ok, rescue_used

    def _dag_event(self, msg: str) -> None:
        self.say(f"[{self.pool.now():5.1f}s] {msg}")

    def _report(
        self,
        alice: list[Job],
        bob: list[Job],
        carol: list[Job],
        flaky: Job,
        frozen: str | None,
        dag_ok: bool,
        rescue_used: bool,
    ) -> dict[str, object]:
        self.header("results")
        samples = hits = shards = 0
        resumed = []
        for job in alice + bob + carol:
            if job.status is not JobStatus.COMPLETED or not job.stdout.strip():
                continue
            out = json.loads(job.stdout.strip().splitlines()[-1])
            if out.get("kind") in ("pi", "longpi"):
                samples += out["samples"]
                hits += out["hits"]
                shards += 1
            if out.get("kind") == "longpi" and out["resumed_from_chunk"] > 0:
                resumed.append(f"{job.job_id} (resumed at chunk {out['resumed_from_chunk']})")
        estimate = 4 * hits / samples if samples else float("nan")
        self.say(
            f"  pi from {shards} shards, {samples:,} samples: {estimate:.5f} (error {abs(estimate - math.pi):.5f})"
        )
        self.say(f"  checkpointing jobs resumed from a checkpoint: {', '.join(resumed) or 'none'}")
        self.say(f"  flaky job {flaky.job_id}: {flaky.status.value} after {flaky.runs} run(s)")
        lost = [j.job_id for j in self.pool.schedd.jobs.values() if j.lost]
        self.say(f"  jobs rescheduled after a lost worker: {', '.join(lost) or 'none'}")
        return {
            "jobs": self.pool.schedd.counts(),
            "submitted_by_user": {"alice": len(alice), "bob": len(bob), "carol": len(carol)},
            "matches": self.matches,
            "preemptions": self.preemptions,
            "negotiation_cycles": self.pool.negotiator.cycles,
            "pi_estimate": round(estimate, 5),
            "checkpoint_resumes": len(resumed),
            "lost_and_rescheduled": len(lost),
            "frozen_worker": frozen,
            "flaky_runs": flaky.runs,
            "dag_completed": dag_ok,
            "dag_used_rescue": rescue_used,
        }


def run_demo(
    scale: str = "full", workdir: Path | None = None, verbose: bool = True
) -> tuple[dict[str, object], list[str]]:
    tmp = None
    if workdir is None:
        tmp = Path(tempfile.mkdtemp(prefix="matchmaker-demo-"))
        workdir = tmp
    try:
        demo = Demo(workdir, SCALES[scale], verbose)
        summary = asyncio.run(demo.run())
        return summary, demo.lines
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="matchmaker-grid demo")
    p.add_argument("--quick", action="store_true", help="smaller workloads (used by tests and CI)")
    p.add_argument("--workdir", type=Path)
    a = p.parse_args(argv)
    summary, _ = run_demo("quick" if a.quick else "full", a.workdir)
    ok = summary["dag_completed"] and summary["jobs"]["failed"] == 0  # type: ignore[index]
    return 0 if ok else 1
