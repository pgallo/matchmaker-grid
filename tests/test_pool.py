"""Integration tests: real collector, startd subprocesses and schedd over TCP."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from matchmaker import collector as collector_mod
from matchmaker.ads import Ad
from matchmaker.cli import matchmaker_main
from matchmaker.demo import run_demo
from matchmaker.pool import Pool, WorkerSpec
from matchmaker.schedd import JobStatus
from matchmaker.workloads import count_words

PY = [sys.executable, "-m", "matchmaker.workloads"]


def make_pool(tmp_path: Path, **kw) -> Pool:
    workers = kw.pop("workers", [WorkerSpec("w1", 2, 4096, {"Site": '"east"'}), WorkerSpec("w2", 1, 1024)])
    return Pool(tmp_path, workers, cycle=0.1, heartbeat_timeout=1.0, ttl=1.0, log=lambda _m: None, **kw)


async def with_pool(pool: Pool, body):
    await pool.start()
    try:
        return await asyncio.wait_for(body(pool), 30)
    finally:
        await pool.stop()


def test_collector_registers_ads_and_answers_constraint_queries(tmp_path):
    async def body(pool: Pool):
        all_ads = await collector_mod.query(pool.collector.address)
        east = await collector_mod.query(pool.collector.address, 'Site == "east"')
        return [str(a.get("Name")) for a in all_ads], [str(a.get("Name")) for a in east], all_ads

    names, east, ads = asyncio.run(with_pool(make_pool(tmp_path), body))
    assert names == ["slot1@w1", "slot1@w2", "slot2@w1"]
    assert east == ["slot1@w1", "slot2@w1"]
    assert all(isinstance(a, Ad) and a.get("MyAddress") for a in ads)


def test_jobs_run_with_stdout_exit_codes_and_retries(tmp_path):
    async def body(pool: Pool):
        sch = pool.schedd
        ok = sch.submit("alice", [*PY, "pi", "--samples", "5000", "--seed", "1"])
        bad = sch.submit("alice", [sys.executable, "-c", "import sys; print('boom'); sys.exit(4)"], max_retries=1)
        flaky = sch.submit("bob", [*PY, "flaky", "--marker", str(tmp_path / "m"), "--fail-times", "1"], max_retries=2)
        fn = sch.submit_callable("bob", "matchmaker.workloads:count_words", "a b a")
        await pool.wait_jobs([ok, bad, flaky, fn], 20)
        return ok, bad, flaky, fn

    ok, bad, flaky, fn = asyncio.run(with_pool(make_pool(tmp_path), body))
    assert ok.status is JobStatus.COMPLETED and json.loads(ok.stdout)["samples"] == 5000
    assert bad.status is JobStatus.FAILED and bad.exit_code == 4 and bad.runs == 2 and "boom" in bad.stdout
    assert flaky.status is JobStatus.COMPLETED and flaky.runs == 2
    assert json.loads(fn.stdout)["result"] == dict(count_words("a b a"))


def test_requirements_steer_jobs_to_matching_machines(tmp_path):
    async def body(pool: Pool):
        sch = pool.schedd
        big = [sch.submit("u", [*PY, "pi", "--samples", "1000"], request_memory=1500) for _ in range(3)]
        west = sch.submit("u", [*PY, "pi", "--samples", "1000"], requirements="TARGET.Site =?= undefined")
        await pool.wait_jobs([*big, west], 20)
        return big, west

    big, west = asyncio.run(with_pool(make_pool(tmp_path), body))
    assert all(j.slot and j.slot.endswith("@w1") for j in big)
    assert west.slot == "slot1@w2"


def test_lost_worker_is_detected_and_checkpointed_job_resumes(tmp_path):
    workers = [WorkerSpec("w1", 1, 2048), WorkerSpec("w2", 1, 2048)]

    async def body(pool: Pool):
        job = pool.schedd.submit(
            "alice",
            [*PY, "longpi", "--chunks", "30", "--samples", "2000", "--pause", "0.05"],
            checkpoint=True,
        )
        ckpt = Path(job.env["MATCHMAKER_CHECKPOINT"])
        while not (ckpt.exists() and json.loads(ckpt.read_text())["chunk"] >= 5):
            await asyncio.sleep(0.02)
        machine = job.slot.split("@")[1]
        pool.freeze_worker(machine)
        while job.lost == 0:
            await asyncio.sleep(0.05)
        await pool.kill_worker(machine)
        await pool.wait_jobs([job], 20)
        return job, machine

    job, machine = asyncio.run(with_pool(make_pool(tmp_path, workers=workers), body))
    assert job.status is JobStatus.COMPLETED
    assert job.lost == 1 and job.runs == 2
    assert not job.slot.endswith(machine)
    out = json.loads(job.stdout)
    assert out["resumed_from_chunk"] >= 5
    # Same seeds per chunk: the resumed result equals an uninterrupted run.
    assert out["hits"] > 0 and abs(out["pi"] - 3.14159) < 0.1


def test_preemption_vacates_and_requeues(tmp_path):
    policy = "MY.RemoteUserPrio > TARGET.SubmitterUserPrio * 1.2"
    workers = [WorkerSpec("w1", 2, 2048, {"PreemptionRequirements": policy})]

    async def body(pool: Pool):
        sch = pool.schedd
        hogs = [sch.submit("alice", [*PY, "pi", "--samples", "5000", "--duration", "2.5"]) for _ in range(2)]
        await asyncio.sleep(1.5)  # alice's priority rises while she holds both slots
        newcomer = sch.submit("bob", [*PY, "pi", "--samples", "5000"])
        await pool.wait_jobs([*hogs, newcomer], 25)
        return hogs, newcomer

    pool = make_pool(tmp_path, workers=workers, half_life=1.0)
    hogs, newcomer = asyncio.run(with_pool(pool, body))
    assert newcomer.status is JobStatus.COMPLETED
    assert sum(h.preempted for h in hogs) == 1
    assert all(h.status is JobStatus.COMPLETED for h in hogs)
    assert any(d.preempts for _, d in pool.decisions)


def test_cli_eval(capsys):
    expr = 'MY.Memory > 2048 && TARGET.OpSys == "LINUX"'
    rc = matchmaker_main(["eval", expr, "--my", "[Memory = 4096]", "--target", '[OpSys="linux"]'])
    assert rc == 0
    assert "value:  true" in capsys.readouterr().out
    assert matchmaker_main(["eval", "MY.Memory > 2048"]) == 0
    assert "value:  undefined" in capsys.readouterr().out
    assert matchmaker_main(["eval", "1 +"]) == 2


def test_end_to_end_demo(tmp_path):
    summary, lines = run_demo("quick", tmp_path / "demo", verbose=False)
    text = "\n".join(lines)
    assert summary["dag_completed"] is True
    assert summary["dag_used_rescue"] is True
    jobs = summary["jobs"]
    assert jobs["failed"] == 0 and jobs["idle"] == 0 and jobs["running"] == 0
    assert jobs["removed"] == 1 and jobs["completed"] >= 40
    assert summary["flaky_runs"] == 3
    assert summary["preemptions"] >= 1
    assert abs(summary["pi_estimate"] - 3.14159) < 0.01
    assert "Match analysis for job" in text and "fair-share" in text
    assert "resumed run executed: report" in text
    assert summary["elapsed"] < 60
