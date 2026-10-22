"""Symmetric matching, rank and diagnostics."""

from __future__ import annotations

from matchmaker.ads import Ad
from matchmaker.matchmaking import analyze, best_machine, rank, requirements_met, symmetric_match


def machine(name: str, **attrs: object) -> Ad:
    ad = Ad({"Name": name, "Cpus": 1, "Memory": 4096, "OpSys": "LINUX", **attrs})
    return ad


def job(requirements: str | None = None, rank_expr: str | None = None, **attrs: object) -> Ad:
    ad = Ad({"Owner": "alice", "RequestCpus": 1, "RequestMemory": 1024, **attrs})
    if requirements:
        ad.set_expr("Requirements", requirements)
    if rank_expr:
        ad.set_expr("Rank", rank_expr)
    return ad


def test_both_sides_must_agree():
    j = job("TARGET.Memory >= MY.RequestMemory")
    m = machine("m1")
    m.set_expr("Requirements", 'TARGET.Owner != "mallory"')
    assert symmetric_match(j, m)
    m.set_expr("Requirements", 'TARGET.Owner == "bob"')
    assert requirements_met(j, m)
    assert not requirements_met(m, j)
    assert not symmetric_match(j, m)


def test_undefined_requirements_never_match():
    j = job("TARGET.Gpus >= 1")  # machine has no Gpus attribute
    assert not symmetric_match(j, machine("m1"))
    # ...unless the expression guards for presence explicitly.
    j2 = job("TARGET.Gpus =?= undefined || TARGET.Gpus >= 1")
    assert symmetric_match(j2, machine("m1"))


def test_missing_requirements_means_unconstrained():
    assert symmetric_match(Ad({"Owner": "x"}), Ad({"Name": "m"}))


def test_rank_on_both_sides():
    j = job(None, 'TARGET.Memory + (TARGET.Site == "east" ? 10000 : 0)')
    m1 = machine("m1", Memory=8192, Site="west")
    m2 = machine("m2", Memory=2048, Site="east")
    assert rank(j, m1) == 8192
    assert rank(j, m2) == 12048
    assert best_machine(j, [m1, m2]) is m2

    owner_pref = machine("m3")
    owner_pref.set_expr("Rank", 'TARGET.Owner == "alice"')
    assert rank(owner_pref, j) == 1.0
    assert rank(owner_pref, job(Owner="bob")) == 0.0


def test_machine_rank_breaks_job_rank_ties():
    j = job()
    plain = machine("a-plain")
    gpu = machine("b-gpu")
    gpu.set_expr("Rank", "TARGET.RequestGpus =?= 1 ? 10 : 1")
    assert best_machine(j, [plain, gpu]) is gpu


def test_non_numeric_rank_is_zero():
    j = job(None, "TARGET.NoSuchAttr * 2")
    assert rank(j, machine("m")) == 0.0
    j2 = job(None, '"high"')
    assert rank(j2, machine("m")) == 0.0


def test_best_machine_none_when_nothing_matches():
    assert best_machine(job("TARGET.Memory > 99999"), [machine("m1"), machine("m2")]) is None


def test_analysis_counts_rejections_per_clause():
    machines = [
        machine("m1", Memory=8192, Gpus=1),
        machine("m2", Memory=8192),
        machine("m3", Memory=1024, Gpus=2),
        machine("m4", Memory=16384, Gpus=2, OpSys="WINDOWS"),
    ]
    j = job('TARGET.Memory >= 4096 && TARGET.Gpus >= 1 && TARGET.OpSys == "LINUX"')
    report = analyze(j, machines, "7.0")
    by_clause = {c.clause: c for c in report.job_clauses}
    mem = by_clause["TARGET.Memory >= 4096"]
    gpus = by_clause["TARGET.Gpus >= 1"]
    assert mem.rejected == 1 and mem.survivors_after == 3
    assert gpus.rejected == 1 and gpus.outcomes["undefined"] == 1
    assert gpus.survivors_after == 2
    assert by_clause['TARGET.OpSys == "LINUX"'].survivors_after == 1
    assert report.matching == ["m1"]
    text = report.render()
    assert "undefined 1" in text and "matching machines: m1" in text


def test_analysis_reports_machine_side_rejections():
    m = machine("picky")
    m.set_expr("Requirements", 'TARGET.Owner == "bob" && TARGET.RequestMemory < 512')
    report = analyze(job(), [m], "1.0")
    assert report.matching == []
    assert report.machines_rejecting_job == {"picky": 'TARGET.Owner == "bob"  -> false'}
    assert "will stay idle" in report.render()
