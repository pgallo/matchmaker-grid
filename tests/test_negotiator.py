"""Fair-share math and negotiation policy (no processes involved)."""

from __future__ import annotations

import math

import pytest

from matchmaker.ads import Ad
from matchmaker.fairshare import MIN_PRIORITY, Accountant, decay_factor, fair_shares
from matchmaker.negotiator import IdleJob, Negotiator, NegotiatorConfig, Slot

POLICY = "MY.RemoteUserPrio > TARGET.SubmitterUserPrio * 1.2"


# -- fair-share ----------------------------------------------------------


def test_decay_factor_halves_every_half_life():
    assert decay_factor(0, 10) == 1.0
    assert decay_factor(10, 10) == pytest.approx(0.5)
    assert decay_factor(30, 10) == pytest.approx(0.125)
    assert decay_factor(5, 0) == 0.0


def test_priority_converges_to_usage_and_decays_back():
    acc = Accountant(half_life=10.0)
    acc.update(0.0, {"alice": 4})
    acc.update(10.0, {"alice": 4})
    # One half-life of 4 cores: 0.5*0.5 + 0.5*4
    assert acc.user("alice").real_priority == pytest.approx(2.25)
    for t in range(20, 400, 10):
        acc.update(float(t), {"alice": 4})
    assert acc.user("alice").real_priority == pytest.approx(4.0, rel=1e-6)
    acc.update(400.0, {})  # alice stops
    acc.update(410.0, {})
    assert acc.user("alice").real_priority == pytest.approx(MIN_PRIORITY + (4.0 - MIN_PRIORITY) / 2, rel=0.15)
    acc.update(410.0 + 10 * 30, {})
    assert acc.user("alice").real_priority == MIN_PRIORITY


def test_decay_is_independent_of_update_granularity():
    coarse, fine = Accountant(8.0), Accountant(8.0)
    coarse.update(0.0, {"u": 3})
    coarse.update(16.0, {"u": 3})
    fine.update(0.0, {"u": 3})
    for i in range(1, 161):
        fine.update(i * 0.1, {"u": 3})
    assert coarse.user("u").real_priority == pytest.approx(fine.user("u").real_priority)
    expected = 0.25 * MIN_PRIORITY + 0.75 * 3
    assert coarse.user("u").real_priority == pytest.approx(expected)


def test_accumulated_usage_and_factors():
    acc = Accountant(5.0)
    acc.set_factor("guest", 10.0)
    acc.update(0.0, {"guest": 2})
    acc.update(3.0, {"guest": 2})
    assert acc.user("guest").accumulated_usage == pytest.approx(6.0)
    assert acc.effective("guest") == pytest.approx(acc.user("guest").real_priority * 10.0)


def test_fair_shares_inverse_to_priority():
    shares = fair_shares({"a": 1.0, "b": 2.0, "c": 4.0}, total=7.0)
    assert shares == pytest.approx({"a": 4.0, "b": 2.0, "c": 1.0})
    assert math.isclose(sum(shares.values()), 7.0)


# -- negotiator ----------------------------------------------------------


def slots(n: int, policy: str | None = None, prefix: str = "s") -> list[Slot]:
    out = []
    for i in range(n):
        ad = Ad({"Name": f"{prefix}{i}", "Cpus": 1, "Memory": 2048})
        if policy:
            ad.set_expr("PreemptionRequirements", policy)
        out.append(Slot(f"{prefix}{i}", ad))
    return out


def jobs(owner: str, n: int, start: int = 0, **attrs: object) -> list[IdleJob]:
    out = []
    for i in range(n):
        ad = Ad({"Owner": owner, "RequestCpus": 1, "RequestMemory": 512, **attrs})
        ad.set_expr("Requirements", "TARGET.Memory >= MY.RequestMemory")
        out.append(IdleJob(f"{owner}{start + i}", owner, ad))
    return out


def claim(slot: Slot, job_id: str, user: str) -> None:
    slot.claimed_job, slot.claimed_user = job_id, user


def test_equal_priorities_split_the_pool():
    neg = Negotiator(Accountant(10.0))
    pool = slots(6)
    decisions = neg.cycle(0.0, jobs("alice", 10) + jobs("bob", 10), pool)
    per_user = {u: sum(d.owner == u for d in decisions) for u in ("alice", "bob")}
    assert per_user == {"alice": 3, "bob": 3}


def test_work_conserving_when_one_user_is_alone():
    neg = Negotiator(Accountant(10.0))
    decisions = neg.cycle(0.0, jobs("alice", 10), slots(4))
    assert len(decisions) == 4


def test_better_priority_gets_more_slots():
    acc = Accountant(10.0)
    acc.update(0.0, {"heavy": 6})
    acc.update(30.0, {})  # heavy built up priority, now nothing running
    neg = Negotiator(acc)
    decisions = neg.cycle(30.0, jobs("heavy", 10) + jobs("light", 10), slots(6))
    light = sum(d.owner == "light" for d in decisions)
    heavy = sum(d.owner == "heavy" for d in decisions)
    assert light > heavy and light + heavy == 6


def test_quota_is_a_hard_cap_even_with_free_slots():
    neg = Negotiator(Accountant(10.0), NegotiatorConfig(quotas={"bob": 2}))
    decisions = neg.cycle(0.0, jobs("bob", 10), slots(6))
    assert len(decisions) == 2


def test_rank_picks_the_preferred_slot():
    pool = slots(3)
    pool[2].ad["Fast"] = True
    j = jobs("alice", 1)[0]
    j.ad.set_expr("Rank", "TARGET.Fast =?= true ? 1 : 0")
    (decision,) = Negotiator(Accountant(10.0)).cycle(0.0, [j], pool)
    assert decision.slot == "s2" and decision.job_rank == 1


def test_preemption_of_lower_priority_claims():
    acc = Accountant(5.0)
    acc.update(0.0, {"alice": 4})
    pool = slots(4, POLICY)
    for i, s in enumerate(pool):
        claim(s, f"alice-old{i}", "alice")
    neg = Negotiator(acc, NegotiatorConfig(max_preemptions_per_cycle=10))
    decisions = neg.cycle(10.0, jobs("bob", 4), pool)
    preempts = [d for d in decisions if d.preempts]
    # bob's fair share of 4 slots is above 2 (his priority is much better), and alice
    # is never pushed below her own share.
    assert 2 <= len(preempts) < 4
    assert all(d.victim_user == "alice" and d.owner == "bob" for d in preempts)
    assert len({d.slot for d in preempts}) == len(preempts)


def test_no_preemption_without_machine_policy():
    acc = Accountant(5.0)
    acc.update(0.0, {"alice": 4})
    pool = slots(4)  # no PreemptionRequirements
    for i, s in enumerate(pool):
        claim(s, f"a{i}", "alice")
    assert Negotiator(acc).cycle(10.0, jobs("bob", 4), pool) == []


def test_no_preemption_when_priorities_are_close():
    acc = Accountant(5.0)
    acc.update(0.0, {"alice": 1, "bob": 1})
    pool = slots(2, POLICY)
    claim(pool[0], "a0", "alice")
    claim(pool[1], "b0", "bob")
    acc.update(5.0, {"alice": 1, "bob": 1})
    assert Negotiator(acc).cycle(5.0, jobs("bob", 2, start=5), pool) == []


def test_policy_can_name_a_privileged_user():
    # The policy ignores priorities and only admits carol: dave, with the same
    # priority, may not preempt.
    pool = slots(3, 'TARGET.Owner == "carol"')
    for i, s in enumerate(pool):
        claim(s, f"a{i}", "alice")
    acc = Accountant(5.0)
    acc.update(0.0, {"alice": 3})
    decisions = Negotiator(acc).cycle(20.0, jobs("carol", 1) + jobs("dave", 1), pool)
    assert [(d.owner, d.victim_user) for d in decisions] == [("carol", "alice")]


def test_victim_is_never_pushed_below_its_rounded_share():
    pool = slots(1, 'TARGET.Owner == "carol"')
    claim(pool[0], "a0", "alice")
    acc = Accountant(5.0)
    acc.update(0.0, {"alice": 1})
    # Three submitters on one slot: every rounded-up share is one slot.
    assert Negotiator(acc).cycle(20.0, jobs("carol", 1) + jobs("dave", 1), pool) == []


def test_checkpointing_victims_are_preferred():
    acc = Accountant(5.0)
    acc.update(0.0, {"alice": 2})
    pool = slots(2, POLICY)
    claim(pool[0], "plain", "alice")
    claim(pool[1], "ckpt", "alice")
    pool[1].claimed_checkpoints = True
    (d,) = Negotiator(acc).cycle(10.0, jobs("bob", 1), pool)
    assert d.preempts == "ckpt"


def test_preemption_respects_cap_per_cycle():
    acc = Accountant(5.0)
    acc.update(0.0, {"alice": 8})
    pool = slots(8, POLICY)
    for i, s in enumerate(pool):
        claim(s, f"a{i}", "alice")
    decisions = Negotiator(acc, NegotiatorConfig(max_preemptions_per_cycle=2)).cycle(10.0, jobs("bob", 8), pool)
    assert len(decisions) == 2
