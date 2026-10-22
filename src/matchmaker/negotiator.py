"""The negotiation cycle, as a pure function of pool state.

Inputs: idle jobs, the current slots (each possibly claimed), the clock.
Outputs: a list of decisions (match, or preempt-and-match). The runtime
applies the decisions; this module never touches sockets or processes,
which keeps every policy decision unit-testable.

Cycle outline:

1. Charge the accountant for the cores each user currently holds.
2. Order submitters by effective priority (best first) and compute each
   user's fair share of the pool, capped by their quota.
3. Pass 1: each user, in priority order, gets matches up to their share.
   Pass 2: leftover free slots go to anyone still under quota (work
   conserving: a fair-share limit never leaves a slot idle).
4. For a job with no free match, consider claimed slots whose machine
   policy (``PreemptionRequirements``) admits the job and whose current
   user is over their share and has a sufficiently worse priority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .ads import Ad, is_true
from .fairshare import Accountant, fair_shares
from .matchmaking import rank, symmetric_match


@dataclass
class Slot:
    name: str
    ad: Ad
    claimed_job: str | None = None
    claimed_user: str | None = None
    claimed_checkpoints: bool = False

    @property
    def cores(self) -> float:
        return float(self.ad.get("Cpus", 1))  # type: ignore[arg-type]


@dataclass
class IdleJob:
    job_id: str
    owner: str
    ad: Ad

    @property
    def cores(self) -> float:
        return float(self.ad.get("RequestCpus", 1))  # type: ignore[arg-type]


@dataclass(frozen=True)
class Decision:
    job_id: str
    owner: str
    slot: str
    job_rank: float
    machine_rank: float
    preempts: str | None = None  # job id evicted from the slot
    victim_user: str | None = None

    def describe(self) -> str:
        base = (
            f"match   {self.owner:<6} job {self.job_id:<6} -> {self.slot:<16}"
            f" (job rank {self.job_rank:g}, machine rank {self.machine_rank:g})"
        )
        if self.preempts:
            base = base.replace("match  ", "preempt", 1) + f" evicting {self.victim_user} {self.preempts}"
        return base


@dataclass
class NegotiatorConfig:
    quotas: dict[str, float] = field(default_factory=dict)  # user -> max cores
    preemption_ratio: float = 1.2  # victim prio must exceed candidate prio by this factor
    max_preemptions_per_cycle: int = 2


class Negotiator:
    def __init__(self, accountant: Accountant, config: NegotiatorConfig | None = None) -> None:
        self.accountant = accountant
        self.config = config or NegotiatorConfig()
        self.cycles = 0

    def cycle(self, now: float, idle: list[IdleJob], slots: list[Slot]) -> list[Decision]:
        self.cycles += 1
        usage: dict[str, float] = {}
        for s in slots:
            if s.claimed_user:
                usage[s.claimed_user] = usage.get(s.claimed_user, 0.0) + s.cores
        self.accountant.update(now, usage)

        submitters = sorted({j.owner for j in idle} | set(usage))
        prios = {u: self.accountant.effective(u) for u in submitters}
        total = sum(s.cores for s in slots)
        shares = fair_shares(prios, total)
        limits = {u: min(shares[u], self.config.quotas.get(u, math.inf)) for u in submitters}
        held = dict(usage)
        order = sorted(submitters, key=lambda u: (prios[u], u))
        by_user: dict[str, list[IdleJob]] = {u: [] for u in order}
        for j in idle:
            by_user[j.owner].append(j)

        free = [s for s in slots if s.claimed_job is None]
        decisions: list[Decision] = []
        matched: set[str] = set()

        def try_free(job: IdleJob) -> bool:
            best = _pick(job, free)
            if best is None:
                return False
            free.remove(best)
            decisions.append(Decision(job.job_id, job.owner, best.name, rank(job.ad, best.ad), rank(best.ad, job.ad)))
            held[job.owner] = held.get(job.owner, 0.0) + job.cores
            matched.add(job.job_id)
            return True

        # Pass 1: up to fair share (rounded up so a small share still gets one slot).
        for user in order:
            for job in by_user[user]:
                if held.get(user, 0.0) + job.cores > _whole(limits[user]):
                    break
                try_free(job)

        # Pass 2: work conserving; quotas are hard, shares are soft.
        for user in order:
            quota = self.config.quotas.get(user, math.inf)
            for job in by_user[user]:
                if job.job_id in matched:
                    continue
                if held.get(user, 0.0) + job.cores > quota:
                    break
                if not free:
                    break
                try_free(job)

        # Pass 3: preemption for users still below their share.
        preemptions = 0
        claimed = [s for s in slots if s.claimed_job is not None]
        for user in order:
            for job in by_user[user]:
                if preemptions >= self.config.max_preemptions_per_cycle:
                    break
                if job.job_id in matched or held.get(user, 0.0) + job.cores > _whole(limits[user]):
                    continue
                victim = self._preemption_target(job, claimed, prios, held, limits)
                if victim is None:
                    continue
                claimed.remove(victim)
                assert victim.claimed_user is not None
                decisions.append(
                    Decision(
                        job.job_id,
                        job.owner,
                        victim.name,
                        rank(job.ad, victim.ad),
                        rank(victim.ad, job.ad),
                        victim.claimed_job,
                        victim.claimed_user,
                    )
                )
                held[victim.claimed_user] -= victim.cores
                held[user] = held.get(user, 0.0) + job.cores
                matched.add(job.job_id)
                preemptions += 1
        return decisions

    def _preemption_target(
        self,
        job: IdleJob,
        claimed: list[Slot],
        prios: dict[str, float],
        held: dict[str, float],
        limits: dict[str, float],
    ) -> Slot | None:
        cand_prio = prios[job.owner]
        options: list[tuple[float, float, str, Slot]] = []
        for s in claimed:
            victim = s.claimed_user
            if victim is None or victim == job.owner:
                continue
            if prios[victim] <= cand_prio * self.config.preemption_ratio:
                continue
            if held.get(victim, 0.0) <= _whole(limits.get(victim, 0.0)):
                continue  # never push a user below their own (rounded-up) fair share
            if not symmetric_match(job.ad, s.ad):
                continue
            policy = s.ad.copy()
            policy["RemoteUser"] = victim
            policy["RemoteUserPrio"] = prios[victim]
            candidate = job.ad.copy()
            candidate["SubmitterUserPrio"] = cand_prio
            expr = policy.lookup("PreemptionRequirements")
            if expr is None or not is_true(policy.eval("PreemptionRequirements", candidate)):
                continue
            # Prefer evicting the worst-priority user, then a job that checkpoints
            # (it loses no work), then the slot the candidate job ranks highest.
            no_ckpt = 0 if s.claimed_checkpoints else 1
            options.append((-prios[victim], no_ckpt, -rank(job.ad, s.ad), s.name, s))
        if not options:
            return None
        return min(options, key=lambda t: t[:4])[4]


def _whole(share: float) -> float:
    """Shares are fractional, slots are not: round up so every submitter can get
    at least one slot. Used for both the candidate and the victim, so the two
    sides of a preemption decision agree on what "within share" means."""
    return share if math.isinf(share) else math.ceil(share - 1e-9)


def _pick(job: IdleJob, free: list[Slot]) -> Slot | None:
    best: Slot | None = None
    best_key: tuple[float, float, str] | None = None
    for s in free:
        if not symmetric_match(job.ad, s.ad):
            continue
        key = (-rank(job.ad, s.ad), -rank(s.ad, job.ad), s.name)
        if best_key is None or key < best_key:
            best, best_key = s, key
    return best
