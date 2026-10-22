"""Fair-share accounting with exponential decay.

Each user has a *real priority* that tracks the number of cores they hold,
smoothed with a half-life ``h``. Over an interval ``dt`` with ``u`` cores in use:

    beta = 0.5 ** (dt / h)
    real = beta * real + (1 - beta) * u

so a user who holds ``u`` cores forever converges to ``real == u``, and a user
who stops running sees their priority halve toward zero every ``h`` seconds.
Real priority never drops below ``MIN_PRIORITY`` so idle users are not
infinitely favoured. The *effective priority* multiplies in a per-user
``priority_factor`` (a static weight, e.g. 2.0 for a guest group).
Lower effective priority is better.

The interval is integrated piecewise-constant: usage is assumed fixed between
two ``update`` calls, which is exactly what a periodic negotiator observes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_PRIORITY = 0.5


@dataclass
class UserRecord:
    real_priority: float = MIN_PRIORITY
    priority_factor: float = 1.0
    cores_in_use: float = 0.0
    accumulated_usage: float = 0.0  # core-seconds, undecayed, for reporting

    @property
    def effective_priority(self) -> float:
        return self.real_priority * self.priority_factor


@dataclass
class Accountant:
    half_life: float
    users: dict[str, UserRecord] = field(default_factory=dict)
    last_update: float | None = None

    def user(self, name: str) -> UserRecord:
        return self.users.setdefault(name, UserRecord())

    def set_factor(self, name: str, factor: float) -> None:
        self.user(name).priority_factor = factor

    def update(self, now: float, usage: dict[str, float]) -> None:
        """Advance the clock to ``now``; ``usage`` is the cores each user holds *from now on*."""
        if self.last_update is not None:
            dt = max(0.0, now - self.last_update)
            beta = decay_factor(dt, self.half_life)
            for rec in self.users.values():
                rec.real_priority = max(MIN_PRIORITY, beta * rec.real_priority + (1 - beta) * rec.cores_in_use)
                rec.accumulated_usage += rec.cores_in_use * dt
        for rec in self.users.values():
            rec.cores_in_use = 0.0
        for name, cores in usage.items():
            self.user(name).cores_in_use = cores
        self.last_update = now

    def effective(self, name: str) -> float:
        return self.user(name).effective_priority

    def snapshot(self) -> dict[str, tuple[float, float, float]]:
        """name -> (cores in use, real priority, effective priority)."""
        return {n: (r.cores_in_use, r.real_priority, r.effective_priority) for n, r in sorted(self.users.items())}


def decay_factor(dt: float, half_life: float) -> float:
    if half_life <= 0:
        return 0.0
    return 0.5 ** (dt / half_life)


def fair_shares(priorities: dict[str, float], total: float) -> dict[str, float]:
    """Split ``total`` cores inversely proportional to effective priority."""
    if not priorities:
        return {}
    inv = {u: 1.0 / p for u, p in priorities.items()}
    norm = sum(inv.values())
    return {u: total * w / norm for u, w in inv.items()}
