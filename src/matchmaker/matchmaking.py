"""Symmetric matchmaking, rank, and "why didn't my job match" diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .ads import ERROR, UNDEFINED, Ad, Value, conjuncts, evaluate, is_true, unparse
from .ads.values import is_number


def requirements_met(ad: Ad, other: Ad) -> bool:
    """One side's Requirements, evaluated with MY=ad and TARGET=other.

    A missing Requirements attribute means "no constraints". Anything other
    than exactly true (false, UNDEFINED, ERROR, a number) is a rejection: a
    match must be positively established, never assumed.
    """
    if "Requirements" not in ad:
        return True
    return is_true(ad.eval("Requirements", other))


def symmetric_match(job: Ad, machine: Ad) -> bool:
    return requirements_met(job, machine) and requirements_met(machine, job)


def rank(ad: Ad, other: Ad) -> float:
    """Rank of ``other`` from ``ad``'s point of view; non-numeric ranks count as 0."""
    if "Rank" not in ad:
        return 0.0
    v = ad.eval("Rank", other)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v) if is_number(v) else 0.0  # type: ignore[arg-type]


def best_machine(job: Ad, machines: list[Ad]) -> Ad | None:
    """Highest job Rank among symmetric matches; ties broken by machine Rank, then name."""
    candidates = [m for m in machines if symmetric_match(job, m)]
    if not candidates:
        return None
    return min(candidates, key=lambda m: (-rank(job, m), -rank(m, job), str(m.get("Name", ""))))


# ---------------------------------------------------------------------------
# diagnostics


def _outcome(v: Value) -> str:
    if v is UNDEFINED:
        return "undefined"
    if v is ERROR:
        return "error"
    if v is True:
        return "true"
    if v is False:
        return "false"
    return "non-boolean"


@dataclass
class ClauseReport:
    clause: str
    rejected: int
    outcomes: Counter[str]
    survivors_after: int  # machines still matching after this clause and all before it


@dataclass
class MatchAnalysis:
    job_id: str
    machines: int
    job_clauses: list[ClauseReport]
    machines_rejecting_job: dict[str, str] = field(default_factory=dict)  # name -> failing clause
    matching: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Match analysis for job {self.job_id} against {self.machines} machine(s):"]
        width = max([len(c.clause) for c in self.job_clauses] + [24])
        lines.append(f"  {'job Requirements clause'.ljust(width)}  rejects  survivors")
        for c in self.job_clauses:
            detail = ", ".join(f"{k} {n}" for k, n in sorted(c.outcomes.items()) if k != "true")
            detail = f" ({detail})" if detail else ""
            lines.append(f"  {c.clause.ljust(width)}  {c.rejected:>7}  {c.survivors_after:>9}{detail}")
        if self.machines_rejecting_job:
            lines.append(f"  machines whose Requirements reject this job: {len(self.machines_rejecting_job)}")
            for name, clause in sorted(self.machines_rejecting_job.items()):
                lines.append(f"    {name}: {clause}")
        if self.matching:
            lines.append(f"  matching machines: {', '.join(sorted(self.matching))}")
        else:
            lines.append("  no machine satisfies both sides; the job will stay idle")
        return "\n".join(lines)


def _first_failing_clause(ad: Ad, other: Ad) -> str | None:
    expr = ad.lookup("Requirements")
    if expr is None:
        return None
    for clause in conjuncts(expr):
        v = evaluate(clause, ad, other)
        if not is_true(v):
            return f"{unparse(clause)}  -> {_outcome(v)}"
    return None


def analyze(job: Ad, machines: list[Ad], job_id: str = "?") -> MatchAnalysis:
    """Break the job's Requirements into its ``&&`` clauses and count, per clause,
    how many machines it rejects and with which outcome. Also reports machines
    that would accept on the job side but reject the job themselves."""
    expr = job.lookup("Requirements")
    clauses = conjuncts(expr) if expr is not None else []
    survivors = list(machines)
    reports: list[ClauseReport] = []
    for clause in clauses:
        outcomes: Counter[str] = Counter()
        for m in machines:
            outcomes[_outcome(evaluate(clause, job, m))] += 1
        survivors = [m for m in survivors if is_true(evaluate(clause, job, m))]
        reports.append(ClauseReport(unparse(clause), len(machines) - outcomes["true"], outcomes, len(survivors)))
    rejecting: dict[str, str] = {}
    for m in survivors:
        failing = _first_failing_clause(m, job)
        if failing is not None:
            rejecting[str(m.get("Name", "?"))] = failing
    matching = [str(m.get("Name", "?")) for m in survivors if requirements_met(m, job)]
    return MatchAnalysis(job_id, len(machines), reports, rejecting, matching)
