"""DAG parsing, ordering, retries, PRE/POST and rescue."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from matchmaker.dag import Dag, DagError, DagRunner, NodeState, latest_rescue

DIAMOND = """
# a diamond with a tail
JOB a echo a
JOB b echo b
JOB c echo c
JOB d echo d
JOB e echo e
PARENT a CHILD b c
PARENT b c CHILD d
PARENT d CHILD e
"""


class FakeCluster:
    """Records job and script executions; results come from per-node scripts."""

    def __init__(self, exits: dict[str, list[int]] | None = None, scripts: dict[str, list[int]] | None = None):
        self.exits = {k: list(v) for k, v in (exits or {}).items()}
        self.scripts = {k: list(v) for k, v in (scripts or {}).items()}
        self.ran: list[str] = []
        self.script_calls: list[list[str]] = []
        self.running = 0
        self.max_running = 0

    async def job(self, node) -> int:
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        self.ran.append(node.name)
        seq = self.exits.get(node.name)
        return seq.pop(0) if seq else 0

    async def script(self, args: list[str]) -> int:
        self.script_calls.append(args)
        seq = self.scripts.get(args[0])
        return seq.pop(0) if seq else 0


def run(dag: Dag, cluster: FakeCluster, **kw) -> tuple[bool, DagRunner]:
    runner = DagRunner(dag, cluster.job, cluster.script, **kw)
    return asyncio.run(runner.run()), runner


def test_topological_order_is_deterministic():
    dag = Dag.parse(DIAMOND)
    assert dag.topological_order() == ["a", "b", "c", "d", "e"]
    assert dag.nodes["d"].parents == {"b", "c"}


def test_cycle_and_unknown_nodes_are_rejected():
    with pytest.raises(DagError, match="cycle"):
        Dag.parse("JOB a x\nJOB b y\nPARENT a CHILD b\nPARENT b CHILD a\n")
    with pytest.raises(DagError, match="unknown node"):
        Dag.parse("JOB a x\nPARENT a CHILD zz\n")
    with pytest.raises(DagError, match="duplicate"):
        Dag.parse("JOB a x\nJOB a y\n")


def test_children_wait_for_all_parents_and_siblings_run_in_parallel():
    cluster = FakeCluster()
    ok, runner = run(Dag.parse(DIAMOND), cluster)
    assert ok
    order = runner.completion_order
    assert order[0] == "a" and order[-2:] == ["d", "e"]
    assert set(order[1:3]) == {"b", "c"}
    assert cluster.max_running == 2


def test_concurrency_limit():
    text = "\n".join(f"JOB n{i} x" for i in range(6))
    cluster = FakeCluster()
    ok, _ = run(Dag.parse(text), cluster, max_concurrent=2)
    assert ok and cluster.max_running == 2


def test_retry_until_success():
    dag = Dag.parse(DIAMOND + "RETRY b 2\n")
    cluster = FakeCluster(exits={"b": [1, 1, 0]})
    ok, _ = run(dag, cluster)
    assert ok
    assert dag.nodes["b"].attempts == 3
    assert cluster.ran.count("b") == 3


def test_failure_marks_descendants_unreachable_but_siblings_finish():
    dag = Dag.parse(DIAMOND + "RETRY b 1\n")
    cluster = FakeCluster(exits={"b": [1, 1]})
    ok, _ = run(dag, cluster)
    assert not ok
    states = {n: dag.nodes[n].state for n in dag.nodes}
    assert states == {
        "a": NodeState.DONE,
        "b": NodeState.FAILED,
        "c": NodeState.DONE,
        "d": NodeState.UNREACHABLE,
        "e": NodeState.UNREACHABLE,
    }
    assert "d" not in cluster.ran


def test_pre_script_failure_skips_the_job():
    dag = Dag.parse(DIAMOND + "SCRIPT PRE c gate\n")
    cluster = FakeCluster(scripts={"gate": [1]})
    ok, _ = run(dag, cluster)
    assert not ok
    assert "c" not in cluster.ran
    assert dag.nodes["c"].state is NodeState.FAILED


def test_post_script_sees_return_code_and_decides():
    # The job fails, but the POST script tolerates it.
    dag = Dag.parse(DIAMOND + "SCRIPT POST b check $RETURN\n")
    cluster = FakeCluster(exits={"b": [7]})
    ok, _ = run(dag, cluster)
    assert ok
    assert ["check", "7"] in cluster.script_calls
    # And the other way round: the job succeeds, the POST vetoes it.
    dag2 = Dag.parse(DIAMOND + "SCRIPT POST e check $RETURN\n")
    ok2, _ = run(dag2, FakeCluster(scripts={"check": [1]}))
    assert not ok2 and dag2.nodes["e"].state is NodeState.FAILED


def test_rescue_file_resumes_only_unfinished_nodes(tmp_path: Path):
    dag_file = tmp_path / "flow.dag"
    dag_file.write_text(DIAMOND)
    dag = Dag.load(dag_file)
    ok, _ = run(dag, FakeCluster(exits={"d": [1]}))
    assert not ok
    rescue = dag.write_rescue()
    assert rescue.name == "flow.dag.rescue001"
    assert rescue.read_text().splitlines()[1:] == ["DONE a", "DONE b", "DONE c"]
    assert latest_rescue(dag_file) == rescue

    resumed = Dag.load(dag_file, latest_rescue(dag_file))
    cluster = FakeCluster()
    ok, runner = run(resumed, cluster)
    assert ok
    assert cluster.ran == ["d", "e"]
    assert runner.completion_order == ["d", "e"]
    # A second failure would get a new, numbered rescue file.
    assert dag.write_rescue().name == "flow.dag.rescue002"
