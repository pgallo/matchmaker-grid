"""DAG workflows: parent/child dependencies, node retries, PRE/POST scripts, rescue files.

DAG file format (one statement per line, ``#`` comments)::

    JOB    <node> <command ...>
    PARENT <node ...> CHILD <node ...>
    RETRY  <node> <n>
    SCRIPT PRE  <node> <command ...>
    SCRIPT POST <node> <command ...>
    DONE   <node>                     # written by rescue files

Semantics follow the classic workflow manager design:

* A node runs once all of its parents succeeded. PRE runs on the submit side
  before the job is queued; a failing PRE fails the attempt without running the job.
* POST runs after the job whatever its exit code, with ``$RETURN`` replaced by
  that exit code; when a POST script exists, *its* exit code decides success,
  so a POST can tolerate or veto a job result.
* RETRY re-runs the whole attempt (PRE, job, POST).
* A failed node does not stop the DAG: everything that does not depend on it
  still runs. At the end a rescue file lists the nodes that succeeded; loading
  the DAG with that rescue file skips them.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class NodeState(StrEnum):
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    UNREACHABLE = "unreachable"  # an ancestor failed


@dataclass
class Node:
    name: str
    command: list[str]
    parents: set[str] = field(default_factory=set)
    children: set[str] = field(default_factory=set)
    retries: int = 0
    pre: list[str] | None = None
    post: list[str] | None = None
    state: NodeState = NodeState.WAITING
    attempts: int = 0
    last_exit: int | None = None


class DagError(ValueError):
    pass


@dataclass
class Dag:
    nodes: dict[str, Node]
    source: Path | None = None

    @classmethod
    def parse(cls, text: str, source: Path | None = None) -> Dag:
        nodes: dict[str, Node] = {}
        edges: list[tuple[str, str]] = []
        done: set[str] = set()
        deferred: list[tuple[int, list[str]]] = []
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            words = shlex.split(line)
            kw = words[0].upper()
            if kw == "JOB":
                if len(words) < 3:
                    raise DagError(f"line {lineno}: JOB needs a name and a command")
                if words[1] in nodes:
                    raise DagError(f"line {lineno}: duplicate node {words[1]!r}")
                nodes[words[1]] = Node(words[1], words[2:])
            else:
                deferred.append((lineno, words))
        for lineno, words in deferred:
            kw = words[0].upper()
            if kw == "PARENT":
                upper = [w.upper() for w in words]
                if "CHILD" not in upper:
                    raise DagError(f"line {lineno}: PARENT without CHILD")
                k = upper.index("CHILD")
                for p in words[1:k]:
                    for c in words[k + 1 :]:
                        edges.append((_known(nodes, p, lineno), _known(nodes, c, lineno)))
            elif kw == "RETRY":
                nodes[_known(nodes, words[1], lineno)].retries = int(words[2])
            elif kw == "SCRIPT":
                when = words[1].upper()
                node = nodes[_known(nodes, words[2], lineno)]
                if when == "PRE":
                    node.pre = words[3:]
                elif when == "POST":
                    node.post = words[3:]
                else:
                    raise DagError(f"line {lineno}: SCRIPT must be PRE or POST")
            elif kw == "DONE":
                done.add(_known(nodes, words[1], lineno))
            else:
                raise DagError(f"line {lineno}: unknown keyword {words[0]!r}")
        for p, c in edges:
            nodes[p].children.add(c)
            nodes[c].parents.add(p)
        dag = cls(nodes, source)
        dag.topological_order()  # raises on cycles
        for name in done:
            nodes[name].state = NodeState.DONE
        return dag

    @classmethod
    def load(cls, path: Path, rescue: Path | None = None) -> Dag:
        text = path.read_text()
        if rescue is not None:
            text += "\n" + rescue.read_text()
        return cls.parse(text, path)

    def topological_order(self) -> list[str]:
        """Kahn's algorithm with name-sorted ready sets, so the order is deterministic."""
        indeg = {n: len(node.parents) for n, node in self.nodes.items()}
        ready = sorted(n for n, d in indeg.items() if d == 0)
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for c in sorted(self.nodes[n].children):
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
            ready.sort()
        if len(order) != len(self.nodes):
            cyclic = sorted(set(self.nodes) - set(order))
            raise DagError(f"cycle detected among nodes: {', '.join(cyclic)}")
        return order

    def rescue_text(self) -> str:
        done = [n for n in self.topological_order() if self.nodes[n].state is NodeState.DONE]
        lines = [f"# Rescue file: {len(done)} of {len(self.nodes)} nodes completed", *(f"DONE {n}" for n in done)]
        return "\n".join(lines) + "\n"

    def write_rescue(self, path: Path | None = None) -> Path:
        if path is None:
            if self.source is None:
                raise DagError("no path for the rescue file")
            path = next_rescue_path(self.source)
        path.write_text(self.rescue_text())
        return path

    @property
    def succeeded(self) -> bool:
        return all(n.state is NodeState.DONE for n in self.nodes.values())


def _known(nodes: dict[str, Node], name: str, lineno: int) -> str:
    if name not in nodes:
        raise DagError(f"line {lineno}: unknown node {name!r}")
    return name


def next_rescue_path(dag_file: Path) -> Path:
    i = 1
    while (p := dag_file.with_name(f"{dag_file.name}.rescue{i:03d}")).exists():
        i += 1
    return p


def latest_rescue(dag_file: Path) -> Path | None:
    found = sorted(dag_file.parent.glob(f"{dag_file.name}.rescue[0-9][0-9][0-9]"))
    return found[-1] if found else None


# ---------------------------------------------------------------------------
# execution

JobRunner = Callable[[Node], Awaitable[int]]
ScriptRunner = Callable[[list[str]], Awaitable[int]]
EventSink = Callable[[str], None]


async def run_local_script(args: list[str], cwd: Path | None = None) -> int:
    argv = [sys.executable if a == "python" else a for a in args]
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    return await proc.wait()


class DagRunner:
    def __init__(
        self,
        dag: Dag,
        run_job: JobRunner,
        run_script: ScriptRunner = run_local_script,
        on_event: EventSink | None = None,
        max_concurrent: int = 0,
    ) -> None:
        self.dag = dag
        self.run_job = run_job
        self.run_script = run_script
        self.emit = on_event or (lambda _m: None)
        self.limit = max_concurrent
        self.completion_order: list[str] = []

    async def run(self) -> bool:
        nodes = self.dag.nodes
        running: dict[asyncio.Task[bool], str] = {}
        skipped = sum(n.state is NodeState.DONE for n in nodes.values())
        if skipped:
            self.emit(f"dag: resuming, {skipped} node(s) already done per rescue file")
        while True:
            for name in self.dag.topological_order():
                node = nodes[name]
                if node.state is not NodeState.WAITING:
                    continue
                pstates = {nodes[p].state for p in node.parents}
                if pstates & {NodeState.FAILED, NodeState.UNREACHABLE}:
                    node.state = NodeState.UNREACHABLE
                elif all(s is NodeState.DONE for s in pstates):
                    node.state = NodeState.READY
            for name in self.dag.topological_order():
                if self.limit and len(running) >= self.limit:
                    break
                if nodes[name].state is NodeState.READY:
                    nodes[name].state = NodeState.RUNNING
                    running[asyncio.create_task(self._attempts(nodes[name]))] = name
            if not running:
                break
            finished, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                name = running.pop(task)
                ok = task.result()
                nodes[name].state = NodeState.DONE if ok else NodeState.FAILED
                if ok:
                    self.completion_order.append(name)
                self.emit(f"dag: node {name} {'done' if ok else 'FAILED'} after {nodes[name].attempts} attempt(s)")
        return self.dag.succeeded

    async def _attempts(self, node: Node) -> bool:
        while True:
            node.attempts += 1
            if await self._attempt(node):
                return True
            if node.attempts > node.retries:
                return False
            self.emit(
                f"dag: node {node.name} attempt {node.attempts} failed; retrying ({node.attempts}/{node.retries})"
            )

    async def _attempt(self, node: Node) -> bool:
        if node.pre is not None:
            rc = await self.run_script(node.pre)
            if rc != 0:
                self.emit(f"dag: PRE script of {node.name} exited {rc}")
                return False
        node.last_exit = await self.run_job(node)
        if node.post is not None:
            argv = [a.replace("$RETURN", str(node.last_exit)) for a in node.post]
            rc = await self.run_script(argv)
            if rc != 0:
                self.emit(f"dag: POST script of {node.name} exited {rc} (job exit {node.last_exit})")
            return rc == 0
        return node.last_exit == 0
