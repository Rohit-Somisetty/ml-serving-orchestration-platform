from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Task:
    name: str
    fn: Callable[[dict[str, object]], None]
    deps: list[str] = field(default_factory=list)
    retries: int = 0
    timeout_s: int = 120

    def run(self, context: dict[str, object]) -> None:
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                self.fn(context)
                duration = time.perf_counter() - start
                if duration > self.timeout_s:
                    raise TimeoutError(
                        f"Task {self.name} exceeded timeout {self.timeout_s}s "
                        f"(took {duration:.2f}s)",
                    )
                return
            except Exception:  # pragma: no cover - retries
                if attempt == self.retries:
                    raise


class DAG:
    def __init__(self, name: str, tasks: list[Task]) -> None:
        self.name = name
        self.tasks = {task.name: task for task in tasks}
        self._validate_dependencies()
        self._ordered_tasks = self._topological_sort()

    def _validate_dependencies(self) -> None:
        for task in self.tasks.values():
            for dep in task.deps:
                if dep not in self.tasks:
                    raise ValueError(f"Task {task.name} depends on unknown task {dep}")

    def _topological_sort(self) -> list[Task]:
        visited: set[str] = set()
        temp_mark: set[str] = set()
        order: list[str] = []

        def visit(node: str) -> None:
            if node in temp_mark:
                raise ValueError("Cycle detected in DAG definition")
            if node not in visited:
                temp_mark.add(node)
                for dep in self.tasks[node].deps:
                    visit(dep)
                temp_mark.remove(node)
                visited.add(node)
                order.append(node)

        for task_name in self.tasks:
            visit(task_name)
        return [self.tasks[name] for name in order]

    def execute(self, context: dict[str, object] | None = None) -> None:
        context = context or {}
        for task in self._ordered_tasks:
            task.run(context)

    def task_names(self) -> list[str]:
        return [task.name for task in self._ordered_tasks]
