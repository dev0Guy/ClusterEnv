from typing import NewType
from abc import ABC, abstractmethod
import numpy as np
from collections import deque

NodeIndex = NewType("NodeIndex", int)
JobIndex = NewType("JobIndex", int)


class AbstractScheduler(ABC):
    def __init__(self, n_jobs: int):
        self._prev_jobs: np.ndarray = np.zeros(n_jobs)

    def received_jobs(self, jobs: np.ndarray) -> list[JobIndex]:
        axis: tuple[int, ...] = tuple(_ for _ in range(1, jobs.ndim))
        current_jobs = np.array(np.any(jobs != 0, axis=axis)).astype(bool)
        delta_jobs = current_jobs & np.logical_xor(current_jobs, self._prev_jobs)
        self._prev_jobs = current_jobs
        return np.where(delta_jobs)[0]

    @staticmethod
    def possible_nodes_for_job(
        jobs: np.ndarray, nodes: np.ndarray, *, j_idx: JobIndex
    ) -> list[NodeIndex]:
        axis: tuple[int, ...] = tuple(_ for _ in range(1, jobs.ndim))
        return np.where(np.all((nodes - jobs[j_idx]) >= 0, axis=axis))[0]

    @staticmethod
    def convert(n_idx: NodeIndex, j_idx: JobIndex, *, n_nodes: int):
        return 1 + n_idx + j_idx * n_nodes

    @abstractmethod
    def select(self, observation: dict[str, np.ndarray]):
        ...


class FirstComeFirstServed(AbstractScheduler):
    def __init__(self, n_jobs: int):
        super().__init__(n_jobs)
        self.queue = deque()

    def select(self, observation: dict[str, np.ndarray]) -> int:
        jobs, nodes = observation["Jobs"], observation["Nodes"]
        for j_idx in self.received_jobs(jobs):
            self.queue.append(j_idx)
        if self.queue:
            j_idx: int = self.queue[-1]
            n_indexes = self.possible_nodes_for_job(jobs, nodes, j_idx=j_idx)
            if len(n_indexes) > 0:
                self.queue.pop()
                return self.convert(n_indexes[0], j_idx, n_nodes=nodes.shape[0])
        return 0


class ShortestJobFirst(AbstractScheduler):
    def __init__(self, n_jobs: int):
        super().__init__(n_jobs)
        self.queue = []

    def select(self, observation: dict[str, np.ndarray]) -> int:
        nodes, jobs = observation["Nodes"], observation["Jobs"]
        jobs_length = (jobs.shape[-1] - (jobs[:, :, ::-1] != 0).argmax(axis=-1)).max(
            axis=-1
        )
        for j_idx in self.received_jobs(jobs):
            self.queue.append([j_idx, jobs_length[j_idx]])
        print("-"*50)
        print(f"{self.queue=}")
        if self.queue:
            queue = np.array(self.queue)
            sorted_idx = np.argsort(queue[:, 1])
            print(f"{sorted_idx=}")
            q_idx = sorted_idx[0]
            j_idx = queue[q_idx, 0]
            print(f"{q_idx=}, {j_idx=}")
            n_indexes = self.possible_nodes_for_job(jobs, nodes, j_idx=j_idx)
            if len(n_indexes) > 0:
                print(f"Removing: {q_idx}")
                self.queue.pop(q_idx)
                action = self.convert(n_indexes[0], j_idx, n_nodes=nodes.shape[0])
                return action
        return 0
