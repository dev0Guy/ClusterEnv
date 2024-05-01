from abc import abstractmethod
from typing import overload
from collections.abc import Sequence
from enum import IntEnum
from numba import uint8, float32, uint16, njit, boolean, int64
from numba.experimental import jitclass
import numpy as np
import numba


class JobStatus(IntEnum):
    PENDING = 0
    NONEXISTENT = 1
    RUNNING = 2
    COMPLETE = 3


@njit
def shift_and_zero_last_channel(usage):
    # Get the shape of the array
    height, width, channels = usage.shape
    # Shift elements by one position to the left along the last axis (channels)
    for i in range(channels - 1):
        for j in range(height):
            for k in range(width):
                usage[j, k, i] = usage[j, k, i + 1]

@jitclass([
    ('spec', float32[:, :]),
    ('usage', float32[:, :])
])
class Node:

    def __init__(
        self,
        spec: np.ndarray,
        usage: np.ndarray,
    ):
        self.spec = spec
        self.usage = usage


@jitclass([
    ('status', uint8),
    ('submission', uint16),
    ('usage', float32[:, :]),
    ('wait_time', uint16),
    ('run_time', uint16),
])
class Job:
    def __init__(
        self,
        status: JobStatus,
        submission: int,
        usage: np.ndarray,
        wait_time: int,
        run_time: int,
    ):
        self.status = status
        self.submission = submission
        self.usage = usage
        self.wait_time = wait_time
        self.run_time = run_time


@jitclass([
    ('_spec', float32[:, :, :]),
    ('_usage', float32[:, :, :]),
])
class Nodes:
    def __init__(self, nodes: np.ndarray):
        self._spec = nodes
        self._usage = np.zeros(nodes.shape, dtype=nodes.dtype)

    def reset(self):
        """Reset usage into zeros."""
        self._usage = np.zeros(self._spec.shape, dtype=self._usage.dtype)

    @property
    def usage(self) -> np.ndarray:
        return self._usage

    @property
    def spec(self) -> np.ndarray:
        return self._spec

    def __len__(self) -> int:
        return len(self._spec)

    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> Node:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> Sequence[Node]:
        ...

    def __getitem__(self, index):
        if isinstance(index, int):
            return Node(self._spec[index], self._usage[index])
        raise ValueError(f"asdasd")

    def roll(self):
        shift_and_zero_last_channel(self.usage)
        self.usage[:, :, -1] = 0



@njit
def _calculate_job_length(arr: float32[:, :, :]) -> uint16[:]:
    n_jobs, n_resource, n_time = arr.shape
    length: uint16[:] = np.zeros(n_jobs, dtype=uint16)
    usage: float32[:, :, :] = arr[:, :, ::-1]
    for i_idx in range(n_jobs):
        channel_usage: float32[:, :] = usage[i_idx]
        flip_channel_last_index: uint16[:] = np.argmax(channel_usage != 0, axis=-1)
        flip_channel_last_value: float32[:] = np.array([
            time_arr[idx] for idx, time_arr in zip(flip_channel_last_index, channel_usage)
        ], dtype=float32)
        non_zero_values: uint16[:] = flip_channel_last_index[flip_channel_last_value > 0]
        if len(non_zero_values) > 0:
            length[i_idx] = (n_time - 1) - np.max(non_zero_values)
    return length


from numpy import uint16


# @jitclass([
#     ('_spec', float32[:, :, :]),
#     ('_usage', float32[:, :, :]),
#     ('_submission', uint16[:]),
#     ('_status', uint8[:]),
#     ('_wait_time', uint16[:]),
#     ('_run_time', uint16[:]),
#     ('_length', uint16[:]),
#     ('_mapper', uint16[:])
# ])
class Jobs:

    def __init__(self, usage: float32[:, :, :], submission: uint16[:]):
        self._spec = usage.copy()
        self._usage = usage.copy()
        self._submission = submission
        self._status = np.full(len(self._submission), JobStatus.NONEXISTENT, dtype=np.uint8)
        self._wait_time = np.zeros(len(self._submission), dtype=uint16)
        self._run_time = np.zeros(len(self._submission), dtype=uint16)
        self._length = np.zeros(len(self._submission), dtype=uint16)
        self._length = _calculate_job_length(self._spec)
        self._mapper = np.arange(len(self._submission))

    def update_status(self, index: int, status: JobStatus):
        self._status[index] = status

    def _create_job(self, index: int) -> Job:
        return Job(
            self._status[index],
            self._submission[index],
            self._usage[index],
            self._wait_time[index],
            self._run_time[index],
        )

    @property
    def status(self) -> np.ndarray:
        return self._status

    @property
    def usage(self) -> np.ndarray:
        return self._usage

    def reset(self):
        self._status = np.full(len(self._submission), JobStatus.NONEXISTENT, dtype=self._status.dtype)
        self._wait_time = np.zeros(len(self._submission), dtype=self._wait_time.dtype)
        self._run_time = np.zeros(len(self._submission), dtype=self._run_time.dtype)

    def update_metric_by_time(self, time: int):
        is_pending: boolean[:] = self._status == uint8(JobStatus.PENDING)
        is_running: boolean[:] = self._status == uint8(JobStatus.RUNNING)
        is_nonexistent: boolean[:] = self._status == uint8(JobStatus.NONEXISTENT)
        is_complete: boolean[:] = self._length == self._run_time
        is_new_job: boolean[:] = np.logical_and(self._submission == time, is_nonexistent)
        # update values
        self._wait_time[is_pending] = self._wait_time[is_pending] + 1
        self._run_time[is_running] = self._run_time[is_running] + 1
        self._status[is_new_job] = uint8(JobStatus.PENDING)
        self._status[is_complete] = JobStatus.COMPLETE

    def by_status(self, status: JobStatus) -> list[Job]:
        indexes: uint16[:] = np.arange(len(self._status))
        indexes = indexes[self.status == uint8(status)]
        return [self._create_job(_) for _ in indexes]

    def _organize_jobs(self) -> None:
        self._mapper = np.argsort(self._status)
        self._usage = self._usage[self._mapper]
        self._status = self._status[self._mapper]
        pass
        # self

    def __getitem__(self, index: int):
        return self._create_job(index)

    def __len__(self):
        return len(self._status)
