from typing import NamedTuple, MutableSequence, TypeVar, Sequence, overload, Optional
from dataclasses import dataclass, field
from abc import abstractmethod
import numpy.typing as npt
from enum import IntEnum
import numpy as np


T = TypeVar("T", bound="JobStruct")
_T = TypeVar("_T")


class NodeStatus(IntEnum):
    DOWN = 0
    RESTARTING = 1
    UP = 2


class JobStatus(IntEnum):
    PENDING = 0
    NONEXISTENT = 1
    RUNNING = 2
    COMPLETE = 3


class JobStruct(NamedTuple):
    status: int
    submission: int
    usage: npt.NDArray[float]

    def change_status(self, status: JobStatus) -> "JobStruct":
        return JobStruct(status, submission=self.submission, usage=self.usage)


class Nodes(Sequence[_T]):
    __slots__ = ("_original", "_usage")
    _original: npt.NDArray[float]
    _usage: npt.NDArray[float]

    @property
    def original(self) -> npt.NDArray[float]:
        return self._original

    @property
    def usage(self) -> npt.NDArray[float]:
        return self._usage

    def __init__(self, nodes: npt.NDArray[float]):
        self._original = nodes
        self._usage = np.zeros(self._original.shape)

    def tick(self):
        self._usage = np.roll(self._usage, shift=-1, axis=-1)
        self._usage[:, :, -1] = 0

    def free_space(self, index: int) -> npt.NDArray[float]:
        return self._original[index] - self._usage[index]

    def __getitem__(self, index: int) -> npt.NDArray[float]:
        return self._usage[index]

    def __setitem__(self, index: int, value: npt.NDArray[float]) -> None:
        self._usage[index] = value

    def __len__(self):
        return self._usage.shape[0]


@dataclass
class Jobs(Sequence[T]):
    EMPTY_JOB_USAGE = -1
    usage: npt.NDArray[np.float64]
    submission: npt.NDArray[int]
    _status: npt.NDArray[int] = field(init=False)
    _length: npt.NDArray[int] = field(init=False)
    _run_time: npt.NDArray[int] = field(init=False)
    _wait_time: npt.NDArray[int] = field(init=False)

    @property
    def status(self) -> npt.NDArray[int]:
        return self._status

    def __post_init__(self):
        _job_start_time: npt.NDArray[np.uint32] = (self.usage == 0).argmax(axis=-1)
        self._status = np.full(
            shape=len(self.submission),
            fill_value=JobStatus.NONEXISTENT.value,
            dtype=np.uint32,
        )
        self._length = np.max(
            np.where(_job_start_time > 0, _job_start_time, self.usage.shape[0]), axis=1
        ).astype(np.uint32)
        self._run_time = np.zeros(len(self)).astype(int)
        self._wait_time = np.zeros(len(self)).astype(int)

    def reorganize(self, mapping: npt.ArrayLike) -> None:
        self.submission = self.submission[mapping]
        self.usage = self.usage[mapping]
        self._status = self._status[mapping]
        self._wait_time = self._wait_time[mapping]
        self._run_time = self._run_time[mapping]
        self._length = self._length[mapping]

    def total_pending_time(self) -> int:
        return int(np.sum(self._wait_time))

    def index_by_status(self, status: JobStatus) -> npt.NDArray[int]:
        return np.array(np.where(self._status == status))[0]

    def inc(self, status: JobStatus) -> None:
        job_indexes: npt.NDArray[int] = np.array(np.where(self._status == status))
        match status:
            case JobStatus.RUNNING:
                self._run_time[job_indexes] += 1
            case JobStatus.PENDING:
                self._wait_time[job_indexes] += 1
            case JobStatus.NONEXISTENT | JobStatus.COMPLETE:
                raise ValueError

    def check_complete_jobs(self) -> npt.NDArray[int]:
        complete: npt.NDArray[int] = np.array(np.where(self._run_time == self._length))
        self._status[complete] = JobStatus.COMPLETE
        self.usage[complete] = self.EMPTY_JOB_USAGE
        return complete

    def check_arrived_jobs(self, current_time: int) -> npt.NDArray[int]:
        arrived: npt.NDArray[int] = np.array(np.where(self.submission == current_time))
        self._status[arrived] = JobStatus.PENDING

        return arrived

    def __getitem__(self, index: int) -> T:
        return JobStruct(
            status=self._status[index],
            usage=self.usage[index].copy(),
            submission=self.submission[index],
        )

    def __setitem__(self, index: int, value: T) -> None:
        self._status[index] = value.status
        self.usage[index] = value.usage.copy()
        self.submission[index] = value.submission

    def __delitem__(self, index: int) -> None:
        raise ValueError(f"Can't Delete form this object: {type(self)}")

    def __len__(self):
        return len(self._status)
