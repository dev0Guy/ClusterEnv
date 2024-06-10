from pydantic import NonNegativeInt
from typing import Tuple, TypeAlias
from enum import IntEnum, StrEnum


MachineIndex: TypeAlias = NonNegativeInt
JobIndex: TypeAlias = NonNegativeInt
SkipTime: TypeAlias = bool


class ScheduleErrorType(StrEnum):
    ResourceError: str = "Not Enogh resource int machine"
    StatusError: str = 'Scheduling a job can only be with "pending" status.'


class Status(IntEnum):
    NotArrived = 0
    Pending = 1
    Running = 2
    Complete = 3
