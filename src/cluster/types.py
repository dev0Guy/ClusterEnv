from typing import Optional, Tuple
from pydantic import NonNegativeInt
from enum import IntEnum, StrEnum


MachineIndex = NonNegativeInt
JobIndex = NonNegativeInt
SkipTime = bool
ClusterTicks = NonNegativeInt


class ScheduleErrorType(StrEnum):
    ResourceError: str = "Not Enogh resource int machine"
    StatusError: str = 'Scheduling a job can only be with "pending" status.'


class Status(IntEnum):
    NotArrived = 0
    Pending = 1
    Running = 2
    Complete = 3


class Color(StrEnum):
    InCorrect = "#feca57"
    Correct = "#00d2d3"


ActionColor = Optional[Tuple[Tuple[MachineIndex, JobIndex], Color]]
