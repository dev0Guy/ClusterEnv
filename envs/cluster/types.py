from enum import IntEnum, StrEnum
from typing import Optional, Tuple

from pydantic import NonNegativeInt

MachineIndex = NonNegativeInt
JobIndex = NonNegativeInt
SkipTime = bool
ClusterTicks = NonNegativeInt


class ScheduleErrorType(StrEnum):
    ResourceError: str = "Not Enogh resource int machine"
    StatusError: str = 'Scheduling a job can only be with "pending" status.'


class Status(IntEnum):
    Pending = 1
    Running = 2
    Complete = 3
    NotArrived = 4


class Color(StrEnum):
    InCorrect = "#feca57"
    Correct = "#00d2d3"


ActionColor = Optional[Tuple[Tuple[MachineIndex, JobIndex], Color]]
