from pydantic import PositiveInt
from typing import Tuple, TypeAlias
from enum import IntEnum

Action: TypeAlias = Tuple[PositiveInt, PositiveInt]

class ScheduleError(ValueError):
    def __init__(self, msg, *args, **kwargs):
        super(ScheduleError, self).__init__(msg, *args, **kwargs)

class ResourceError(ScheduleError):
    def __init__(self, msg, *args, **kwargs):
        super(ResourceError, self).__init__(msg, *args, **kwargs)

class StatusError(ScheduleError):
    def __init__(self, msg, *args, **kwargs):
        super(StatusError, self).__init__(msg, *args, **kwargs)

class Status(IntEnum):
    NotArrived = 0
    Pending = 1
    Running = 2
    Complete = 3