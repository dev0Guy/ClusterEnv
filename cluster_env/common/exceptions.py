class NotEnoughSpaceScheduleError(ValueError):
    def __init__(self, msg, *args):
        super(NotEnoughSpaceScheduleError, self).__init__(msg, *args)


class JobStatusScheduleError(ValueError):
    def __init__(self, msg, *args):
        super(JobStatusScheduleError, self).__init__(msg, *args)
