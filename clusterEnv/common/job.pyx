import numpy as np

cdef class Job:

    def __cinit__(
        self,
        cnp.ndarray[double, ndim=2] spec,
        JobStatus status = JobStatus.NONEXISTENT,
        unsigned long submission = 0,
        unsigned long wait_time = 0,
        unsigned long run_time = 0,
    ):
        self.spec = spec
        self.status = status
        self.submission = submission
        self.wait_time = wait_time
        self.run_time = run_time

    def __str__(self) -> str:
        return f"{type(self).__name__}(status={self.status}, submission={self.submission}, usage={self.spec}, wait_time={self.wait_time}, run_time={self.run_time})"

cdef class Jobs:

    def __cinit__(
        self,
        cnp.ndarray[double, ndim=3] spec,
        cnp.ndarray[unsigned long] submission,
    ):
        self.spec = spec
        self.submission = submission
        self.status = np.full(len(submission), JobStatus.NONEXISTENT, dtype=np.uint)
        self.wait_time = np.zeros(len(submission), dtype=np.uint)
        self.run_time = np.zeros(len(submission), dtype=np.uint)


    cdef void tick(self, unsigned long time):
        cdef:
            cnp.ndarray is_nonexistent = self.status == JobStatus.NONEXISTENT
            cnp.ndarray is_pending = self.status == JobStatus.PENDING
            cnp.ndarray is_running = self.status == JobStatus.RUNNING
            cnp.ndarray is_complete = self.length == self.run_time
            cnp.ndarray is_new_job = np.logical_and(self.submission == time, is_nonexistent)
        self.status[is_new_job] = JobStatus.PENDING
        self.status[is_complete] = JobStatus.COMPLETE
        self.wait_time[is_pending] +=  + 1
        self.run_time[is_running] += 1


    def __getitem__(self, index: int) -> Job:
        return Job(
            self.spec[index], self.status[index], self.submission[index], self.wait_time[index], self.run_time[index]
        )

    def __setitem__(self, index: int, status: JobStatus):
        self.status[index] = status

    def __len__(self) -> int:
        return len(self.submission)