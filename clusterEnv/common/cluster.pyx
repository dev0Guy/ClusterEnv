import numpy as np
from .job cimport JobStatus

class NotEnoughSpaceScheduleError(ValueError):
    def __init__(self, msg, *args):
        super(NotEnoughSpaceScheduleError, self).__init__(msg, *args)


class JobStatusScheduleError(ValueError):
    def __init__(self, msg, *args):
        super(JobStatusScheduleError, self).__init__(msg, *args)



cdef class Cluster:

    def __cinit__(
        self,
        unsigned int nodes,
        unsigned int jobs,
        unsigned int resource,
        unsigned int time,
        *,
        unsigned int max_episode_steps = 0,
    ):
        if max_episode_steps <= 0:
            self.max_episode_steps = time * nodes * jobs
        self.current_time = 0
        self.num_steps = 0
        # TODO: change into a real one
        self.nodes = Nodes(
            np.ones((nodes, resource, time),dtype=np.float64)
        )
        self.jobs = Jobs(
            np.ones((jobs, resource, time), dtype=np.float64),
            np.ones(jobs, dtype=np.uint),
        )

    cdef inline (NodeIndex, JobIndex) convert_action(self, unsigned int action) except *:
        cdef size_t n_job
        if action == 0:
            raise ValueError(f"Action should be bigger than 0. else a time tick action.")
        n_job = len(self.nodes)
        return action % n_job, action // n_job


    cpdef void schedule(self, NodeIndex n_idx, JobIndex j_idx) except *:
        cdef:
            cnp.ndarray[double, ndim=2] space_before_schedule, space_after_schedule
            Node node = self.nodes[n_idx]
            Job job = self.jobs[j_idx]
            bint can_job_fit
        print('Status', job.status)
        if job.status != JobStatus.PENDING: # TODO change back
            raise JobStatusScheduleError(
                f"Job with status {job.status} can't be schedule, only with status: {JobStatus.PENDING}"
            )
        space_before_schedule = node.spec - node.usage
        space_after_schedule = space_before_schedule - job.spec
        can_job_fit = np.all(space_after_schedule >= 0)
        if not can_job_fit:
            raise NotEnoughSpaceScheduleError(
                f"Job={j_idx} can't be allocated into Node={n_idx}"
            )
        job.status = JobStatus.RUNNING
        node.usage += job.spec

    cdef inline dict observation(self):
        return dict(
            Nodes=self.nodes.usage,
            Jobs=self.jobs.spec,
            JobStatus=self.jobs.status,
        )

    def step(self, unsigned int action) -> tuple[dict, float, bool, bool, dict]:
        cdef:
            NodeIndex node_idx
            JobIndex job_idx
            double reward
            bint terminate, trounced
            size_t completed_jobs, running_jobs, total_jobs
        self.num_steps += 1
        if action == 0:
            self.jobs.tick(self.current_time)
            self.nodes.tick(self.current_time)
            self.current_time += 1
        else:
            node_idx, job_idx = self.convert_action(action)
            try:
                self.schedule(node_idx, job_idx)
            except JobStatusScheduleError as e:
                print(e)
                reward = -1_000
            except NotEnoughSpaceScheduleError as e:
                print(e)
                reward = -100
        total_jobs = len(self.jobs)
        running_jobs = np.count_nonzero(self.jobs.status == JobStatus.RUNNING)
        completed_jobs = np.count_nonzero(self.jobs.status == JobStatus.COMPLETE)
        terminate = total_jobs == completed_jobs
        trounced = (
                total_jobs == (completed_jobs + running_jobs)
                or self.num_steps >= self.max_episode_steps
        )
        return self.observation(), reward, terminate, trounced, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        self.current_time = 0
        self.num_steps = 0
        # TODO: reset job, nodes
        self.nodes.usage[:] = 0
        self.jobs.tick(0)
        return self.observation(), {}