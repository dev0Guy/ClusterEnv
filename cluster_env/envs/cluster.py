import logging
from typing import SupportsFloat, Any, NewType, Tuple

import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
import gymnasium as gym

from cluster_env.common.types import Nodes, Node, Jobs, Job, JobStatus
from cluster_env.common.exceptions import (
    JobStatusScheduleError,
    NotEnoughSpaceScheduleError,
)
from cluster_env.common.render import ClusterRenderer


NodeIndex = NewType("NodeIndex", int)
JobIndex = NewType("JobIndex", int)


class ClusterEnv(gym.Env):
    EMPTY_ACTION: int = 0
    metadata = {"render_modes": ["human", "rgb_array", ""], "render_fps": 4}

    def __init__(
        self,
        nodes: int,
        jobs: int,
        resource: int,
        time: int,
        *,
        max_episode_steps: int,
        render_mode: str = "",
    ):
        super(ClusterEnv, self).__init__()
        self._max_episode_steps: int = max_episode_steps
        self._current_time: int = 0
        self._n_steps: int = 0
        self._nodes: Nodes = Nodes(np.ones((nodes, resource, time), dtype=np.float32))
        usage = np.ones((jobs, resource, time), dtype=np.float32)
        self._jobs: Jobs = Jobs(
            submission=np.zeros(jobs, dtype=np.uint16), usage=usage
        )
        self.action_space = gym.spaces.Discrete(jobs * nodes + 1)
        #
        self.render_mode = render_mode
        self._plotter = ClusterRenderer(
            nodes=nodes,
            jobs=jobs,
            resource=resource,
            time=time,
            render_mode=render_mode,
            cooldown=1e-4
        )
        self._correct_action = None
        self._error_action = None

    def convert(self, action: ActType) -> Tuple[NodeIndex, JobIndex]:
        action = int(action) - 1
        return NodeIndex(action % len(self._nodes)), JobIndex(
            action // len(self._nodes)
        )

    def schedule(self, action: Tuple[NodeIndex, JobIndex]) -> None:
        n_idx, j_idx = action
        node: Node = self._nodes[n_idx]  # take the value and take the usage
        job: Job = self._jobs[j_idx]
        if job.status != JobStatus.PENDING:
            raise JobStatusScheduleError(
                f"Job with status {job.status} can't be schedule, only with status: {JobStatus.PENDING}"
            )
        space_before_schedule: np.ndarray = node.spec - node.usage
        space_after_schedule: np.ndarray = space_before_schedule - job.usage
        can_job_fit: bool = np.all(space_after_schedule >= 0)
        if not can_job_fit:
            raise NotEnoughSpaceScheduleError(
                f"Job={j_idx} can't be allocated into Node={n_idx}"
            )
        self._jobs.update_status(j_idx, JobStatus.RUNNING)
        self._nodes[n_idx].usage += job.usage

    def _get_observation(self) -> dict:
        return dict(
            Usage=self._nodes.usage,
            Nodes=self._nodes.spec,
            Status=self._jobs.status,
            Queue=self._jobs.usage,
        )

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._n_steps += 1
        self._correct_action = None
        self._error_action = None
        reward: float = 0
        if action == self.EMPTY_ACTION:
            self._jobs.update_metric_by_time(self._current_time)
            self._current_time += 1
            self._nodes.roll()
        else:
            action: Tuple[NodeIndex, JobIndex] = self.convert(action)
            try:
                self.schedule(action)
            except JobStatusScheduleError as e:
                logging.warning(str(e))
                self._error_action = action
                reward = -1_000
            except NotEnoughSpaceScheduleError as e:
                logging.warning(str(e))
                self._error_action = action
                reward = -100
            else:
                self._correct_action = action
        total_jobs: int = len(self._jobs)
        logging.info(self._jobs.status)
        running_jobs: int = len(self._jobs.by_status(JobStatus.RUNNING))
        completed_jobs: int = len(self._jobs.by_status(JobStatus.COMPLETE))
        terminate: bool = total_jobs == completed_jobs
        trounced: bool = (
            total_jobs == (completed_jobs + running_jobs)
            or self._n_steps >= self._max_episode_steps
        )
        self.render()
        return self._get_observation(), reward, terminate, trounced, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self._current_time = 0
        self._n_steps = 0
        self._nodes.reset()
        self._jobs.reset()
        self._jobs.update_metric_by_time(0)
        self.render()
        self._jobs._organize_jobs()
        return self._get_observation(), {}

    def close(self):
        pass

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode:
            fig = self._plotter(
                self._get_observation(),
                current_time=self._current_time,
                error=self._error_action,
                correct=self._correct_action,
            )
            if self.render_mode == "rgb_array":
                buf = fig.canvas.tostring_rgb()
                width, height = fig.canvas.get_width_height()
                expected_height = int(fig.get_figheight() * fig.dpi)
                expected_width = int(fig.get_figwidth() * fig.dpi)
                width_mult: int = expected_width // width
                height_mult: int = expected_height // height
                return np.frombuffer(buf, dtype=np.uint8).reshape(
                    (height_mult * height, width_mult * width, 3)
                )
