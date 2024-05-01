from clusterEnv.common import Cluster
import gymnasium as gym


class ClusterEnv(gym.Env):
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
        self.render_mode = render_mode
        self.cluster: Cluster = Cluster(nodes, jobs, resource, time, max_episode_steps=max_episode_steps)


