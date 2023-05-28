from dilation import Dailation, ActionType, Action
from tensorforce.environments import Environment
from puller import PrometheusPuller
from typing import List
import tensorflow as tf
import math



class CustomEnvironment(Environment):
    @classmethod
    def __calc_stae_shape(cls, nodes_n: int, metrics_n: int) -> tuple:
        r = math.ceil(math.sqrt(nodes_n))
        return (r, r, metrics_n)

    def __init__(
        self,
        url: str,
        node_number: int,
        metrics_names: List[str],
    ):
        super().__init__()
        self._shape = self.__calc_stae_shape(
                nodes_n=node_number, metrics_n=metrics_names
            )
        self._puller = PrometheusPuller(
            url=url,
            selected_metrics=metrics_names,
            desired_shape=self._shape,
        )
        self._action_n = 10

    def states(self):
        # return Dailation(self._puller.current, self._shape)
        return dict(type="float", shape=self._shape)

    def actions(self):
        return dict(type="int", num_values=self._action_n)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(8,))
        return state

    def execute(self, actions):
        next_state = np.random.random(size=(8,))
        terminal = False  # Always False if no "natural" terminal state
        reward = np.random.random()
        return next_state, terminal, reward
