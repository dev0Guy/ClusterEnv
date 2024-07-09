import tianshou as ts
from tianshou.utils import TensorboardLogger
from tianshou.utils.logger.tensorboard import SummaryWriter
from envs.cluster.env import ClusterEnv
from envs.cluster.wrapper import ClusterWrapper
import numpy as np
import torch
from torch import nn


def create_env(mode=None):
    return ClusterWrapper(ClusterEnv(n_machines=2, n_jobs=3, n_resource=1, max_ticks=1, render_mode=mode, max_steps=10), queue_size=2)  # TODO: find more dynamic way for max_steps


env = create_env()
train_envs = ts.env.DummyVectorEnv([create_env for _ in range(1)])
test_envs = ts.env.DummyVectorEnv([create_env for _ in range(1)])


class Net(nn.Module):
    def __init__(self, state: dict, action: dict):
        super().__init__()
        machines = state["machinesAvailability"]
        jobs = state["jobsUsage"]
        combined: int = np.prod(machines.shape) + np.prod(jobs.shape)
        self.model = nn.Sequential(
            nn.Linear(combined, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action.n),
            nn.Softmax(dim=-1),
        )

    def forward(self, obs, state=None, **kwargs):
        if not isinstance(obs, dict):
            obs = {k: torch.tensor(v, dtype=torch.float) for k, v in obs.items()}
        batch_size = obs["machinesAvailability"].shape[0]
        machines = obs["machinesAvailability"].view(batch_size, -1)
        jobs = obs["jobsUsage"].view(batch_size, -1)
        x = torch.cat([machines, jobs], dim=1)
        logits = self.model(x)
        return logits, state


state_shape = env.observation_space
action_shape = env.action_space

net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)
policy = ts.policy.PGPolicy(
    actor=net,
    optim=optim,
    action_space=env.action_space,
    dist_fn=torch.distributions.Categorical,
    action_scaling=False,
)
logger = TensorboardLogger(writer=SummaryWriter("assets/log"))

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(10_000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)
print("Start Running")
result = ts.trainer.OnpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=100,
    episode_per_collect=10,
    step_per_epoch=200,
    batch_size=32,
    episode_per_test=1,
    logger=logger,
    repeat_per_collect=1,
).run()

print("Finsih Training")
