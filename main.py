import gymnasium as gym
import numpy as np
import tianshou as ts
import torch
from torch import nn
from src.cluster.env import ClusterEnvironment
from src.cluster.wrapper import ClusterActionWrapper
import wandb

from tianshou.utils.logger.wandb import WandbLogger
from torch.utils.tensorboard import SummaryWriter


# Create the environment
def create_cluster_env():
    return ClusterActionWrapper(ClusterEnvironment(n_machines=5, n_jobs=5, n_resources=1, time=1))


# Define the neural network model
class Net(nn.Module):
    def __init__(self, state: dict, action: dict):
        super().__init__()
        # extract observation information
        machines = state["machines"]
        jobs = state["jobs"]
        combined = np.prod(machines.shape) + np.prod(jobs.shape)
        self.model = nn.Sequential(
            nn.Linear(combined, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action.n),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs, state=None, **_):
        if not isinstance(obs, dict):
            obs = {k: torch.tensor(v, dtype=torch.float) for k, v in obs.items()}
        batch_size = obs['machines'].shape[0]
        machines = obs['machines'].view(batch_size, -1)
        jobs = obs['jobs'].view(batch_size, -1)
        x = torch.cat([machines, jobs], dim=1)
        logits = self.model(x)

        return logits, state


def main() -> None:
    wandb.require("core")
    wandb.init(project="thises")
    config = wandb.config
    config.lr = 1e-3
    logger = WandbLogger()
    logger.load(SummaryWriter("./logs"))

    env = create_cluster_env()
    state_shape = env.observation_space
    action_shape = env.action_space
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=config.lr)

    train_envs = ts.env.DummyVectorEnv([create_cluster_env for _ in range(10)])
    test_envs = ts.env.DummyVectorEnv([create_cluster_env for _ in range(100)])
    dist = torch.distributions.Categorical

    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space
    )

    train_collector = ts.data.Collector(
        policy,
        train_envs,
        ts.data.VectorReplayBuffer(20000, 10),
        exploration_noise=True,
    )
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        logger=logger,  # Use the custom WandbLogger here
        batch_size=64,
        max_epoch=100,
        episode_per_test=20,
        step_per_epoch=1_000,
        episode_per_collect=10,
        repeat_per_collect=1,
    ).run()

    print(result)


if __name__ == "__main__":
    main()




