import clusterEnv
import gymnasium as gym
from clusterEnv.wrapper import ConcatenateObservationDict
import tianshou as ts
from torch import nn
import numpy as np
import torch

from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers.record_video import RecordVideo

WANDB_PROJECT: str = "Thesis"
LOG_FOLDER: str = "./logs"
N_NODES: int = 3


class SimpleANNetwork(nn.Module):

    def __init__(
        self, state_shape: dict[str, gym.spaces.Box], n_action: int, n: int
    ) -> None:
        super().__init__()
        self.jobs, *_ = state_shape["Jobs"].shape
        self.nodes, self.resource, self.time = state_shape["Nodes"].shape

        inp_features = (self.jobs + self.nodes) * self.resource * self.time
        self.netowrk = nn.Sequential(
            nn.Linear(in_features=inp_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_action),
            nn.Softmax(),
        )

    def forward(self, obs, state=None, info={}):
        nodes = obs["Nodes"]
        jobs = obs["Jobs"]
        # cast to tensor if not
        nodes = (
            torch.tensor(nodes, dtype=torch.float)
            if not isinstance(obs, torch.Tensor)
            else nodes
        )
        jobs = (
            torch.tensor(jobs, dtype=torch.float)
            if not isinstance(obs, torch.Tensor)
            else jobs
        )
        nodes = nodes.flatten(1)
        jobs = jobs.flatten(1)
        data = torch.concat((nodes, jobs), dim=1)
        logits = self.netowrk(data)
        return logits, state


class Net(nn.Module):
    def __init__(self, state_shape: dict[str, gym.spaces.Box], n_action: int, n: int):
        super().__init__()
        print(state_shape)
        self.j, *_ = state_shape["Jobs"].shape
        self.n, self.r, self.t = state_shape["Nodes"].shape
        self.nodes_net = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n, out_channels=16, kernel_size=(self.r, self.t)
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(1),
        )
        self.jobs_net = nn.Sequential(
            nn.Conv2d(
                in_channels=self.j, out_channels=16, kernel_size=(self.r, self.t)
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(1),
        )
        self.combined = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_action),
            nn.Softmax(),
        )
        print("=" * 50)
        print(self.nodes_net)
        print("=" * 50)
        print(self.jobs_net)
        print("=" * 50)

    def forward(self, obs, state=None, info={}):
        nodes = obs["Nodes"]
        jobs = obs["Jobs"]
        # cast to tensor if not
        nodes = (
            torch.tensor(nodes, dtype=torch.float)
            if not isinstance(obs, torch.Tensor)
            else nodes
        )
        jobs = (
            torch.tensor(jobs, dtype=torch.float)
            if not isinstance(obs, torch.Tensor)
            else jobs
        )
        # embedding
        node_emb = self.nodes_net(nodes)
        job_emb = self.jobs_net(jobs)
        combined = torch.concat((node_emb, job_emb), dim=1)
        logits = self.combined(combined)
        return logits, state


def create_env(render_mode="", n_jobs: int = 3, n_nodes: int = N_NODES):
    return gym.make(
        "cluster-v0",
        nodes=n_nodes,
        jobs=n_jobs,
        resource=3,
        time=2,
        render_mode=render_mode,
    )


def train(lr: float = 4e-4):
    logger: WandbLogger = WandbLogger(project=WANDB_PROJECT)
    writer: SummaryWriter = SummaryWriter(LOG_FOLDER)
    logger.load(writer=writer)
    env = create_env()
    train_envs = ts.env.DummyVectorEnv([create_env for _ in range(10)])
    test_envs = ts.env.DummyVectorEnv([create_env for _ in range(2)])
    # test_envs = ts.env.DummyVectorEnv([lambda: RecordVideo(create_env('rgb_array'), video_folder='./video') for _ in range(2)])
    state_shape = env.observation_space
    action_shape = env.action_space.n
    net = SimpleANNetwork(state_shape, action_shape, n=N_NODES)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    policy: ts.policy.DQNPolicy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=0.9,
        estimation_step=3,
        target_update_freq=500,
    )
    train_collector = ts.data.Collector(
        policy,
        train_envs,
        ts.data.VectorReplayBuffer(20_000, 10),
        exploration_noise=True,
    )
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)
    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=80,
        step_per_epoch=1_000,
        step_per_collect=10,
        update_per_step=0.1,
        episode_per_test=100,
        batch_size=32,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.03),
        logger=logger,
    ).run()


if __name__ == "__main__":
    train()
