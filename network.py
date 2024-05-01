# from typing import Tuple, Dict, Optional
# import gymnasium as gym
# from torch import nn
# import numpy as np
# import torch
#
#
# class Net(nn.Module):
#     def __init__(self, state_shape, action_shape) -> None:
#         super(Net, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
#             nn.Linear(128, 128), nn.ReLU(inplace=True),
#             nn.Linear(128, 128), nn.ReLU(inplace=True),
#             nn.Linear(128, np.prod(action_shape)),
#         )
#
#     def forward(self, obs: gym.Space, state: Optional[np.array] = None, info: Dict = {}) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
#         if not isinstance(obs, torch.Tensor):
#             obs = torch.tensor(obs, dtype=torch.float)
#         batch = obs.shape[0]
#         logits = self.model(obs.view(batch, -1))
#         return logits, state
#
#
# class CnnNetwork(nn.Module):
#     def __init__(self, state_shape, action_shape) -> None:
#         super(CnnNetwork, self).__init__()
#         I, C, R, T = state_shape
#         time_resource_c: int = 32
#         self.conv1d_step1 = nn.Sequential(
#             nn.Conv1d(in_channels=R, out_channels=time_resource_c, kernel_size=T, stride=T),
#             nn.ReLU()
#         )
#         self.conv1d_step2 = nn.Sequential(
#             nn.Conv1d(in_channels=C, out_channels=np.prod(action_shape), kernel_size=time_resource_c, stride=time_resource_c),
#             nn.ReLU(),
#         )
#         self.final_step = nn.Sequential(
#             nn.Linear(in_features=I*np.prod(action_shape), out_features=np.prod(action_shape)),
#             nn.Softmax()
#         )
#
#     def forward(self, obs: gym.Space, state: Optional[np.array] = None, info: Dict = {}) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
#         B, I, C, R, T = obs.shape
#         if not isinstance(obs, torch.Tensor):
#             obs = torch.tensor(obs, dtype=torch.float)
#         obs = obs.reshape(B*I*C, R, T)
#         obs = self.conv1d_step1(obs)
#         obs = obs.reshape(B*I, C, -1)
#         obs = self.conv1d_step2(obs)
#         obs = obs.reshape(B, -1)
#         logits = self.final_step(obs)
#         return logits, state
