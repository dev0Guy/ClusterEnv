# # import gymnasium as gym
# # import _tmo
# # import logging
# #
# #
# # def main():
# #     logging.basicConfig(level=logging.INFO)
# #     env = _tmo.wrappers.ConcatenateObservationDict(
# #         gym.make('cluster-v0', render_mode='rgb_array', max_episode_steps=10, cooldown=0.01)
# #     )
# #     obs, _ = env.reset()
# #     for _ in range(100):
# #         action = env.action_space.sample() if _ % 5 != 0 else 0
# #         observation, reward, terminate, truncated, info = env.step(action)
# #         print('Reward: ', reward)
# #         env.reset()
# #     env.close()
# #
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
# from gymnasium.wrappers.record_video import RecordVideo
# from torch.utils.tensorboard import SummaryWriter
# from tianshou.utils import WandbLogger
# from network import Net, CnnNetwork
# import gymnasium as gym
# import tianshou as ts
# import _tmo
# import torch
#
#
# WANDB_PROJECT: str = 'Thesis'
# LOG_FOLDER = './logs'
# ENV_NAME: str = 'cluster-v0'
#
#
# def create_env(render_mode=None) -> gym.Env:
#     return _tmo.wrappers.ConcatenateObservationDict(
#         gym.make(
#             ENV_NAME,
#             render_mode=render_mode,
#             n_nodes=1,
#             n_jobs=4,
#             n_resource=1,
#             cooldown=1e-5,
#             max_episode_steps=20,
#             max_time=3,
#         )
#     )
#
#
# def main():
#     logger: WandbLogger = WandbLogger(project=WANDB_PROJECT)
#     writer: SummaryWriter = SummaryWriter(LOG_FOLDER)
#     logger.load(writer=writer)
#     train_env: gym.Env = create_env('')
#     test_env: gym.Env = create_env('')
#     state_shape = train_env.observation_space.shape or train_env.observation_space.n
#     action_shape = train_env.action_space.shape or train_env.action_space.n
#     net = Net(state_shape, action_shape)
#     optim = torch.optim.Adam(net.parameters(), lr=1e-4)
#     policy = ts.policy.DQNPolicy(
#         model=net,
#         optim=optim,
#         action_space=train_env.action_space,
#         discount_factor=0.9,
#         target_update_freq=50
#     )
#     train_collector = ts.data.Collector(
#         policy,
#         train_env,
#         ts.data.VectorReplayBuffer(10_000, 20),
#         exploration_noise=True
#     )
#     for _ in range(5):
#         test_collector = ts.data.Collector(
#             policy,
#             test_env,
#             exploration_noise=True,
#         )
#         trainer = ts.trainer.OffpolicyTrainer(
#             policy=policy,
#             train_collector=train_collector,
#             test_collector=test_collector,
#             max_epoch=50,
#             step_per_epoch=100,
#             step_per_collect=1_000,
#             update_per_step=0.05,
#             episode_per_test=500,
#             batch_size=32,
#             train_fn=lambda epoch, env_step: policy.set_eps(0.3),
#             test_fn=lambda epoch, env_step: policy.set_eps(0.02),
#             # stop_fn=lambda mean_rewards: mean_rewards > 0,
#             # logger=logger,
#         )
#         trainer.run()
#         test_collector = ts.data.Collector(
#             policy,
#             create_env('rgb_array'),
#             exploration_noise=True,
#         )
#         test_policy = policy  # Use the same policy for testing
#         test_trainer = ts.trainer.OffpolicyTrainer(
#             policy=test_policy,
#             train_collector=test_collector,
#             max_epoch=1,  # Only run testing for 1 epoch
#             step_per_epoch=100,  # Adjust as needed
#             step_per_collect=1_000,  # Adjust as needed
#             update_per_step=0,  # No updates during testing
#             episode_per_test=1_000,  # Adjust as needed
#             batch_size=32,
#             train_fn=None,  # No training function needed for testing
#             test_fn=None,  # No testing function needed for testing
#             # logger=logger,
#         )
#         test_trainer.run()
#
#
# if __name__ == "__main__":
#     main()
