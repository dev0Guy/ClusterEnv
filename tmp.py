from cluster_env.envs.cluster import ClusterEnv
import logging

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    env = ClusterEnv(nodes=3, jobs=5, resource=3, time=3, max_episode_steps=10_000, render_mode='rgb_array')
    env.reset()
    for idx in range(10_000):
        # action =env.action_space.sample()
        action = env.action_space.sample() if idx % 2 == 0 else 0
        obs, reward, terminate, truncated, _ = env.step(action)
        if terminate:
            print(obs['Status'])
            print('Here', idx)
            break
