from collections import defaultdict
import gym
import numpy as np
import policies


def qlearn(
        env: gym.Env,
        alpha0: float,
        gamma: float,
        max_steps: int
):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = policies.EpsilonGreedyPolicy(Q)
    obs = 0
    terminated = True

    for step in range(max_steps):

        if terminated:
            env.reset()

        eps = (max_steps - step) / max_steps
        action = policy(obs, eps)

        obs2, rew, terminated, truncated, info = env.step(action)

        Q[obs][action] += alpha0 * (rew + gamma * np.max(Q[obs2]) - Q[obs][action])
        obs = obs2
    return Q

def rollouts(
        env: gym.Env,
        policy,
        gamma: float,
        n_episodes: int,
        render=False
) -> float:
    sum_returns = 0.0

    terminated = False
    obs = env.reset()
    discounting = 1
    ep = 0
    if render:
        env.render()

    while True:

        if terminated:
            if render:
                print("New episode")
            obs = env.reset()
            discounting = 1
            ep += 1
            if ep >= n_episodes:
                break

        action = policy(obs)

        obs, rew, terminated, truncated, info = env.step(action)

        sum_returns += rew * discounting
        discounting *= gamma

        if render:
            env.render()

    return sum_returns/n_episodes









