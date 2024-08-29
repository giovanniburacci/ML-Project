import gym
from algorithm import rollouts
from gym.wrappers import TimeLimit
import policies
import algorithm
taxi_env = gym.make("Taxi-v3", render_mode='ansi')
taxi_env = TimeLimit(taxi_env, max_episode_steps=50)

print("Observation space", taxi_env.observation_space)
print("Action space", taxi_env.action_space)

avg_return = rollouts(
    env=taxi_env,
    policy=policies.RandomPolicy(taxi_env.action_space.n),
    gamma=0.95,
    n_episodes=5,
    render=True
)

print("Average return", avg_return)


qtable = algorithm.qlearn(env=taxi_env, alpha0=0.1, gamma=0.95, max_steps=200000)
policy_obj = policies.GreedyPolicy(qtable)
greedy_policy = {obs: policy_obj(obs) for obs in range(taxi_env.observation_space.n)}
print(greedy_policy)
avg_return = rollouts(env=taxi_env, policy=policies.GreedyPolicy(qtable), gamma=0.95, n_episodes=20)
print("Average return", avg_return)
