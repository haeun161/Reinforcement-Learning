import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from collections import namedtuple
from matplotlib import pyplot as plt
from lib.tile_coding import IHT, tiles

# Set matplotlib style
matplotlib.style.use('ggplot')

# Environment setup
env = gym.make("MountainCar-v0")
env._max_episode_steps = 1000
np.random.seed(6)  # Reproducible results

class QEstimator:
    """
    Linear action-value (q-value) function approximator for
    semi-gradient methods with state-action featurization via tile coding.
    """

    def __init__(self, step_size, num_tilings=8, max_size=4096, tiling_dim=None, trace=False):
        self.trace = trace
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.tiling_dim = tiling_dim or num_tilings

        # Step size for learning rate
        self.alpha = step_size / num_tilings

        # Initialize tile coding and weights
        self.iht = IHT(max_size)
        self.weights = np.zeros(max_size)
        self.z = np.zeros(max_size) if trace else None

        # Scaling factors for tile coding
        self.position_scale = self.tiling_dim / (env.observation_space.high[0] - env.observation_space.low[0])
        self.velocity_scale = self.tiling_dim / (env.observation_space.high[1] - env.observation_space.low[1])

    def featurize_state_action(self, state, action):
        return tiles(self.iht, self.num_tilings, [
            self.position_scale * state[0],
            self.velocity_scale * state[1]
        ], [action])

    def predict(self, state, action=None):
        if action is None:
            features = [self.featurize_state_action(state, a) for a in range(env.action_space.n)]
        else:
            features = [self.featurize_state_action(state, action)]
        return [np.sum(self.weights[f]) for f in features]

    def update(self, state, action, target):
        features = self.featurize_state_action(state, action)
        estimation = np.sum(self.weights[features])
        delta = target - estimation

        if self.trace:
            self.z[features] = 1  # Replacing trace
            self.weights += self.alpha * delta * self.z
        else:
            self.weights[features] += self.alpha * delta

    def reset(self, z_only=False):
        if z_only and self.trace:
            self.z = np.zeros_like(self.z)
        elif not z_only:
            self.weights = np.zeros_like(self.weights)
            if self.trace:
                self.z = np.zeros_like(self.z)

def make_epsilon_greedy_policy(estimator, epsilon, num_actions):
    def policy_fn(observation):
        action_probs = np.ones(num_actions) * epsilon / num_actions
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        action_probs[best_action] += (1.0 - epsilon)
        return action_probs
    return policy_fn

def sarsa_lambda(lmbda, env, estimator, gamma=1.0, epsilon=0):
    """
    Sarsa(Lambda) algorithm
    """
    # Reset the eligibility trace
    estimator.reset(z_only=True)

    # Create epsilon-greedy policy
    policy = make_epsilon_greedy_policy(
        estimator, epsilon, env.action_space.n)

    # Reset the environment and pick the first action
    state, _ = env.reset()
    action_probs = policy(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    ret = 0
    # Track W and Z updates for this episode
    w_history = [estimator.weights.copy()]
    z_history = [estimator.z.copy()]
    
    # Step through episode
    for t in itertools.count():
        # Take a step
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        ret += reward

        if done:
            target = reward
            estimator.update(state, action, target)
            w_history.append(estimator.weights.copy())
            z_history.append(estimator.z.copy())
            break
        else:
            # Take next step
            next_action_probs = policy(next_state)
            next_action = np.random.choice(
                np.arange(len(next_action_probs)), p=next_action_probs)

            # Estimate q-value at next state-action
            q_new = estimator.predict(next_state, next_action)[0]
            target = reward + gamma * q_new
            
            # Update step
            estimator.update(state, action, target)
            estimator.z *= gamma * lmbda

        w_history.append(estimator.weights.copy())
        z_history.append(estimator.z.copy())

        state = next_state
        action = next_action    
    
    return t, ret, w_history, z_history


def plot_steps_per_episode(run_stats_list, smoothing_window=10):
    plt.figure(figsize=(10, 5))
    for run_stats, tile_size in run_stats_list:
        steps_per_episode = pd.Series(run_stats.steps).rolling(smoothing_window).mean()
        plt.plot(steps_per_episode, label=f'Tile Size {tile_size}')
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps per Episode for Different Tile Sizes")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_returns(run_stats_list, window_size=10):
    plt.figure(figsize=(10, 6))
    for run_stats, tile_size in run_stats_list:
        returns_smoothed = pd.Series(run_stats.returns).rolling(window_size).mean()
        plt.plot(returns_smoothed, label=f'Tile Size {tile_size}')
    plt.xlabel("Episode")
    plt.ylabel("Returns")
    plt.title("Learning Curves for Different Tile Sizes")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

RunStats = namedtuple('RunStats', ['algorithm', 'steps', 'returns'])

def run(algorithm, num_episodes=300, **algorithm_kwargs):
    stats = RunStats(
        algorithm=algorithm,
        steps=np.zeros(num_episodes),
        returns=np.zeros(num_episodes)
    )

    algorithm_fn = globals()[algorithm]

    for i in range(num_episodes):
        episode_steps, episode_return, *_ = algorithm_fn(**algorithm_kwargs)
        stats.steps[i] = episode_steps
        stats.returns[i] = episode_return
        sys.stdout.flush()
        print(f"\rEpisode {i + 1}/{num_episodes} Return {episode_return}", end="")

    return stats

# Example of running SARSA(lambda) and plotting
if __name__ == "__main__":
    tile_sizes = [8, 16, 32]
    num_episodes = 300
    step_size = 0.1
    lambda_val = 0.9
    gamma = 1.0
    epsilon = 0.1

    run_stats_list = []

    for tile_size in tile_sizes:
        estimator = QEstimator(step_size=step_size, num_tilings=tile_size, trace=True)
        stats = run(
            algorithm="sarsa_lambda",
            num_episodes=num_episodes,
            lmbda=lambda_val,
            env=env,
            estimator=estimator,
            gamma=gamma,
            epsilon=epsilon
        )
        run_stats_list.append((stats, tile_size))

    plot_steps_per_episode(run_stats_list)
    plot_returns(run_stats_list)
