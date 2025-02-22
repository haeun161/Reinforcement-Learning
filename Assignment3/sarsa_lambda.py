import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import time
import timeit
from collections import namedtuple
import os
import glob
import timeit
import numpy as np
import matplotlib.pyplot as plt

from lib.tile_coding import IHT, tiles
from matplotlib import pyplot as plt
from matplotlib import cm
matplotlib.style.use('ggplot')

import io
import base64
from IPython.display import HTML

env = gym.make("MountainCar-v0")
env._max_episode_steps = 1000  # Increase upper time limit so we can plot full behaviour.
np.random.seed(6)  # Make plots reproducible


class QEstimator():
    """
    Linear action-value (q-value) function approximator for 
    semi-gradient methods with state-action featurization via tile coding. 
    """
    
    def __init__(self, step_size, num_tilings=8, max_size=4096,
                 tiling_dim=None, trace=False):
        
        self.trace = trace
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.tiling_dim = tiling_dim or num_tilings

        # Step size is interpreted as the fraction of the way we want 
        # to move towards the target. To compute the learning rate alpha,
        # scale by number of tilings. 
        self.alpha = step_size / num_tilings #learning rate
        
        # Initialize index hash table (IHT) for tile coding.
        # This assigns a unique index to each tile up to max_size tiles.
        # Ensure max_size >= total number of tiles (num_tilings x tiling_dim x tiling_dim)
        # to ensure no duplicates.
        self.iht = IHT(max_size)

        # Initialize weights (and optional trace)
        self.weights = np.zeros(max_size)
        if self.trace:
            self.z = np.zeros(max_size)

        # Tilecoding software partitions at integer boundaries, so must rescale
        # position and velocity space to span tiling_dim x tiling_dim region.
        self.position_scale = self.tiling_dim / (env.observation_space.high[0] \
                                                  - env.observation_space.low[0])
        self.velocity_scale = self.tiling_dim / (env.observation_space.high[1] \
                                                  - env.observation_space.low[1])
        
    def featurize_state_action(self, state, action):
        """
        Returns the featurized representation for a 
        state-action pair.
        """
        featurized = tiles(self.iht, self.num_tilings, 
                           [self.position_scale * state[0], 
                            self.velocity_scale * state[1]], 
                           [action])
        return featurized
    
    def predict(self, s, a=None):
        """
        Predicts q-value(s) using linear FA.
        If action a is given then returns prediction
        for single state-action pair (s, a).
        Otherwise returns predictions for all actions 
        in environment paired with s.   
        """
    
        if a is None:
            features = [self.featurize_state_action(s, i) for 
                        i in range(env.action_space.n)]
        else:
            features = [self.featurize_state_action(s, a)]
            
        return [np.sum(self.weights[f]) for f in features]
    
    def update(self, s, a, target):
        """
        Updates the estimator parameters
        for a given state and action towards
        the target using the gradient update rule 
        (and the eligibility trace if one has been set).
        """
        features = self.featurize_state_action(s, a)
        estimation = np.sum(self.weights[features])  # Linear FA
        delta = (target - estimation)
        
        if self.trace:
            # self.z[features] += 1  # Accumulating trace
            self.z[features] = 1  # Replacing trace
            self.weights += self.alpha * delta * self.z
        else:
            self.weights[features] += self.alpha * delta
    
    def reset(self, z_only=False):
        """
        Resets the eligibility trace (must be done at 
        the start of every epoch) and optionally the
        weight vector (if we want to restart training
        from scratch).
        """
        
        if z_only:
            assert self.trace, 'q-value estimator has no z to reset.'
            self.z = np.zeros(self.max_size)
        else:
            if self.trace:
                self.z = np.zeros(self.max_size)
            self.weights = np.zeros(self.max_size)
            
"========epsilon_greedy=========="
def make_epsilon_greedy_policy(estimator, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based on a 
    given q-value approximator and epsilon.    
    """
    def policy_fn(observation):
        action_probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = estimator.predict(observation)
        best_action_idx = np.argmax(q_values)
        action_probs[best_action_idx] += (1.0 - epsilon)
        return action_probs
    return policy_fn

"=========sarsa_lambda========="

def sarsa_lambda(lmbda, env, estimator, gamma=1.0, epsilon=0):
    
    """
    Sarsa(Lambda) algorithm
    for finding optimal q and pi via Linear
    FA with eligibility traces.
    """
    
    # Reset the eligibility trace
    estimator.reset(z_only=True)

    # Create epsilon-greedy policy
    policy = make_epsilon_greedy_policy(
        estimator, epsilon, env.action_space.n)

    # Reset the environment and pick the first action
    #state = env.reset()
    state, _ = env.reset()
    action_probs = policy(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    ret = 0
    # Step through episode
    for t in itertools.count():
        # Take a step
        # next_state, reward, done, _ = env.step(action)
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        ret += reward

        if done:
            target = reward
            estimator.update(state, action, target)
            break

        else:
            # Take next step
            next_action_probs = policy(next_state)
            next_action = np.random.choice(
                np.arange(len(next_action_probs)), p=next_action_probs)

            # Estimate q-value at next state-action
            q_new = estimator.predict(
                next_state, next_action)[0]
            target = reward + gamma * q_new
            # Update step
            estimator.update(state, action, target)
            estimator.z *= gamma * lmbda
        
        # Track the values of z and w for plotting
        # z_values.append(estimator.z.copy())  # Copy current eligibility trace
        # w_values.append(estimator.weights.copy())  # Copy current weights

        state = next_state
        action = next_action    
    
    return t, ret

"=========Plot========="
# 각 실험에 대한 결과가 저장된 run_stats_list와 tile_sizes
def plot_steps_per_episode(run_stats_list, smoothing_window=10):
    plt.figure(figsize=(10, 5))

    # 각 타일 크기별로 결과를 시각화
    for run_stats, tile_size in run_stats_list:
        # 에피소드별 스텝 수를 구하고 롤링 평균으로 부드럽게 처리
        steps_per_episode = pd.Series(run_stats.steps).rolling(smoothing_window).mean()
        
        # 타일 크기별로 레이블 설정
        plt.plot(steps_per_episode, label=f'Tile Size {tile_size}')
    
    # 그래프 꾸미기
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps per Episode for Different Tile Sizes")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# 타일 사이즈 별 리워드 곡선 그리기
def plot_returns(run_stats_list, window_size=10):
    plt.figure(figsize=(10, 6))
    for run_stats, tile_size in run_stats_list:
        plt.plot(run_stats.steps, label=f'Tile Size {tile_size}')
    
    plt.xlabel('Episode')
    plt.ylabel('Returns')
    plt.title('Learning Curves for Different Tile Sizes')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
"=========Running Func========="
RunStats = namedtuple('RunStats', ['algorithm', 'steps', 'returns'])
def run(algorithm, num_episodes=300, **algorithm_kwargs):
    """
    Runs algorithm over multilple episodes and logs
    for each episode the complete return (G_t) and the
    number of steps taken.
    """
    
    stats = RunStats(
        algorithm=algorithm, 
        steps=np.zeros(num_episodes), 
        returns=np.zeros(num_episodes))
    
    algorithm_fn = globals()[algorithm]
    
    for i in range(num_episodes):
        episode_steps, episode_return = algorithm_fn(**algorithm_kwargs)
        stats.steps[i] = episode_steps
        stats.returns[i] = episode_return
        sys.stdout.flush()
        print("\rEpisode {}/{} Return {}".format(
            i + 1, num_episodes, episode_return), end="")
    return stats

"=========Run Sarsa_lamda========="

# # QEstimator 초기화
# step_size = 0.5  # 학습률
# lmbda = 0.9  # Eligibility trace의 람다 값
# num_episodes = 300

# # 각 실험을 위한 결과를 저장할 리스트
# run_stats_list = []

# # 타일 크기별로 실험을 실행
# tile_sizes = [4, 8, 12]
# for tile_size in tile_sizes:
#     estimator = QEstimator(step_size=step_size, num_tilings=tile_size, trace=True)
    
#     start_time = timeit.default_timer()
#     run_stats = run(
#         'sarsa_lambda', 
#         num_episodes=num_episodes, 
#         lmbda=lmbda, 
#         env=env, 
#         estimator=estimator
#     )
#     elapsed_time = timeit.default_timer() - start_time
#     print('{} episodes completed for num_tilings={}: {:.2f}s'.format(num_episodes, tile_size, elapsed_time))
    
#     run_stats_list.append((run_stats, tile_size))


# plot_steps_per_episode(run_stats_list, 10)
# plot_returns(run_stats_list, 10)


"=========Run 2========="

import matplotlib.pyplot as plt

# Add these two lists to store the values of z and w
z_values = []
w_values = []

run_stats_list=[]

# Running the experiment with the modified tracking
num_episodes = 300
step_size = 0.5
lmbda = 0.9
tile_sizes = [4, 8, 12]

for tile_size in tile_sizes:
    estimator = QEstimator(step_size=step_size, num_tilings=tile_size, trace=True)
    
    start_time = timeit.default_timer()
    run_stats = run(
        'sarsa_lambda', 
        num_episodes=num_episodes, 
        lmbda=lmbda, 
        env=env, 
        estimator=estimator
    )
    elapsed_time = timeit.default_timer() - start_time
    print('{} episodes completed for num_tilings={}: {:.2f}s'.format(num_episodes, tile_size, elapsed_time))
    
    run_stats_list.append((run_stats, tile_size))
    z_values.append((run_stats, z_values))
    w_values.append((run_stats, w_values))

# After all episodes, plot z and w values
plt.figure(figsize=(12, 6))

# Plot z (eligibility trace)
plt.subplot(1, 2, 1)
plt.title("Short-term Memory Vector (z) Over Episodes")
plt.plot(np.array(z_values), alpha=0.5)
plt.xlabel("Step")
plt.ylabel("Value of z")

# Plot w (weight vector)
plt.subplot(1, 2, 2)
plt.title("Long-term Weight Vector (w) Over Episodes")
plt.plot(np.array(w_values), alpha=0.5)
plt.xlabel("Step")
plt.ylabel("Value of w")

plt.tight_layout()
plt.show()
