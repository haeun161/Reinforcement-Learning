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
    
        
        if isinstance(state, dict):
            position = state["position"]
            velocity = state["velocity"]
        else:
            position = state[0]
            velocity = state[1]
            
        print("position:",type(state[0]), state[0])
        print("velocity:",type(state[1]), state[1])
    
        featurized = tiles(self.iht, self.num_tilings, 
                           [self.position_scale * position, 
                            self.velocity_scale * velocity], 
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

"=========sarsa_n========="
def sarsa_n(n, env, estimator, gamma=1.0, epsilon=0):
    """
    n-step semi-gradient Sarsa algorithm
    for finding optimal q and pi via Linear
    FA with n-step TD updates.
    """
    
    # Create epsilon-greedy policy
    policy = make_epsilon_greedy_policy(
        estimator, epsilon, env.action_space.n)

    # Reset the environment and pick the first action
    state = env.reset()
    action_probs = policy(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    # Set up trackers
    states = [state]
    actions = [action]
    rewards = [0.0]

    # Step through episode
    T = float('inf')
    for t in itertools.count():
        if t < T:           
            # Take a step
            next_state, reward, done, _ = env.step(action)
            states.append(next_state)
            rewards.append(reward)

            if done:
                T = t + 1

            else:
                # Take next step
                next_action_probs = policy(next_state)
                next_action = np.random.choice(
                    np.arange(len(next_action_probs)), p=next_action_probs)

                actions.append(next_action)

        update_time = t + 1 - n  # Specifies state to be updated
        if update_time >= 0:       
            # Build target
            target = 0
            for i in range(update_time + 1, min(T, update_time + n) + 1):
                target += np.power(gamma, i - update_time - 1) * rewards[i]
            if update_time + n < T:
                q_values_next = estimator.predict(states[update_time + n])
                target += q_values_next[actions[update_time + n]]
            
            # Update step
            estimator.update(states[update_time], actions[update_time], target)
        
        if update_time == T - 1:
            break

        state = next_state
        action = next_action
    
    ret = np.sum(rewards)
    
    return t, ret

"=========Plot========="
def plot_learning_curves(stats, smoothing_window=10):
    """
    Plots the number of steps taken by the agent
    to solve the task as a function of episode number,
    smoothed over the last smoothing_window episodes. 
    """
    
    plt.figure(figsize=(10,5))
    for algo_stats in stats:
        steps_per_episode = pd.Series(algo_stats.steps).rolling(
            smoothing_window).mean()  # smooth
        plt.plot(steps_per_episode, label=algo_stats.algorithm)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps per Episode")
    plt.legend()
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


"=========Run Sarsa_n========="

step_size = 0.5  # Fraction of the way we want to move towards target
n = 1  # Level of bootstrapping (set to intermediate value)
num_episodes = 300

estimator_n = QEstimator(step_size=step_size)

start_time = timeit.default_timer()
#def sarsa_n(n, env, estimator, gamma=1.0, epsilon=0):
run_stats_lambda = run('sarsa_n', num_episodes, n=n, env=env, estimator=estimator_n)
elapsed_time = timeit.default_timer() - start_time

plot_learning_curves([run_stats_lambda], 10)
print('{} episodes completed in {:.2f}s'.format(num_episodes, elapsed_time))