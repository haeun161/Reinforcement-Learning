#import gym #강화 학습 환경을 제공하는 라이브러리
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

from collections import defaultdict
#from gym.envs.toy_text import CliffWalkingEnv
from envs.cliff_walking import CliffWalkingEnv
import matplotlib.pyplot as plt
import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * (epsilon / nA) #각 행동에 대한 초기 확률을 설정
        best_action = np.argmax(Q[observation]) #가장 높은 Q값 반환
        A[best_action] += (1.0 - epsilon) #해당 행동이 선택될 확률을 증가
        return A
    return policy_fn

def n_step_sarsa(env, num_episodes, n=3, discount_factor=1.0, alpha=0.5, epsilon=0.1): #off_policy
    """
    Off-policy n-step SARSA algorithm: TD control with behavior and target policies.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        n: The number of steps to look ahead.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action in the behavior policy.
    
    Returns:
        A tuple (Q, stats).
        Q is the action-value function learned under the target policy, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # Action-value function
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )

    # Behavior policy (epsilon-greedy) and target policy (greedy)
    behavior_policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    target_policy = lambda state: np.argmax(Q[state])  # deterministic greedy policy
    
    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Initialize the episode
        state = env.reset()
        action_probs = behavior_policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # Initialize lists to store the rewards, states, and actions
        rewards = [0]
        states = [state]
        actions = [action]
        
        T = float('inf')  # end of episode
        t = 0  # time step
        
        while True:
            if t < T:
                # Take a step and store the transition
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                states.append(next_state)
                
                if done:
                    T = t + 1
                else:
                    next_action_probs = behavior_policy(next_state)
                    next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
                    actions.append(next_action)
                    action = next_action
            
            tau = t - n + 1
            if tau >= 0:
                G = sum([discount_factor**(i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1)])
                
                if tau + n < T:
                    target_action = target_policy(states[tau + n])
                    G += discount_factor**n * Q[states[tau + n]][target_action]
                
                # Calculate importance sampling ratio
                rho = 1.0
                for k in range(tau + 1, min(tau + n, T)):
                    if actions[k] == target_policy(states[k]):
                        rho *= 1 / (1 - epsilon + epsilon / env.action_space.n)
                    else:
                        rho *= 0  # if a non-greedy action was taken, ignore this update
                
                # Update Q with importance sampling
                if rho != 0:
                    state_tau, action_tau = states[tau], actions[tau]
                    Q[state_tau][action_tau] += alpha * rho * (G - Q[state_tau][action_tau])
                
                # Update statistics
                stats.episode_rewards[i_episode] += rewards[tau + 1]
                stats.episode_lengths[i_episode] = t
            
            if tau == T - 1:
                break
            
            t += 1
    
    return Q, stats


# 실행
Q, stats = n_step_sarsa(env, num_episodes=500, n=3)
plotting.plot_episode_stats(stats)

Q, stats = n_step_sarsa(env, 500)

plotting.plot_episode_stats(stats)