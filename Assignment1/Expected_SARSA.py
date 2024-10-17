import gym #강화 학습 환경을 제공하는 라이브러리
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

def expected_sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n)) #Q-값이 0으로 초기화

    # Keeps track of useful statistics #에피소드에 대한 보상과 길이를 기록하는 배열 초기환
    stats = plotting.EpisodeStats( 
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset() 
        #print(state) #(36, {'prob': 1})
        
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count(): #에피소드의 각 타임 스텝을 카운트
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs) #행동을 선택
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            # 기대값 계산
            action_probs_next = policy(next_state)  # 다음 상태에서의 행동 확률
            expectation = np.dot(action_probs_next, Q[next_state])  # 기대값 계산

            # TD 목표값 및 Q 업데이트
            td_target = reward + discount_factor * expectation
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
    return Q, stats

Q, stats = expected_sarsa(env, 500)

plotting.plot_episode_stats(stats)