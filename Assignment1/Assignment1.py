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

# 목적: 에이전트가 시작 지점에서 목표 지점으로 이동하는 것이며, 절벽(위험 지역)을 피해야 합니다.
# 상태 공간: 격자 맵의 각 위치를 상태로 나타내며, 각 위치에 대한 보상이 다릅니다.
# 행동 공간: 에이전트가 이동할 수 있는 방향(상, 하, 좌, 우)입니다.
# 보상: 목표 지점에 도착하면 긍정적인 보상을 받고, 절벽에 떨어지면 부정적인 보상을 받습니다

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
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
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
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
    return Q, stats

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Keeps track of useful statistics
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
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # One step in the environment
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)
            
            # Pick the next action
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
    
            if done:
                break
                
            action = next_action
            state = next_state        
    
    return Q, stats

def expected_sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Expected-SARSA algorithm: Off-policy TD control. Uses ecpected value over nect state-action pairs
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

# 보상 그래프를 한 번에 그리기 위한 함수 작성
def plot_comparison(stats_q_learning, stats_sarsa, stats_expected_sarsa, smoothing_window=10):
    # 보상 그래프를 비교하기 위한 Figure 설정
    plt.figure(figsize=(10, 6))

    # Q-Learning 보상
    rewards_smoothed_q_learning = pd.Series(stats_q_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed_q_learning, label="Q-Learning")

    # SARSA 보상
    rewards_smoothed_sarsa = pd.Series(stats_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed_sarsa, label="SARSA")

    # Expected SARSA 보상
    rewards_smoothed_expected_sarsa = pd.Series(stats_expected_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed_expected_sarsa, label="Expected SARSA")

    # 그래프 세부 설정
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards (Smoothed) during episode")
    plt.title("Comparison of Reinforcement Learning Methods")
    plt.legend(loc="best")

    # 그래프 출력
    plt.show()


# 각 학습 방법의 결과를 저장
Q_q_learning, stats_q_learning = q_learning(env, 500)
Q_sarsa, stats_sarsa = sarsa(env, 500)
Q_expected_sarsa, stats_expected_sarsa = expected_sarsa(env, 500)

# 그래프 출력 함수 호출
plot_comparison(stats_q_learning, stats_sarsa, stats_expected_sarsa)