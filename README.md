# Reinforcement-Learning
강화학습을 공부하고, 코드를 구현해보는 학습공간입니다

- **Assignment1**: Temporal-Difference Learning(TD(0))의 SARSA, Q-Learning, Expected-SARSA를 활용하여 Cliff-Walking 문제를 해결하고, 결과 분석
- **Assignment2**:  Cliff-Walking 문제에 off-policy n-step Sarsa 추가

# 중요 부분

### 요약:
- **SARSA**: 다음 상태에서 실제로 선택한 행동을 사용하여 업데이트.
- **Q-Learning**: 다음 상태에서 가장 큰 Q-value를 선택하여 업데이트.
- **Expected-SARSA**: 다음 상태에서의 모든 행동에 대해 정책에 따른 기대값을 사용하여 업데이트.

### 1. **SARSA (State-Action-Reward-State-Action)**
SARSA는 현재 상태와 행동에 대한 Q-value를 업데이트합니다. 즉, 새로운 상태에서 취할 행동을 예측하여 업데이트합니다.

**상태 업데이트 공식**:
\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right)
\]
- \( Q(s_t, a_t) \): 현재 상태 \( s_t \)와 행동 \( a_t \)에 대한 Q-value
- \( r_{t+1} \): 보상
- \( \gamma \): 할인율
- \( Q(s_{t+1}, a_{t+1}) \): 다음 상태 \( s_{t+1} \)에서 취한 행동 \( a_{t+1} \)의 Q-value
- \( \alpha \): 학습률

### 2. **Q-Learning**
Q-Learning은 오프-폴리시 알고리즘으로, 최적 행동을 선택하여 Q-value를 업데이트합니다. 현재 상태에서의 행동을 기반으로, 최대 보상을 얻을 수 있는 행동을 선택하여 업데이트합니다.

**상태 업데이트 공식**:
\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right)
\]
- \( \max_{a'} Q(s_{t+1}, a') \): 다음 상태 \( s_{t+1} \)에서 가능한 모든 행동 중 최적의 행동에 대한 Q-value

### 3. **Expected-SARSA**
Expected-SARSA는 SARSA의 변형으로, 현재 상태에서의 가능한 모든 행동에 대해 기대값을 계산하여 Q-value를 업데이트합니다. 이 방법은 정책을 확률적으로 따라가며 업데이트합니다.

**상태 업데이트 공식**:
\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_{t+1} + \gamma \sum_{a'} \pi(a'|s_{t+1}) Q(s_{t+1}, a') - Q(s_t, a_t) \right)
\]
- \( \sum_{a'} \pi(a'|s_{t+1}) Q(s_{t+1}, a') \): 다음 상태 \( s_{t+1} \)에서 가능한 모든 행동에 대한 기대값을 합산
- \( \pi(a'|s_{t+1}) \): 다음 상태에서 각 행동을 선택할 확률 (정책)

