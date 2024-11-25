#!/usr/bin/env python3
"""a function that performs SARSA(λ)"""
import numpy as np


def sarsa_lambtha(
        env,
        Q,
        lambtha,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1,
        min_epsilon=0.1,
        epsilon_decay=0.05):
    """Returns: Q, the updated Q table"""
    def eps_greedy(state, Q, epsilon):
        """selects action using epsilon greedy policy"""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, Q.shape[1])
        else:
            return np.argmax(Q[state])
    epsi = epsilon
    for ep in range(episodes):
        state, _ = env.reset()
        action = eps_greedy(state, Q, epsilon)
        eligibility_trace = np.zeros_like(Q)
        for step in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)
            next_action = eps_greedy(next_state, Q, epsilon)
            td_error = reward + gamma * \
                Q[next_state, next_action] - Q[state, action]
            eligibility_trace[state, action] += 1
            eligibility_trace *= gamma * lambtha
            Q += alpha * td_error * eligibility_trace
            if done:
                break
            state = next_state
            action = next_action
        epsilon = min_epsilon + (epsi - min_epsilon) * \
            np.exp(-epsilon_decay * ep)
    return Q
