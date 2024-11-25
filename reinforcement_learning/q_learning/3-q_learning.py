#!/usr/bin/env python3
"""a function that performs Q-learning"""
import numpy as np


def train(
        env,
        Q,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1,
        min_epsilon=0.1,
        epsilon_decay=0.05):
    """Returns: Q, total_rewards"""
    epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            if reward == 0 and done:
                reward = -1
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            ep_reward += reward
            state = next_state
            if done:
                break
        total_rewards.append(ep_reward)
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    return Q, total_rewards
