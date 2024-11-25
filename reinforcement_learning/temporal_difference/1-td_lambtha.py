#!/usr/bin/env python3
"""a function that performs the TD(λ) algorithm"""
import numpy as np


def td_lambtha(
        env,
        V,
        policy,
        lambtha,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99):
    """Returns: V, the updated value estimate"""
    for ep in range(episodes):
        state, _ = env.reset()
        eligibility_trace = np.zeros_like(V)
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            td_error = reward + gamma * V[next_state] - V[state]
            eligibility_trace[state] += 1
            V += alpha * td_error * eligibility_trace
            eligibility_trace *= gamma * lambtha
            state = next_state
            if done:
                break
    return V
