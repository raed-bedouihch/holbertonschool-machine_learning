#!/usr/bin/env python3
"""a function that performs the Monte Carlo algorithm"""
import numpy as np


def monte_carlo(
        env,
        V,
        policy,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99):
    """Returns: V, the updated value estimate"""
    for ep in range(episodes):
        state, _ = env.reset()
        ep_hist = []
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            ep_hist.append((state, reward))
            if done:
                break
            state = next_state

        g = 0
        ep_hist = np.array(ep_hist, dtype=int)
        for state, reward in reversed(ep_hist):
            g = reward + gamma * g
            if state not in ep_hist[:ep, 0]:
                V[state] += alpha * (g - V[state])
    return V
