#!/usr/bin/env python3
"""a function that has the trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """Returns: The total rewards for the episode and
    a list of rendered outputs representing
    the board state at each step."""
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []
    rendered_outputs.append(env.render())
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        rendered_outputs.append(env.render())
        state = next_state
        total_rewards += reward
        if done:
            break
    return total_rewards, rendered_outputs
