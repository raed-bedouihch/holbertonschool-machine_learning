#!/usr/bin/env python3
"""a function that loads the pre-made FrozenLakeEnv
evnironment from gymnasium"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Returns: the environment"""
    return gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi")
