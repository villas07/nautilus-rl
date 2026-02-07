"""
Exit Gate Environment — Ticket 6

gym.Wrapper around NautilusBacktestEnv that restricts the RL agent
to EXIT decisions only (HOLD vs CLOSE).  Entry decisions (LONG/SHORT)
are delegated to a DeterministicBaseline (MA20/MA50 cross).

Action space: Discrete(2)
    0 = HOLD  (keep current position or let baseline act when flat)
    1 = CLOSE (close current position)

When the inner env is FLAT (position == 0):
    → The baseline decides the entry action (HOLD / LONG / SHORT).
    → The RL action is ignored.

When the inner env is in a position (LONG or SHORT):
    → The baseline is ignored.
    → RL decides: 0 → inner HOLD (action 0), 1 → inner CLOSE (action 1).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ExitGateEnv(gym.Wrapper):
    """
    Wrapper that converts a 4-action NautilusBacktestEnv into a 2-action
    exit-only environment.

    The wrapper owns a baseline policy instance that controls entries.
    The RL agent only controls exits.
    """

    def __init__(self, env: gym.Env, baseline):
        """
        Args:
            env: Inner NautilusBacktestEnv (Discrete(4) action space).
            baseline: A policy with .act(obs, env) -> int and .reset().
                      Expected to be DeterministicBaseline from evaluate_policies.
        """
        super().__init__(env)

        self.baseline = baseline

        # Override action space to 2 actions: HOLD(0), CLOSE(1)
        self.action_space = spaces.Discrete(2)

        # Observation space stays the same as inner env (36 features)
        # gym.Wrapper already delegates observation_space to self.env

    def reset(self, *, seed=None, options=None):
        """Reset inner env and baseline."""
        self.baseline.reset()
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        """
        Route action to inner env based on position state.

        If FLAT:  baseline decides entry → pass to inner env, RL action ignored.
        If in position: RL decides exit → map 0->HOLD(0), 1->CLOSE(1).
        """
        action = int(action)

        # Check current position from inner env
        position = self.env._position

        if position == 0:
            # FLAT — baseline controls entry
            obs_for_baseline = self._get_current_obs()
            inner_action = self.baseline.act(obs_for_baseline, self.env)
        else:
            # IN POSITION — RL controls exit
            if action == 1:
                inner_action = 1  # CLOSE
            else:
                inner_action = 0  # HOLD

        return self.env.step(inner_action)

    # Expose inner env attributes needed by evaluate_policy() and DeterministicBaseline
    @property
    def _position(self):
        return self.env._position

    @property
    def _current_bar_idx(self):
        return self.env._current_bar_idx

    @property
    def _bars_data(self):
        return self.env._bars_data

    def _get_current_obs(self):
        """Get current observation without stepping (for baseline decision)."""
        return self.env._get_observation_direct()
