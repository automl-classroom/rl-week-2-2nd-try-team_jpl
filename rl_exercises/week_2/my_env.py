from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initializes the observation and action space for the environment."""
        # Initialize Spaces and others
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

        self.state = 0
        self.states = np.arange(2)
        self.actions = np.arange(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        # Set state and current steps to 0 to reset, since the start is always at 0

        self.state = 0

        # Returns observation which is the current state

        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        # Copied from Mars Rover to prevent values for action other than 0 or 1
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        # State/ Obs equals action since action = 0/1 leads to being in state = 0/1
        self.state = action

        # Reward equals action
        reward = float(action)

        # Since the task actually never terminates/ finishes

        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def get_reward_per_action(self) -> np.ndarray:
        # Obtaining observation and action space dimensions
        nS, nA = self.observation_space.n, self.action_space.n

        # Creating "empty"/ 0 matrix
        R = np.zeros((nS, nA), dtype=int)

        # Fill in matrix
        for s in range(nS):
            for a in range(nA):
                # The reward is equal to the taken action as described in the function description
                R[s, a] = a
        return R

    def get_transition_matrix(self) -> np.ndarray:
        # Obtaining observation and action space dimensions
        nS, nA = self.observation_space.n, self.action_space.n

        # Creating transition matrix with the right dimensions
        T = np.zeros((nS, nA, nS), dtype=float)

        # Calculate the transition probabilities for each state transition
        for s in range(nS):
            for a in range(nA):
                # The next state is the previous executed action (s_next = a) with a probability of 1 (deterministic)
                # If a takes the value 1 then s_next also takes value 1 to 100%. It's not possible to transition to a
                # different stage other than the action, which is why all other transition have 0% chance to occur
                T[s, a, a] = 1.0
        return T


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        super().__init__(env)

        assert 0.0 <= noise <= 1.0, "noise must be in [0,1]"

        self.noise = noise

        # Create equal distribution to compare with noise prob
        self.rng = np.random.default_rng(seed)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        # Call reset fct from MyEnv
        true_obs, info = self.env.reset(seed=seed, options=options)

        # return the noisy observation
        return self._noisy_obs(true_obs), info

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        # Call step fct
        true_obs, reward, terminated, truncated, info = self.env.step(action)

        # return the noisy observation
        return self._noisy_obs(true_obs), reward, terminated, truncated, info

    def _noisy_obs(self, true_obs: int) -> int:
        # If random generated number is smaller than noise, simulate that the opposite action/ step is taken
        # 0 -> 1 and 1 -> 0
        if self.rng.random() < self.noise:
            noisy_value = (
                1 - true_obs
            )  # This formula ensures that 0 becomes 1 and 1 becomes 0

        # Otherwise, in the more common case, the action stays the same
        else:
            noisy_value = true_obs

        return noisy_value
