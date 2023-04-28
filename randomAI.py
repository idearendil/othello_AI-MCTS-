"""
Othello ai agent model with random policy.
"""

import numpy as np

from fights.base import BaseAgent
from fights.envs import othello


class RandomAgent(BaseAgent):
    """
    Just a random agent
    """
    env_id = ("othello", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: othello.OthelloState):
        actions = []
        for cx in range(othello.OthelloEnv.board_size):
            for cy in range(othello.OthelloEnv.board_size):
                action = [cx, cy]
                if state.legal_actions[self.agent_id][cx][cy]:
                    actions.append(action)
        return actions

    def __call__(self, state: othello.OthelloState) -> othello.OthelloAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)
