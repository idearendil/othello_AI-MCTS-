"""
Othello ai agent model with manual policy.
"""

import numpy as np

from fights.base import BaseAgent
from fights.envs import othello


class ManualAgent(BaseAgent):
    """
    Just a manual agent
    """
    env_id = ("othello", 0)  # type: ignore

    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id  # type: ignore

    def __call__(self, state: othello.OthelloState) -> othello.OthelloAction:
        while True:
            a, b = input().split()
            a, b = int(a), int(b)
            if state.legal_actions[self.agent_id][a][b]:
                return (a, b)
            else:
                print("It's illegal action!!")
