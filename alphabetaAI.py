"""
Othello ai agent model with minimax search algorithm and
alpha-beta pruning.
"""

import numpy as np

from fights.base import BaseAgent
from fights.envs import othello


class AlphaBetaAgent(BaseAgent):
    """
    AI agent with minimax search algorithm and alpha-beta pruning.
    """
    env_id = ("othello", 0)

    def __init__(self, agent_id: int, depth: int, seed: int) -> None:
        self.agent_id = agent_id
        self.depth = depth
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: othello.OthelloState):
        actions = []
        for cx in range(othello.OthelloEnv.board_size):
            for cy in range(othello.OthelloEnv.board_size):
                action = [cx, cy]
                if state.legal_actions[self.agent_id][cx][cy]:
                    actions.append(action)
        return actions

    def value_function(self, state: othello.OthelloState):
        """
        Getting value of specific state.
        (as known as evaluation function)
        """
        return np.count_nonzero(state.board[self.agent_id]) - np.count_nonzero(
            state.board[1 - self.agent_id])

    def minimax_search(
            self, state: othello.OthelloState, depth, is_opp, threshold):

        """
        Tree searching with minimax algorithm and
        alpha-beta pruning algorithm.
        """
        if depth == self.depth:
            return self.value_function(state)

        if state.done:
            return state.reward[self.agent_id] * 100

        if is_opp:
            maxmin_point = 99999
        else:
            maxmin_point = -99999

        possible_actions = []
        for cx in range(othello.OthelloEnv.board_size):
            for cy in range(othello.OthelloEnv.board_size):
                action = [cx, cy]
                if state.legal_actions[self.agent_id ^ is_opp][cx][cy]:
                    possible_actions.append(action)

        for action in possible_actions:
            next_state = othello.OthelloEnv().step(
                state, self.agent_id ^ is_opp, action)
            next_point = self.minimax_search(
                next_state, depth+1, 1-is_opp, maxmin_point)

            if is_opp:
                if maxmin_point > next_point:
                    maxmin_point = next_point
                    if threshold > next_point:
                        return next_point
            else:
                if maxmin_point < next_point:
                    maxmin_point = next_point
                    if threshold < next_point:
                        return next_point

        return maxmin_point

    def __call__(self, state: othello.OthelloState) -> othello.OthelloAction:
        possible_actions = []
        for cx in range(othello.OthelloEnv.board_size):
            for cy in range(othello.OthelloEnv.board_size):
                action = [cx, cy]
                if state.legal_actions[self.agent_id][cx][cy]:
                    possible_actions.append(action)

        max_point = -99999
        best_actions = []
        for action in possible_actions:
            next_state = othello.OthelloEnv().step(
                state, self.agent_id, action)
            next_point = self.minimax_search(
                next_state, 1, 1, max_point)
            if next_point > max_point:
                max_point = next_point
                best_actions = [action]
            elif next_point == max_point:
                best_actions.append(action)

        return self._rng.choice(best_actions)
