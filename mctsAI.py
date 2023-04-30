"""
Othello ai agent model with MCTS.
"""

import random
from math import sqrt
import numpy as np
import numpy.typing as npt
import torch

from fights.base import BaseAgent
from fights.envs import othello

from dual_network import load_network
from dual_network import reset_network


class MCTSAgent(BaseAgent):
    """
    The MCTS agent.
    """
    env_id = ("othello", 0)  # type: ignore

    def __init__(self,
                 agent_id: int,
                 simulation_cnt=100,
                 tau=1.0,
                 model_name='best') -> None:
        self.agent_id = agent_id  # type: ignore
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self.model = load_network(self.device, model_name)
        except FileNotFoundError:
            reset_network(self.device, model_name)
            self.model = load_network(self.device, model_name)
        self.model.to(self.device)
        # for i in range(16):
        #     self.model.layers[i].to(self.device)
        self.simulation_cnt = simulation_cnt
        self.tau = tau

    def __call__(self, state: othello.OthelloState) -> othello.OthelloAction:
        scores = self.MCTS(state)
        return self.select_action(state, scores)

    def _observe_state(self, state: othello.OthelloState, agent_id):
        board = state.perspective(agent_id).copy()
        return torch.Tensor(board).to(self.device)

    def _predict(self, state: othello.OthelloState, agent_id):
        observation = self._observe_state(state, agent_id)
        observation = observation.unsqueeze(0)
        with torch.no_grad():
            self.model.eval()
            y = self.model.forward(observation)
            if agent_id:
                y = (np.flip(y[0].squeeze().cpu().numpy()),
                     y[1].squeeze().cpu().numpy())
            else:
                y = (y[0].squeeze().cpu().numpy(),
                     y[1].squeeze().cpu().numpy())

        policy = y[0] * np.reshape(state.legal_actions[agent_id], (64))
        policy /= sum(policy)

        value = y[1]

        return policy, value

    def _boltzmann(self, weights: npt.NDArray[np.float32]):
        # print(weights)
        max_weight = np.max(weights)
        exp_weights = np.exp((weights - max_weight) / self.tau)
        sum_exp_weights = np.sum(exp_weights)
        final_weights = exp_weights / sum_exp_weights
        # print(final_weights)
        return final_weights

    def MCTS(self, state):
        """
        This function executes Monte Carlo Tree Search one time
        on current state(root node).
        """
        class Node:
            def __init__(self, state, p):
                self.state = state
                self.p = p
                self.w = 0
                self.n = 0
                self.child_nodes = None

        def nodes_to_scores(nodes):
            scores = []
            for node in nodes:
                scores.append(node.n)
            return scores

        def evaluate_node(node, agent_id):
            if node.state.done:
                value = node.state.reward[agent_id]
                node.w += value
                node.n += 1
                return value
            if not node.child_nodes:
                policy, value = self._predict(node.state, agent_id)
                node.w += value
                node.n += 1
                node.child_nodes = []
                for i in range(64):
                    col_r = i // 8
                    col_c = i % 8
                    if node.state.legal_actions[agent_id][col_r][col_c]:
                        next_state = othello.OthelloEnv().step(
                            node.state, agent_id, (col_r, col_c))
                        node.child_nodes.append(Node(next_state, policy[i]))
                return value
            else:
                next_child_node = select_next_node(node.child_nodes)
                value = -evaluate_node(next_child_node, 1-agent_id)
                node.w += value
                node.n += 1
                return value

        def select_next_node(child_nodes):
            C_PUCT = 1.0
            t = sum(nodes_to_scores(child_nodes))
            pucb_values = []
            for child_node in child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) + C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))
            return child_nodes[np.argmax(pucb_values)]

        root_node = Node(state, 0)

        for _ in range(self.simulation_cnt):
            evaluate_node(root_node, self.agent_id)

        scores = nodes_to_scores(root_node.child_nodes)
        scores = np.array(scores, dtype=np.float32)
        if self.tau == 0:
            action = np.argmax(scores)
            scores = np.zeros(len(scores), dtype=np.float32)
            scores[action] = 1
        else:
            scores = self._boltzmann(scores)
        return scores

    def select_action(self, state, scores):
        possible_actions = np.transpose(
            np.nonzero(state.legal_actions[self.agent_id]))
        return random.choices(possible_actions, weights=scores, k=1)[0]
