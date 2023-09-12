"""
Train mctsAI by self play.
"""

import numpy as np
from multiprocessing import Process, Queue

import torch
import torch.nn as nn

from fights.envs import othello

from mctsAI import MCTSAgent

from dual_network import save_network
from dual_network import load_network
from saved_data import SavedData

TRAIN_GAME_NUM = 200  # 500
TEST_GAME_NUM = 40  # 100
PROCESS_NUM = 8

LEARNING_RATE = 1e-3
EPOCH_NUM = 40
BATCH_SIZE = 1024
CYCLE_NUM = 100

SIMULATION_CNT = 100
TAU = 1.0


def self_play(q):
    """
    Executes self play and collect records from the games.
    """
    current_agent = MCTSAgent(
        agent_id=0, simulation_cnt=SIMULATION_CNT, tau=TAU, model_name='best')

    for game_id in range(TRAIN_GAME_NUM // PROCESS_NUM):
        print(game_id)
        state = othello.OthelloEnv().initialize_state()

        history = []

        while not state.done:

            for agent_id in range(2):
                current_agent.agent_id = agent_id

                policy = current_agent.MCTS(state)
                sparse_policy = []
                policy_id = 0
                for col_r in range(8):
                    for col_c in range(8):
                        if state.legal_actions[agent_id][col_r][col_c]:
                            sparse_policy.append(policy[policy_id])
                            policy_id += 1
                        else:
                            sparse_policy.append(0)
                sparse_policy = np.array(sparse_policy)
                if agent_id == 0:
                    history.append([state.board, sparse_policy, None])
                else:
                    flipped_board = state.perspective(agent_id).copy()
                    history.append(
                        [flipped_board, np.flip(sparse_policy), None])
                action = current_agent.select_action(state, policy)
                state = othello.OthelloEnv().step(state, agent_id, action)

                if state.done:
                    break

        for record_id, record in enumerate(history):
            record[2] = state.reward[record_id % 2]
            q.put((record[0], record[1], record[2]))

    q.put(None)
    return


def train(learning_rate, epoch_num, batch_size, saved_data):
    """
    Train DualNetwork with the data from SavedData.
    """
    current_agent = MCTSAgent(
        agent_id=0, simulation_cnt=SIMULATION_CNT, tau=TAU, model_name='best')

    current_agent.model.train()

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(current_agent.model.parameters(),
                                 lr=learning_rate)

    for _ in range(epoch_num):

        batch_data = saved_data.pull(batch_size)

        s_tensor = torch.from_numpy(
            np.array(batch_data[0], dtype=np.float32)).to(current_agent.device)
        a_tensor = torch.tensor(
            np.array(batch_data[1], dtype=np.float32)).to(current_agent.device)
        v_tensor = torch.tensor(
            np.array(batch_data[2], dtype=np.float32)).to(current_agent.device)

        prediction = current_agent.model.forward(s_tensor)
        prediction_cat = torch.cat(
            (prediction[0].squeeze(), prediction[1].squeeze().unsqueeze(1)),
            dim=1)
        y = torch.cat((a_tensor, v_tensor.unsqueeze(1)), dim=1)
        loss = criterion(prediction_cat, y)
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_network(current_agent.model, 'challenger')


def evaluate_network(q):
    """
    Running power test.

    [ Newly trained network ]
    vs
    [ Best network so far ]
    """

    win_rate = [0, 0, 0]

    current_agent = MCTSAgent(agent_id=0,
                              simulation_cnt=SIMULATION_CNT,
                              tau=TAU,
                              model_name='challenger')
    best_agent = MCTSAgent(
        agent_id=0, simulation_cnt=SIMULATION_CNT, tau=TAU, model_name='best')

    agents = [current_agent, best_agent]
    current_agent.agent_id = 0
    best_agent.agent_id = 1

    for _ in range(TEST_GAME_NUM // (2 * PROCESS_NUM)):

        state = othello.OthelloEnv().initialize_state()

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = othello.OthelloEnv().step(
                    state, agent.agent_id, action)

                if state.done:
                    if state.reward[0] == 1:
                        win_rate[0] += 1
                    elif state.reward[1] == 1:
                        win_rate[1] += 1
                    else:
                        win_rate[2] += 1
                    break

    agents = [best_agent, current_agent]
    current_agent.agent_id = 1
    best_agent.agent_id = 0

    for _ in range(TEST_GAME_NUM // (2 * PROCESS_NUM)):

        state = othello.OthelloEnv().initialize_state()

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = othello.OthelloEnv().step(
                    state, agent.agent_id, action)

                if state.done:
                    if state.reward[0] == 1:
                        win_rate[1] += 1
                    elif state.reward[1] == 1:
                        win_rate[0] += 1
                    else:
                        win_rate[2] += 1
                    break

    q.put((win_rate[0], win_rate[1]))
    q.put(None)
    return


def train_cycle():
    """
    Executes overall training cycle of MCTSAgent.
    The result network(DualNetwork) is saved in ./model/best.pt
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saved_data = SavedData(device=device)

    for i in range(CYCLE_NUM):
        print('Train', i, '====================')

        process_lst = []
        queue_lst = []
        for process_id in range(PROCESS_NUM):
            q = Queue()
            queue_lst.append(q)
            a_process = Process(
                target=self_play, args=(queue_lst[process_id],))
            process_lst.append(a_process)
            process_lst[process_id].start()
        for process_id in range(PROCESS_NUM):
            while True:
                item = queue_lst[process_id].get()
                if item is None:
                    break
                saved_data.push(item)
            process_lst[process_id].join()

        train(learning_rate=LEARNING_RATE,
              epoch_num=EPOCH_NUM,
              batch_size=BATCH_SIZE,
              saved_data=saved_data)

        win_rate = [0, 0]
        process_lst = []
        for process_id in range(PROCESS_NUM):
            q = Queue()
            queue_lst.append(q)
            a_process = Process(
                target=evaluate_network, args=(queue_lst[process_id],))
            process_lst.append(a_process)
            process_lst[process_id].start()
        for process_id in range(PROCESS_NUM):
            while True:
                item = queue_lst[process_id].get()
                if item is None:
                    break
                win_rate[0] += item[0]
                win_rate[1] += item[1]
            process_lst[process_id].join()

        print(win_rate[0], win_rate[1])
        if win_rate[0] > win_rate[1]:
            model = load_network(device, 'challenger')
            save_network(model, 'best')


if __name__ == '__main__':
    train_cycle()
