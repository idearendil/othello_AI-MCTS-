"""
Train mctsAI by self play.
"""

import numpy as np

import torch
import torch.nn as nn

from fights.envs import othello

from mctsAI import MCTSAgent

from dual_network import save_network

SP_GAME_COUNT = 100  # 500
TEST_GAME_COUNT = 20  # 100
RN_EPOCHS = 100  # 100


def evaluate_network(current_agent):
    """
    Running power test.

    [ Newly trained network ]
    vs
    [ Best network so far ]
    """

    win_rate = [0, 0, 0]

    current_agent.agent_id = 0
    agents = [current_agent, MCTSAgent(1)]

    for _ in range(TEST_GAME_COUNT // 2):

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

    print(win_rate[0], win_rate[1])
    current_agent.agent_id = 1
    agents = [MCTSAgent(0), current_agent]

    for _ in range(TEST_GAME_COUNT // 2):

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

    print(win_rate[0], win_rate[1])
    return win_rate[0] > win_rate[1]


def self_play(current_agent):
    """
    Executes self play and collect records from the games.
    """

    for game_id in range(SP_GAME_COUNT):
        print(game_id)
        state = othello.OthelloEnv().initialize_state()

        history = []

        while not state.done:

            for agent_id in range(2):
                current_agent.agent_id = agent_id

                policy = current_agent.MCTS(state)
                sparse_policy = []
                id = 0
                for col_r in range(8):
                    for col_c in range(8):
                        if state.legal_actions[agent_id][col_r][col_c]:
                            sparse_policy.append(policy[id])
                            id += 1
                        else:
                            sparse_policy.append(0)
                sparse_policy = np.array(sparse_policy)
                if agent_id == 0:
                    history.append([state.board, sparse_policy, None])
                else:
                    flipped_board = np.array((state.board[1], state.board[0]))
                    history.append([flipped_board, sparse_policy, None])
                action = current_agent.select_action(state, policy)
                state = othello.OthelloEnv().step(state, agent_id, action)

                if state.done:
                    break

        for record_id, record in enumerate(history):
            record[2] = state.reward[record_id % 2]
            current_agent.saved_data.push((record[0], record[1], record[2]))

    return


def train(current_agent, learning_rate, epoch_num, batch_size):
    """
    Train DualNetwork with the data from SavedData.
    """

    current_agent.model.train()

    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(current_agent.model.parameters(),
                                 lr=learning_rate)

    for epoch_id in range(epoch_num):

        batch_data = current_agent.saved_data.pull(batch_size)

        s_tensor = torch.from_numpy(
            np.array(batch_data[0], dtype=np.float32)).to(current_agent.device)
        a_tensor = torch.tensor(
            np.array(batch_data[1], dtype=np.float32)).to(current_agent.device)
        v_tensor = torch.tensor(
            np.array(batch_data[2], dtype=np.float32)).to(current_agent.device)

        prediction = current_agent.model.forward(s_tensor)
        prediction = (prediction[0].squeeze(), prediction[1].squeeze())
        prediction = torch.cat((prediction[0], prediction[1].unsqueeze(1)), dim=1)
        y = torch.cat((a_tensor, v_tensor.unsqueeze(1)), dim=1)
        loss = criterion(prediction, y).to(current_agent.device)
        if epoch_id == epoch_num - 1:
            print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_cycle():
    """
    Executes overall training cycle of MCTSAgent.
    The result network(DualNetwork) is saved in ./model/best.pt
    """

    simulation_cnt = 100
    tau = 1.0
    learning_rate = 0.001
    epoch_num = 100
    batch_size = 128
    train_time = 10

    for i in range(train_time):
        print('Train', i, '====================')

        agent = MCTSAgent(agent_id=0, simulation_cnt=simulation_cnt, tau=tau)

        self_play(agent)

        train(current_agent=agent,
              learning_rate=learning_rate,
              epoch_num=epoch_num,
              batch_size=batch_size)

        if evaluate_network(agent):
            save_network(agent.model, 'best')


if __name__ == '__main__':
    train_cycle()
