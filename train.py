"""
Train mctsAI by self play.
"""

import numpy as np

import torch
import torch.nn as nn

from fights.envs import othello

from mctsAI import MCTSAgent

SP_GAME_COUNT = 10
RN_EPOCHS = 100


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

    for game_id in range(50):

        state = othello.OthelloEnv().initialize_state()

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = othello.OthelloEnv().step(
                    state, agent.agent_id, action)

                if state.done:
                    print(f"game {game_id} is done!!")
                    if state.reward[0] == 1:
                        win_rate[0] += 1
                    elif state.reward[1] == 1:
                        win_rate[1] += 1
                    else:
                        win_rate[2] += 1
                    break

    current_agent.agent_id = 1
    agents = [MCTSAgent(0), current_agent]

    for game_id in range(50):

        state = othello.OthelloEnv().initialize_state()

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = othello.OthelloEnv().step(
                    state, agent.agent_id, action)

                if state.done:
                    print(f"game {game_id} is done!!")
                    if state.reward[0] == 1:
                        win_rate[1] += 1
                    elif state.reward[1] == 1:
                        win_rate[0] += 1
                    else:
                        win_rate[2] += 1
                    break

    return win_rate[0] > win_rate[1]


def self_play(current_agent):
    """
    Executes self play and collect records from the games.
    """

    for _ in range(SP_GAME_COUNT):
        state = othello.OthelloEnv().initialize_state()

        history = []

        while not state.done:

            for agent_id in range(2):
                current_agent.agent_id = agent_id

                policy = current_agent.MCTS(state)
                if agent_id == 0:
                    history.append([state.board, policy, None])
                else:
                    flipped_board = np.array((state.board[1], state.board[0]))
                    history.append([flipped_board, policy, None])
                action = current_agent.select_action(state, policy)
                state = othello.OthelloEnv().step(state, agent_id, action)

                if state.done:
                    break

        for record_id, record in enumerate(history):
            record[2] = state.reward[record_id % 2]
            current_agent.saved_data.push((record[0], record[1], record[2]))
        print(len(history))

    return


def train(current_agent, learning_rate, epoch_num, batch_size):
    """
    Train DualNetwork with the data from SavedData.
    """

    current_agent.model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(current_agent.model.parameters(),
                                lr=learning_rate)

    for _ in range(epoch_num):

        batch_data = current_agent.saved_data.pull(batch_size)

        s_tensor = torch.from_numpy(np.array(batch_data[0])).to(
            current_agent.device)
        a_tensor = torch.tensor(np.array(batch_data[1])).to(
            current_agent.device)
        v_tensor = torch.tensor(np.array(batch_data[2])).to(
            current_agent.device)

        prediction = current_agent.model.forward(s_tensor)
        loss = criterion(prediction, (a_tensor, v_tensor)).to(
            current_agent.device)

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
    learning_rate = 0.0001
    epoch_num = 100
    batch_size = 128
    train_time = 10

    agent = MCTSAgent(agent_id=0, simulation_cnt=simulation_cnt, tau=tau)

    for i in range(train_time):
        print('Train', i, '====================')

        self_play(agent)

        train(current_agent=agent,
              learning_rate=learning_rate,
              epoch_num=epoch_num,
              batch_size=batch_size)

        evaluate_network(agent)


if __name__ == '__main__':
    train_cycle()
