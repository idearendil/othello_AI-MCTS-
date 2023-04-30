"""
Power test file for othello ai agents.
Plays 100 games and prints the win rate of each agent.
"""

from fights.envs import othello

from randomAI import RandomAgent
from mctsAI import MCTSAgent


def run():
    """
    Running test.
    """

    win_rate = [0, 0, 0]

    for game_id in range(50):

        state = othello.OthelloEnv().initialize_state()
        agents = [RandomAgent(0, seed=game_id), MCTSAgent(1)]

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

    print("\n########################################")
    print(f"agent 0 wins : {win_rate[0]} times")
    print(f"agent 1 wins : {win_rate[1]} times")
    print(f"draw : {win_rate[2]} times")
    print("########################################")


if __name__ == "__main__":
    run()
