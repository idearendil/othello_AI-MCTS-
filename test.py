"""
Othello game example.
Prints board state to stdout with random agents by default.
"""

import re
import sys
import colorama
import numpy as np

from fights.envs import othello

from mctsAI import MCTSAgent

sys.path.append("../")


def fallback_to_ascii(s: str) -> str:
    """
    for the windows env
    """
    try:
        s.encode(sys.stdout.encoding)
    except UnicodeEncodeError:
        s = re.sub(
            "[┌┬┐├┼┤└┴┘╋]", "+", re.sub("[─━]", "-", re.sub("[│┃]", "|", s)))
    return s


def run():
    """
    Just run the othello environment with two random agents.
    """
    colorama.init()

    state = othello.OthelloEnv().initialize_state()
    agents = [MCTSAgent(0), MCTSAgent(1)]

    while not state.done:

        # print("\x1b[1;1H")
        print(fallback_to_ascii(str(state)))

        for agent in agents:

            action = agent(state)
            state = othello.OthelloEnv().step(state, agent.agent_id, action)

            # print("\x1b[1;1H")
            print(fallback_to_ascii(str(state)))

            pause = input()

            if state.done:
                if state.reward[0] != 0:
                    print(f"agent {np.argmax(state.reward)} won!!")
                else:
                    print("draw!!")
                break


if __name__ == "__main__":
    run()
