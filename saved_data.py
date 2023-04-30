"""
Saved Data class file.
"""

import random


class SavedData():
    """
    Class of saved game play data.
    This saved data includes functions such as push and pull.
    Each data consists of state, action, value.
    """
    def __init__(self, device):
        self.buffer = []
        self.device = device

    def push(self, data):
        """
        Push one set of data into buffer.

        :arg data:
            data should be a tuple of state, action, value.
        """
        self.buffer.append(data)

    def pull(self, data_size):
        """
        Pull data of size data_size from buffer.

        :arg data_size:
            The size of data which will be pulled from buffer.

        :return:
            A tuple which consists of lists of state, action, value.
        """
        minibatch = random.sample(self.buffer, data_size)
        s_lst, a_lst, v_lst = [], [], []

        for data in minibatch:
            state, action, value = data
            s_lst.append(state)
            a_lst.append(action)
            v_lst.append(value)

        return s_lst, a_lst, v_lst

    def size(self):
        """
        Return the size of buffer.
        """
        return len(self.buffer)

    def clear(self):
        """
        Clear all saved data.
        """
        self.buffer = []
