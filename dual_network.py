"""
This file defines dual network, which is the neural network of mctsAI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DN_FILTERS = 128
DN_RESIDUAL_NUM = 16
DN_INPUT_SHAPE = (8, 8, 2)
DN_OUTPUT_SIZE = 64


class ResidualBlock(nn.Module):
    """
    This is the definition of a basic residual block of Dual_Network
    """
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(DN_FILTERS, DN_FILTERS, 3, 1, padding="same"),
            nn.BatchNorm2d(DN_FILTERS),
            nn.ReLU(),
            nn.Conv2d(DN_FILTERS, DN_FILTERS, 3, 1, padding="same"),
            nn.BatchNorm2d(DN_FILTERS),
        )

    def forward(self, x):
        out = self.layer(x)
        out = F.relu(out + x)
        return out


class DualNetwork(nn.Module):
    """
    This is the definition of DualNetwork
    """
    def __init__(self):
        super(DualNetwork, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(DN_INPUT_SHAPE[2], DN_FILTERS, 3, 1, padding="same"),
            nn.BatchNorm2d(DN_FILTERS),
            nn.ReLU(),
        )
        self.layers = [ResidualBlock() for _ in range(DN_RESIDUAL_NUM)]
        self.policy_layer = nn.Linear(DN_FILTERS, DN_OUTPUT_SIZE)
        self.value_layer = nn.Linear(DN_FILTERS, 1)

    def forward(self, x):
        x = self.first_layer(x)
        for i in range(DN_RESIDUAL_NUM):
            x = self.layers[i](x)
        x = F.avg_pool2d(x, DN_INPUT_SHAPE[0]).squeeze().unsqueeze(0)
        p = F.softmax(self.policy_layer(x), dim=1)
        v = F.tanh(self.value_layer(x)).squeeze(1)
        return p, v


def save_network(model, model_name):
    """
    Saving neural network model in ./model file with model_name.

    :arg model:
        Neural network model to save.

    :arg model_name:
        The model's name which the model will be saved as.
    """
    file_name = './model/' + model_name + '.pt'
    torch.save(model.state_dict(), file_name)


def load_network(device, model_name) -> DualNetwork:
    """
    Load neural network model with model_name from ./model file.

    :arg device:
        Which device used for model. 'cpu' or 'cuda'.

    :arg model_name:
        The name of model which will be loaded.
    """
    model_path = './model/' + model_name + '.pt'
    model = DualNetwork().to(device)
    model.load_state_dict(torch.load(model_path))
    return model


def reset_network(device, model_name):
    """
    Reset neural network which name is model_name.

    :arg device:
        Which device used for model. 'cpu' or 'cuda'.

    :arg model_name:
        The name of model which will be resetted.
    """
    model = DualNetwork().to(device)
    save_network(model, model_name)
