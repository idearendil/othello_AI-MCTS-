"""
This file defines dual network, which is the neural network of mctsAI.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FILTERS_NUM = 64  # 128
RESIDUAL_NUM = 4  # 16
INPUT_SHAPE = (2, 8, 8)
POLICY_OUTPUT_SIZE = 64
HIDDEN_LAYER_SIZE = 128  # 256


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (
        dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (
        dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class ResidualBlock(nn.Module):
    """
    This is the definition of a basic residual block of Dual_Network
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=FILTERS_NUM,
                out_channels=FILTERS_NUM,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=FILTERS_NUM),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=FILTERS_NUM,
                out_channels=FILTERS_NUM,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=FILTERS_NUM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class DualNetwork(nn.Module):
    """
    This is the definition of DualNetwork
    """

    def __init__(self) -> None:
        super().__init__()
        c, h, w = INPUT_SHAPE
        conv_out_hw = calc_conv2d_output((h, w), 3, 1, 1)
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=FILTERS_NUM,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=FILTERS_NUM),
            nn.ReLU()
        )

        res_blocks = []
        for _ in range(RESIDUAL_NUM):
            res_blocks.append(ResidualBlock())
        self.res_blocks = nn.Sequential(*res_blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=FILTERS_NUM,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, POLICY_OUTPUT_SIZE)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=FILTERS_NUM,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)

        p = self.policy_head(features)
        v = self.value_head(features)

        return p, v


class DummyNetwork(nn.Module):
    """
    This is the definition of DummyNetwork
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.policy_layer = nn.Linear(84, POLICY_OUTPUT_SIZE)
        self.value_layer = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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


def load_network(device, model_name) -> DummyNetwork:
    """
    Load neural network model with model_name from ./model file.

    :arg device:
        Which device used for model. 'cpu' or 'cuda'.

    :arg model_name:
        The name of model which will be loaded.
    """
    model_path = './model/' + model_name + '.pt'
    model = DummyNetwork().to(device)
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
    model = DummyNetwork().to(device)
    save_network(model, model_name)
