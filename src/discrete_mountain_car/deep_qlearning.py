import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pkg_resources

print(pkg_resources.get_distribution("torch").version)

print(torch.version.cuda)


class DQN(nn.Module):
    def __init__(self, learning_rate:float=0.001) -> None:
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #Send network to the device
        self.model.to(self.device)


    def forward(self, curr_state) -> torch.tensor:
        actions = self.model(curr_state)

        return actions


class NaiveDQLAgent():
    def __init__(self, gamma:float=0.99, epsilon:float=0.95) -> None:
        self.gamma


