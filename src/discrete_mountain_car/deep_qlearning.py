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


class Agent:
    def __init__(self, n_actions, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_linear_decay=1e-5, epsilon_min=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_linear_decay = epsilon_linear_decay
        self.epsilon_min = epsilon_min
        #Q network of the agent
        self.Q = DQN(learning_rate)
        #Set of possible actions
        self.action_space = [i for i in range(n_actions)]

    
    def select_action(self, observation):
        if np.random.random() > self.epsilon:
            #Pick the best action, to exploit the policy
            state = torch.tensor(observation, dtype=torch.float).to(self.Q.device)
            return torch.argmax(self.Q.forward(state)).item()
        else:
            #Pick a random action, to explore
            return np.random.choice(self.action_space)
