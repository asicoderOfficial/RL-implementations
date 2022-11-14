import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pkg_resources


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


    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_linear_decay if self.epsilon > self.epsilon_min else self.epsilon_min
    

    def learn(self, curr_state, action, reward, new_state):
        #Zero out gradients, not to accumulate previous gradient descent progress
        self.Q.optimizer.zero_grad()
        #Convert the data provided to pytorch cuda tensors, so that they can be used for learning with the DQN, with the self.device specified (luckily, a gpu)
        curr_states = torch.tensor(curr_state, dtype=torch.float).to(self.Q.device)
        actions = torch.tensor(action, dtype=torch.uint8).to(self.Q.device)
        rewards = torch.tensor(reward, dtype=torch.float).to(self.Q.device)
        new_states = torch.tensor(new_state, dtype=torch.float).to(self.Q.device)

        #Calculation of the target value:
        #Calculate the q estimates, given the current state of the environment
        q_estimates = self.Q.forward(curr_states)[actions]
        #Calculate the maximal action for the agent's estimates value of the resulting states
        q_next = self.Q.forward(new_states).max()
        #Calculate what we want to get, the optimal value
        q_target = reward + self.gamma * q_next

        #Calculate the loss: distance (MSE) between the action the agent took, and the maximizing action the agent could have taken
        loss = self.Q.loss_function(q_target, q_estimates).to(self.Q.device)

        #Learn! Backpropagate
        loss.backward()
        self.Q.optimizer.step()
        
        #Decrement the value of epsilon, to little by little, start exploiting more and exploring less
        self.decrement_epsilon()