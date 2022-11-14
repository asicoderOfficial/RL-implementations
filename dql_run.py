import gym
import time
import numpy as np
import argparse
from src.discrete_mountain_car.deep_qlearning import Agent
from src.discrete_mountain_car.naive import MCDNaive




def main():
    #Constant environment variables
    env = gym.make('MountainCar-v0').env #Avoid truncation at 200 steps
    n_actions = 3

    parser = argparse.ArgumentParser()
    #Training phase parameters
    parser.add_argument('-ne', '--numberofepisodes', type=int, help='Number of episodes to train', default=1_000)
    parser.add_argument('-d', '--plotsdirectory', type=str, help='The directory for storing plots of training progress', default='')
    #Optional arguments specified by the user to tweak the Agent and DQN hyperparameters
    parser.add_argument('-lr', '--learningrate', type=float, help='The learning rate for backprop', default=0.001)
    parser.add_argument('-g', '--gamma', type=float, help='Gamma, the discount rate', default=0.99)
    parser.add_argument('-e', '--epsilon', type=float, help='Epsilon, for epsilon-greedy strategy', default=1.0)
    parser.add_argument('-em', '--epsilonmin', type=float, help='Epsilon min, for epsilon-greedy strategy', default=1.0)
    parser.add_argument('-ed', '--epsilondecay', type=float, help='Epsilon decay, for epsilon-greedy strategy', default=1e-5)
    args = parser.parse_args()

    n_episodes = args.numberofepisodes

    #Variables to store training data to later evaluate model performance with plots
    scores = []
    epsilon_history = []

    #The agent
    naive_dql_agent = Agent(n_actions)

    #Make states finite
    mcdnaive = MCDNaive(20, 100, 100)

    #Training loop
    for i in range(n_episodes):
        score = 0
        done = False
        curr_state = env.reset()
        env.render()
        curr_state = mcdnaive.get_state_for_position_and_velocity(curr_state[0], curr_state[1])
        while not done:
            #Pick what is thought to be the most optimal action for the current environment obervation, transition and save score
            action = naive_dql_agent.select_action(curr_state)
            new_state, reward, done, _ = env.step(action)
            new_state = mcdnaive.get_state_for_position_and_velocity(new_state[0], new_state[1])
            reward = reward * (abs(new_state[1]) * 10 + (abs(0.07 - new_state[0]) * 10 * abs(reward)))
            score += reward
            #Learn from the action taken
            naive_dql_agent.learn(curr_state, action, reward, new_state)
            #Update the current observation
            curr_state = new_state
            env.render()
            time.sleep(0.001)

        scores.append(score)
        epsilon_history.append(naive_dql_agent.epsilon)

        if (i + 1) % 5 == 0:
            print(f'-- Episode {i + 1} -> Avg last 5 score: {np.mean(scores[-5])} --')

if __name__ == '__main__':
    main()