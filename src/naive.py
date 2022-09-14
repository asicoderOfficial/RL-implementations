import gym
from decimal import Decimal
from time import sleep
import numpy as np

env_id = 'MountainCar-v0'
env = gym.make(env_id)
#The origin is the middle point in the road between the 2 mountains.
#The naive approach for solving this problem is the following:
#By knowing the upper and lower bounds of the space, we can split it into squares, and define for each, a probability of transition for each possible action.
#This probability distribution, is defined as the 
#Each iteration always starts in the origin, and ends once the car reaches the goal or surpasses the maximum number of training steps.
#The training phase ends once the number of steps needed to reach the goal converges, or the maximum number of training iterations is reached. 
#This way, a basic framework is created for experimenting with multiple reward functions. 

high = env.observation_space.high
low = env.observation_space.low
#We will divide the space into a grid of 10x10 rectangles with equal area
number_of_squares = 10
grid_rectangles_width = round((abs(low[0] - high[0]))/number_of_squares, 3)
grid_rectangles_height = round((abs(low[1] - high[1]))/number_of_squares, 3)
grid_x = np.arange(start=low[0], stop=high[0], step=grid_rectangles_width)
grid_x = [round(x, 2) for x in grid_x]
grid_y = np.arange(start=low[1], stop=high[1], step=grid_rectangles_height)
grid_y = [round(y, 2) for y in grid_y]
grid = [(x, y) for x in grid_x for y in grid_y]

def create_rectangle(down_left_x, down_left_y):
    return [(down_left_x, down_left_y), \
        (down_left_x, down_left_y + grid_rectangles_height), \
        (down_left_x + grid_rectangles_width, down_left_y), \
        (down_left_x + grid_rectangles_width, down_left_y + grid_rectangles_height)]


def is_point_in_rectangle(x, y, down_left_x, down_left_y):
    rectangle = create_rectangle(down_left_x, down_left_y)
    return rectangle[0][0] <= x <= rectangle[2][0] and rectangle[0][1] <= y <= rectangle[1][1]


def get_rectangle_for_point(x, y):
    for down_left_x, down_left_y in grid:
        if is_point_in_rectangle(x, y, down_left_x, down_left_y):
            return down_left_x, down_left_y


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return float('.'.join([i, (d+'0'*n)[:n]]))


#Define starting probabilities: set all to equally probable.
probs = {area : {0:0, 1:0, 2:0} for area in grid}

obs_space = env.observation_space #Tuples of two numbers: velocity and position
action_space = env.action_space 

max_num_steps = 10000
max_num_iterations = 0


def train():
    for iteration in range(max_num_iterations+1):
        print(f'-------- Iteration {iteration} ----------')
        pos, velocity = env.reset()
        for step in range(max_num_steps):
            curr_rectangle = get_rectangle_for_point(truncate(pos, 2), truncate(velocity, 2))
            rewards = probs[curr_rectangle] 
            if step == 0 and iteration == 1:
                #Start with a random action.
                action = env.action_space.sample()
            else:
                #select the action with maximum reward
                action = max(rewards, key=rewards.get)
            # apply the action
            observation, reward, done, info = env.step(action)
            #save the reward: mean of rewards
            probs[curr_rectangle][action] = (probs[curr_rectangle][action] + reward) / 2
            #update pos and velocity
            pos = truncate(observation[0], 2)
            velocity = truncate(observation[1], 2)
            
            # Render the env
            env.render()

            # Wait a bit before the next frame unless you want to see a crazy fast video
            sleep(0.001)
            
            # If the epsiode is up, then start another one
            if done and not info['TimeLimit.truncated']:
                #Termination, goal reached!
                break

train()
# Close the env
env.close()