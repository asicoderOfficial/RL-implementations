import gym
from time import sleep
import numpy as np

env = gym.make('MountainCar-v0')
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
grid_y = np.arange(start=low[1], stop=high[1], step=grid_rectangles_height)
grid = list(zip(grid_x, grid_y))


#Define starting probabilities: set all to equally probable.
probs = {area : {0:0.33, 1:0.33, 2:0.33} for area in grid}
