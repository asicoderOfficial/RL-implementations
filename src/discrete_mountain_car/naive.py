import gym
from typing import Callable
from time import sleep
from numpy import arange
from math import sin
from src.utilities.numeric import truncate
from src.discrete_mountain_car.mcd import MCD


#TODO: Record training metrics
class MCDNaive(MCD):
    # Parent class of all naive implementations of the discrete mountain car problem.

    def __init__(self, n_splits:int, max_steps:int, max_iterations:int, results_path:str='', stop_at_first_flag:bool=False) -> None:
        """ Initialize the class.

        Args:
            n_splits (int): Number of equal splits of velocity and position.
            max_steps (int): Maximum number of steps per iteration.
            max_iterations (int): Maximum number of iterations.
            results_path (str, optional): Where to store the training results. Defaults to ''.
            stop_at_first_flag (bool, optional): Stop the first time is reached at training, or continue training. Defaults to False.

        Raises:
            TypeError: If n_splits is not an integer.
            ValueError: If n_splits is not greater than 1.
            TypeError: If max_steps is not an integer.
            ValueError: If max_steps is not greater than 1.
            TypeError: If max_iterations is not an integer.
            ValueError: If max_iterations is not greater than 1.
            TypeError: If results_path is not a string.
            TypeError: If stop_at_first_flag is not a boolean.
        
        Returns:
            None
        """        
        if not isinstance(n_splits, int): raise TypeError('n_splits must be an integer')
        if not 1 <= n_splits <= 30: raise ValueError('n_splits must be between 1 and 30')
        if not isinstance(max_steps, int): raise TypeError('max_steps must be an integer')
        if max_steps <= 0: raise ValueError('max_steps must be greater than 0')
        if not isinstance(max_iterations, int): raise TypeError('max_iterations must be an integer')
        if max_iterations <= 0: raise ValueError('max_iterations must be greater than 0')
        if not isinstance(results_path, str): raise TypeError('results_path must be a string')
        if not isinstance(stop_at_first_flag, bool): raise TypeError('stop_at_first_flag must be a boolean')

        self.n_splits = n_splits
        self.max_steps = max_steps
        self.max_iterations = max_iterations
        self.results_path = results_path
        self.stop_at_first_flag = stop_at_first_flag

        env_id = 'MountainCar-v0'
        self.env = gym.make(env_id)
        self.env_high = self.env.observation_space.high
        self.env_low = self.env.observation_space.low
        super().define_states()


    def train(self, reward_function:Callable[[dict, tuple, int, float, dict], float], **kwargs) -> None:
        """ Train the agent.

        Args:
            reward_function (Callable): Custom reward function.
            **kwargs: Arguments for the reward function.
        
        Returns:
            None
        """        
        for iteration in range(1, self.max_iterations + 1):
            print(f'-------- Iteration {iteration} ----------')
            pos, velocity = self.env.reset()
            for step in range(1, self.max_steps + 1):
                curr_state = self.get_state_for_position_and_velocity(float(truncate(pos, 2)), float(truncate(velocity, 2)))
                rewards = self.rewards[curr_state] 
                if step == 1 and iteration == 1:
                    #Start with a random action.
                    action = self.env.action_space.sample()
                else:
                    #select the action with maximum reward
                    action = max(rewards, key=rewards.get)
                # apply the action
                observation, gym_reward, done, info = self.env.step(action)
                #save the reward: mean of rewards
                self.rewards[curr_state][action] = reward_function(self.rewards, curr_state, action, gym_reward, **kwargs)
                #update pos and velocity
                pos = float(truncate(observation[0], 2))
                velocity = float(truncate(observation[1], 2))
                # Render the self.env
                self.env.render()
                # Wait a bit before the next frame unless you want to see a crazy fast video
                sleep(0.001)
                # If the epsiode is up, then start another one
                if done and not info['TimeLimit.truncated']:
                    if self.stop_at_first_flag:
                        self.env.close()
                        return
                    else:
                        #Termination, goal reached!
                        break


class MCDNaiveMean(MCDNaive):
    # Naive mean reward

    def __init__(self, n_splits:int, max_steps:int, max_iterations:int, results_path:str='', stop_at_first_flag:bool=False) -> None:
        """ Initialize the class, using the parent class MCDNaive constructor.

        Args:
            n_splits (int): Number of equal splits of velocity and position.
            max_steps (int): Maximum number of steps per iteration.
            max_iterations (int): Maximum number of iterations.
            results_path (str, optional): Where to store the training results. Defaults to ''.
            stop_at_first_flag (bool, optional): Stop the first time is reached at training, or continue training. Defaults to False.
        
        Returns:
            None
        """
        super().__init__(n_splits, max_steps, max_iterations, results_path, stop_at_first_flag)
    

    def _reward_function(rewards: dict, curr_state:tuple, action:int, gym_reward:float) -> float:
        """ Reward function for the naive mean.

        Args:
            rewards (dict): All rewards for all states and actions.
            curr_state (tuple): Current state, tuple of (position, velocity).
            action (int): Action taken.
            gym_reward (float): The reward given by OpenAI Gym.

        Returns:
            float: Reward.
        """        
        return (rewards[curr_state][action] + gym_reward) / 2


    def train(self) -> None:
        """ Train the agent by maximizing the mean reward, calculated as the mean of the previous reward and the current reward.

        Returns:
            None
        """        
        super().train(MCDNaiveMean._reward_function)


class MCDNaiveSin(MCDNaive):
    # Naive sin reward

    def __init__(self, n_splits:int, max_steps:int, max_iterations:int, sin_reward_reducing:int=100, velocity_reward_reducing:int=100, flag_distance_mult:int=10, results_path:str='', stop_at_first_flag:bool=False) -> None:
        """ Initialize the class, using the parent class MCDNaive constructor.

        Args:
            n_splits (int): Number of equal splits of velocity and position.
            max_steps (int): Maximum number of steps per iteration.
            max_iterations (int): Maximum number of iterations.
            sin_reward_reducing (int, optional): The bigger, the less the sin(position) rewards the agent. Defaults to 100.
            velocity_reward_reducing (int, optional): The bigger, the less the velocity rewards the agent. Defaults to 100.
            flag_distance_mult (int, optional): The bigger, the more the distance from the current position to the flag rewards the agent. Defaults to 10.
            results_path (str, optional): Where to store the training results. Defaults to ''.
            stop_at_first_flag (bool, optional): Stop the first time is reached at training, or continue training. Defaults to False.
        
        Raises:
            TypeError: If sin_reward_reducing is not an integer.
            TypeError: If velocity_reward_reducing is not an integer.
            TypeError: If flag_distance_mult is not an integer.
            ValueError: If sin_reward_reducing is not between 10 and 1000.
            ValueError: If velocity_reward_reducing is not between 10 and 1000.
            ValueError: If flag_distance_mult is not between 1 and 100.

        Returns:
            None
        """
        super().__init__(n_splits, max_steps, max_iterations, results_path, stop_at_first_flag)
        if not isinstance(sin_reward_reducing, int): raise TypeError('sin_reward_reducing must be an integer.')
        if not 10 <= sin_reward_reducing <= 1000: raise ValueError('sin_reward_reducing must be between 10 and 1000.')
        if not isinstance(velocity_reward_reducing, int): raise TypeError('velocity_reward_reducing must be an integer.')
        if not 10 <= velocity_reward_reducing <= 1000: raise ValueError('velocity_reward_reducing must be between 10 and 1000.')
        if not isinstance(flag_distance_mult, int): raise TypeError('flag_distance_mult must be an integer.')
        if not 1 <= flag_distance_mult <= 100: raise ValueError('flag_distance_mult must be between 1 and 100.')
        self.sin_reward_reducing = sin_reward_reducing
        self.velocity_reward_reducing = velocity_reward_reducing
        self.flag_distance_mult = flag_distance_mult


    def _reward_function(rewards: dict, curr_state:tuple, action:int, gym_reward:float, **kwargs:dict) -> float:
        """ Reward function with 4 components:
        - The gym_reward: penalization for how much time the agent needs to reach the flag.
        - How 'high' is the agent in the sin wave: this induces the agent to both gain momentum and use it to reach the flag.
        - How fast is the agent: this induces the agent to achieve the goal faster.
        - How close is the agent to the flag: this induces the agent to reach the flag, which is the final goal.

        Then, the mean between the previous reward and the current reward is calculated and returned.

        Args:
            rewards (dict): All rewards for all states and actions.
            curr_state (tuple): Current state, tuple of (position, velocity).
            action (int): Action taken.
            gym_reward (float): The reward given by OpenAI Gym.

        Returns:
            float: Reward.
        """        
        pos = curr_state[0]
        velocity = curr_state[1]
        if velocity == 0:
            velocity = 0.001
        sin_of_pos = sin(pos)
        if sin_of_pos == 0:
            sin_of_pos = 0.001
        sin_reward_reducing = kwargs['sin_reward_reducing']
        velocity_reward_reducing = kwargs['velocity_reward_reducing']
        flag_distance_mult = kwargs['flag_distance_mult']
        env_high = kwargs['env_high_pos']
        return (rewards[curr_state][action] +
            (gym_reward * \
            (1 / (sin_of_pos * sin_reward_reducing)) * \
            (1 / (abs(velocity) * velocity_reward_reducing)) * \
            (1 / (abs(env_high - pos) * flag_distance_mult)))) / 2


    def train(self, **kwargs) -> None:
        """ Train the agent by maximizing the _reward_function.

        Returns:
            None
        """        
        super().train(MCDNaiveSin._reward_function, **{'sin_reward_reducing': self.sin_reward_reducing, 'velocity_reward_reducing': self.velocity_reward_reducing, 'flag_distance_mult': self.flag_distance_mult, 'env_high_pos': self.env_high[0]})

m = MCDNaiveMean(10, 1000, 1000, stop_at_first_flag=True)
m.train()