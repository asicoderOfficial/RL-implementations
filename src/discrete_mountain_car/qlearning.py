from src.discrete_mountain_car.mcd import MCD
from src.utilities.numeric import truncate
import random
from numpy import exp
from time import sleep



class MCDQlearning(MCD):
    # Q-learning algorithm implementation class.
    # For balance between exploration and exploitation, an epsilon greedy strategy is used.
    # The reward function used is the default one given by gym.

    def __init__(self, n_splits:int, max_steps:int, max_episodes:int, \
        learning_rate:float=0.01, discount_factor:float=0.99, exploration_rate:float=1.0, max_exploration_rate:float=1.0, min_exploration_rate:float=0.01, exploration_decay_rate:float=0.01, \
        results_path:str='', stop_at_first_flag:bool=False) -> None:
        """ Initialize the class.

        Args:
            n_splits (int): Number of equal splits of velocity and position.
            max_steps (int): Maximum number of steps per episode.
            max_episodes (int): Maximum number of episodes.
            learning_rate (float, optional): How much weight to give to the new value, and how much to the old one. Usually seen in equations as alpha. Defaults to 0.01.
                In other words, it represents how quickly the agent abandons the previous Q-value in the Q-table for a given state-action pair for the new Q-value. 
            discount_factor (float, optional): How much weight to give to the current reward and to the future ones. Usually seen in equations as gamma. Defaults to 0.99.
            exploration_rate (float, optional): It is the probability that our agent will explore the environment rather than exploit it. Usually seen in equations as epsilon. Defaults to 1.
            max_exploration_rate (float, optional): The maximum value that the exploration rate can take. Defaults to 1.
            min_exploration_rate (float, optional): The minimum value that the exploration rate can take. Defaults to 0.01.
            exploration_decay_rate (float): The rate at which the exploration rate decreases. As the agent learns more, it should exploit more and explore less. Defaults to 0.01.
            results_path (str, optional): Where to store the training results. Defaults to ''.
            stop_at_first_flag (bool, optional): Stop the first time is reached at training, or continue training. Defaults to False.

        Raises:
            TypeError: If learning_rate, discount_factor, exploration_rate, max_exploration_rate, min_exploration_rate or exploration_decay_rate are not float.
            ValueError: If learning_rate is negative.
            ValueError: If discount_factor, exploration_rate, max_exploration_rate, min_exploration_rate or exploration_decay_rate are not between 0 and 1.
            ValueError: If max_exploration_rate is less than min_exploration_rate.
            ValueError: If the sum of max_exploration_rate and min_exploration_rate is greater than 2.
            ValueError: If exploration_rate is not between min_exploration_rate and max_exploration_rate.

        Returns:
            None
        """        
        super().__init__(n_splits, max_steps, max_episodes, results_path, stop_at_first_flag)
        super().define_states()

        if not isinstance(learning_rate, float): raise TypeError('learning_rate must be a float.')
        if not learning_rate >= 0.0: raise ValueError('learning_rate must be positive.')
        if not isinstance(discount_factor, float): raise TypeError('discount_factor must be a float.')
        if not 0 <= discount_factor <= 1: raise ValueError('discount_factor must be between 0 and 1 inclusive.')
        if not isinstance(exploration_rate, float): raise TypeError('exploration_rate must be a float.')
        if not 0 <= exploration_rate <= 1: raise ValueError('exploration_rate must be between 0 and 1 inclusive.')
        if not isinstance(max_exploration_rate, float): raise TypeError('max_exploration_rate must be a float.')
        if not 0 <= max_exploration_rate <= 1: raise ValueError('max_exploration_rate must be between 0 and 1 inclusive.')
        if not isinstance(min_exploration_rate, float): raise TypeError('min_exploration_rate must be a float.')
        if not 0 <= min_exploration_rate <= 1: raise ValueError('min_exploration_rate must be between 0 and 1 inclusive.')
        if not max_exploration_rate > min_exploration_rate: raise ValueError('max_exploration_rate must be greater than min_exploration_rate.')
        if not max_exploration_rate + min_exploration_rate <= 2: raise ValueError('max_exploration_rate + min_exploration_rate must be less than or equal to 1.')
        if not min_exploration_rate <= exploration_rate <= max_exploration_rate: raise ValueError('exploration_rate must be between min_exploration_rate and max_exploration_rate inclusive.')
        if not isinstance(exploration_decay_rate, float): raise TypeError('exploration_decay_rate must be a float.')
        if not 0 <= exploration_decay_rate <= 1: raise ValueError('exploration_decay_rate must be between 0 and 1 inclusive.')

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
    

    def train(self) -> None:
        for episode in range(self.max_episodes):
            print(f'-------- Episode {episode} ----------')
            pos, velocity = self.env.reset()
            for _ in range(self.max_steps):
                curr_state = self.get_state_for_position_and_velocity(float(truncate(pos, 2)), float(truncate(velocity, 2)))
                rewards = self.rewards[curr_state] 
                # Exploration-exploitation trade-off
                exploration_rate_threshold = random.uniform(0, 1)
                if exploration_rate_threshold > self.exploration_rate:
                    action = max(rewards, key=rewards.get)
                else:
                    action = self.env.action_space.sample()
                # Take the action.
                observation, gym_reward, done, info = self.env.step(action)
                # Update the Q-table for the current state-action pair.
                self.rewards[curr_state][action] = self.rewards[curr_state][action] * (1 - self.learning_rate) + \
                    self.learning_rate * (gym_reward + self.discount_factor * max(self.rewards[curr_state].values()))
                # Update the state.
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
            # Exploration rate decay.
            self.exploration_rate = self.min_exploration_rate + \
                (self.max_exploration_rate - self.min_exploration_rate) * \
                exp(-self.exploration_decay_rate * episode)

        self.env.close()
