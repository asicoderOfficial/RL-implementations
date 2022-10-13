from itertools import count
from numpy.random import dirichlet
from src.frozen_lake.frozen_lake import FrozenLake
from numpy import ones
from random import choice
from types import NoneType
from typing import Union


class FLPolicyIteration(FrozenLake):
    def __init__(self, theta:float=0.01, max_iterations:int=10, discount_factor:float=0.99, desc:Union[int, NoneType]=None, map_name:str='4x4', is_slippery:bool=True) -> None:
        """ Initialize the class.

        Args:
            theta (float, optional): The difference threshold between the current value and the new value, for which at max, the algorithm will stop. Defaults to 0.01.
            max_iterations (int, optional): The maximum number of iterations, times to repeat the value iteration algorithm. Defaults to 10.
            discount_factor (float, optional): The discount factor. It makes more important the rewards from the start, and less the ones at the end. Defaults to 0.99.
            desc (Union[int, NoneType], optional): The description of the map. Defaults to None.
            map_name (str, optional): The name of the map. Defaults to '4x4'.
            is_slippery (bool, optional): Whether the environment is slippery or not. Defaults to True.
        
        Raises:
            TypeError: If theta is not a float.
            ValueError: If max_iterations is not greater than 0.
            ValueError: If theta is not greater than 0.
        
        Returns:
            None
        """

        super().__init__(discount_factor=discount_factor, desc=desc, map_name=map_name, is_slippery=is_slippery)

        if not isinstance(theta, float): raise TypeError('theta must be a float.')
        if max_iterations <= 0: raise ValueError('max_iterations must be greater than 0.')
        if theta <= 0: raise ValueError('theta must be greater than 0.')

        self.theta = theta
        self.max_iterations = max_iterations
        self.value_table = [-1 for _ in range(self.map_size)]
        """
        optimal_policy = []
        for s in range(self.map_size):
            curr_optimal_policy = []
            possible_actions = FrozenLake._possible_actions(s, self.map_size)
            #Use the dirichlet distribution, as all values are in range [0, 1] and sum up to 1.
            random_possible_actions_probabilities = dirichlet(ones(len(possible_actions)), size=1)
            random_possible_actions_idx = 0
            for action in self.actions_list:
                if action in possible_actions:
                    #Add a random initial probability of picking that possible action.
                    curr_optimal_policy.append(random_possible_actions_probabilities[0][random_possible_actions_idx])
                    random_possible_actions_idx += 1
                else:
                    #It is not possible to pick that action, assign it the probability 0.
                    curr_optimal_policy.append(0)
            optimal_policy.append(curr_optimal_policy)
        self.optimal_policy = optimal_policy
        """
        self.optimal_policy = [choice(self.actions_list) for _ in range(self.map_size)]
        self.goal_state = 15 if map_name == '4x4' else 63
    

    def render_policy(self) -> None:
        """ Render the policy. """
        curr_cell = self.env.reset()
        while curr_cell != self.goal_state:
            action = self.optimal_policy[curr_cell]
            curr_cell, _, _, _ = self.env.step(action)
            self.env.render()
        

    def train(self) -> None:
        """ Train the model. """
        self.env.reset()
        prev_optimal_policy = self.optimal_policy.copy()
        for i in range(self.max_iterations):
            print()
            print(i)
            self.value_table, self.optimal_policy = FLPolicyIteration._policy_iteration_iterative(self.env, self.theta, self.value_table, self.optimal_policy, self.transition_probability, self.discount_factor, self.map_size, self.is_slippery)
            if self.optimal_policy == prev_optimal_policy:
                break
            prev_optimal_policy = self.optimal_policy.copy()
        self.render_policy()


    def _policy_iteration_iterative(env, theta:int, value_table:int, optimal_policy:list, transition_probability:float, discount_factor:float, map_size:int, is_slippery:bool) -> tuple:
        """ Compute one iteration of the value iteration algorithm.

        Args:
            i (int): Row index  of the position in the grid.
            j (int): Column index of the position in the grid.
            theta (int): The difference threshold between the current value and the new value, for which at max, the algorithm will stop.
            value_table (int): The value table, where the expected return of each state is stored.
            optimal_policy (list): The actions table, where the best action for each state is stored.

        Returns:
            tuple: The value table and the actions table.
        """    
        delta = 0
        policy_stable = False
        while not policy_stable:
            #Policy evaluation
            while theta > delta:
                for s in range(map_size):
                    v = value_table[s]
                    curr_optimal_action = optimal_policy[s]
                    all_possible_optimal_states_from_s = FrozenLake._movement(s, curr_optimal_action, is_slippery, map_size)
                    value_table[s] = sum([(1 / len(all_possible_optimal_states_from_s)) * (FrozenLake._reward(new_s, map_size) + discount_factor * value_table[new_s]) for new_s in all_possible_optimal_states_from_s])
                    delta = max(delta, abs(v - value_table[s]))

            #Policy improvement
            for s in range(map_size):
                old_action = optimal_policy[s]
                action_value_dict = {action : sum([1 / len(FrozenLake._movement(s, action, is_slippery, map_size)) * (FrozenLake._reward(s, map_size) + discount_factor * value_table[new_s]) for new_s in FrozenLake._movement(s, action, is_slippery, map_size)]) for action in FrozenLake._possible_actions(s, map_size)}
                optimal_policy[s] = max(action_value_dict, key=action_value_dict.get)
                if old_action == optimal_policy[s]:
                    policy_stable = True

        return value_table, optimal_policy
