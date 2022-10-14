from types import NoneType
from math import sqrt
import gym
from typing import Union
from gym.envs.toy_text.frozen_lake import generate_random_map


class FrozenLake:
    def __init__(self, discount_factor:float=0.99, desc:Union[int, NoneType]=None, map_name:str='4x4', is_slippery:bool=True) -> None:
        """ Initialize the class.

        Args:
            discount_factor (float, optional): The discount factor. Defaults to 0.99.
            desc (Union[int, NoneType], optional): The description of the map. If an integer, it creates a random map with the specified size. Defaults to None (4x4 or 8x8 as specified at map_name).
            map_name (str, optional): The name of the map. Defaults to '4x4'.
            is_slippery (bool, optional): If the environment is slippery or not. Defaults to True.

        Raises:
            TypeError: If the map_name is not a string.
            TypeError: If the discount factor is not a float.
            TypeError: If the description is not an int or None.
            TypeError: If the is_slippery is not a bool.
            ValueError: If the map_name is not '4x4' or '8x8'.
            ValueError: If the discount factor is not between 0 and 1.

        Returns:
            None
        """
        if not isinstance(discount_factor, float): raise TypeError("The discount factor must be a float.")
        if not isinstance(desc, (int, NoneType)): raise TypeError("The description must be an int or None.")
        if not isinstance(map_name, str): raise TypeError("The map_name must be a string.")
        if not isinstance(is_slippery, bool): raise TypeError("The is_slippery must be a bool.")

        if map_name not in ['4x4', '8x8']: raise ValueError("The map_name must be '4x4' or '8x8'.")
        if discount_factor < 0 or discount_factor > 1: raise ValueError("The discount factor must be between 0 and 1.")

        desc = generate_random_map(size=desc) if desc is not None else None
        self.env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=is_slippery)
        self.map_size = int(map_name.split('x')[0]) ** 2
        self.is_slippery = is_slippery
        self.transition_probability = 1/3 if is_slippery else 1
        self.discount_factor = discount_factor
        self.actions_list = [0, 1, 2, 3]
        self.n_actions = 4
        

    #Helper methods for this environment
    def _possible_actions(s:int, map_size:int) -> list:
        """ Given a position, return the list of possible actions available by their numeric codes.

        Args:
            pos (int): The position in the grid, linearly, from 0 to map_size, calculated as current_row * nrows + current_col.
            map_size (int): The size of the map (number of states).

        Returns:
            list: The list of possible actions available at the current position.
        """    
        i, j = FrozenLake._from_pos_to_coordinates(s, map_size)
        possible_actions_list = []
        if i != 0:
            possible_actions_list.append(3)
        if i != 3:
            possible_actions_list.append(1)
        if j != 0:
            possible_actions_list.append(0)
        if j != 3:
            possible_actions_list.append(2)
        return possible_actions_list


    def _from_pos_to_coordinates(pos:int, map_size:int) -> tuple:
        """ Given a position, return the coordinates in the grid.

        Args:
            pos (int): The position in the grid, linearly, from 0 to map_size, calculated as current_row * nrows + current_col.
            map_size (int): The size of the map (number of states).

        Returns:
            tuple: The coordinates of the position in the grid.
        """    
        i = pos // sqrt(map_size)
        j = pos % sqrt(map_size)
        return i, j


    def _movement(pos:int, action:int, is_slippery:bool, map_size:int) -> list:
        """ Given a position and an action, return the new position. Depends if it is slippery or not.

        Args:
            pos (int): The position in the grid, linearly, from 0 to map_size, calculated as current_row * nrows + current_col.
            action (int): The action to perform.
            is_slippery (bool): If the environment is slippery or not.
            map_size (int): The size of the map (number of states).

        Returns:
            list: The new positions after performing the action. It will only have 1 element in the non slippery case.
        """    
        new_pos = []
        i, j = FrozenLake._from_pos_to_coordinates(pos, map_size)
        possible_actions_list = FrozenLake._possible_actions(pos, map_size)
        #Non slippery case
        if action in possible_actions_list:
            #No random weird initialization
            if action == 0:
                #Left
                new_pos.append(i * sqrt(map_size) + j - 1)
            elif action == 1:
                #Down
                new_pos.append((i + 1) * sqrt(map_size) + j)
            elif action == 2:
                #Right
                new_pos.append(i * sqrt(map_size) + j + 1)
            else:
                #Up
                new_pos.append((i - 1) * sqrt(map_size) + j)
        if is_slippery:
            #Slippery case
            if action == 0 or action == 2:
                if 1 in possible_actions_list:
                    new_pos.append((i + 1) * sqrt(map_size) + j)
                if 3 in possible_actions_list:
                    new_pos.append((i - 1) * sqrt(map_size) + j)
            elif action == 1 or action == 3:
                if 0 in possible_actions_list:
                    new_pos.append(i * sqrt(map_size) + j - 1)
                if 2 in possible_actions_list:
                    new_pos.append(i * sqrt(map_size) + j + 1)
        
        new_pos = list(map(int, new_pos))

        return new_pos


    def _reward(pos:int, map_size:int) -> int:
        """ Given a position, return the reward of the position.
        Goal -> 1
        The rest -> 0

        Args:
            pos (int): The position in the grid, linearly, from 0 to map_size, calculated as current_row * nrows + current_col.
            map_size (int): The size of the map (number of states).

        Returns:
            int: The reward of the position.
        """    
        i, j = FrozenLake._from_pos_to_coordinates(pos, map_size)
        if i == 0 and j == 0:
            #Starting point
            return 0
        if (i == 1 and j == 1) or \
            (i == 1 and j == 3) or \
            (i == 2 and j == 3) or \
            (i == 3 and j == 0):
            #Hole
            return -1
        if i == 3 and j == 3:
            #Goal
            return 1
        return 0
