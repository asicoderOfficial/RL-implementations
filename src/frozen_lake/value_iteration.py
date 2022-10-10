
#With value iteration (dynamic programming), we will compute all the subproblems until we fill the table, and then play the game, starting from s0,
#as the table does not change
PROBABILITY = 1
DISCOUNT_FACTOR = 0.01
ACTIONS = [0, 1, 2, 3]

#Helper methods for this environment
def possible_actions(i:int, j:int) -> list:
    """ Given a position, return the list of possible actions available by their numeric codes.

    Args:
        i (int): Row index  of the position in the grid.
        j (int): Column index of the position in the grid.

    Returns:
        list: The list of possible actions available at the current position.
    """    
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


def movement(i:int, j:int, action:int) -> tuple:
    """ Given a position and an action, return the new position.

    Args:
        i (int): Row index  of the position in the grid.
        j (int): Column index of the position in the grid.
        action (int): The action to be performed.

    Returns:
        tuple: The new position after performing the action.
    """    
    if action == 0:
        #Left
        j -= 1
    elif action == 1:
        #Down
        i += 1
    elif action == 2:
        #Right
        j += 1
    else:
        #Up
        i -= 1

    return i, j
            


def reward(i:int, j:int) -> int:
    """ Given a position, return the reward of the position.
    Goal -> 1
    The rest -> 0

    Args:
        i (int): Row index  of the position in the grid.
        j (int): Column index of the position in the grid.

    Returns:
        int: The reward of the position.
    """    
    if i == 0 and j == 0:
        #Starting point
        return 0
    if (i == 1 and j == 1) or \
        (i == 1 and j == 3) or \
        (i == 2 and j == 3) or \
        (i == 3 and j == 0):
        #Hole
        return 0
    if i == 3 and j == 3:
        return 1
    return 0


def value_iteration_iterative(i:int, j:int, theta:int, value_table:int, actions_table:int) -> tuple:
    """ Compute one iteration of the value iteration algorithm.

    Args:
        i (int): Row index  of the position in the grid.
        j (int): Column index of the position in the grid.
        theta (int): The difference threshold between the current value and the new value, for which at max, the algorithm will stop.
        value_table (int): The value table, where the expected return of each state is stored.
        actions_table (int): The actions table, where the best action for each state is stored.

    Returns:
        tuple: The value table and the actions table.
    """    
    delta = 0
    while delta < theta:
        for i in range(4):
            for j in range(4):
                v = value_table[i][j]
                actions = possible_actions(i, j)
                action_values = {}
                for action in actions:
                    new_i, new_j = movement(i, j, action)
                    r = reward(new_i, new_j)
                    action_values[action] = PROBABILITY * (r + DISCOUNT_FACTOR * value_table[new_i][new_j])
                best_action = max(action_values, key=action_values.get)
                value_table[i][j] = action_values[best_action]
                actions_table[i][j] = best_action
                delta = max(delta, abs(v - value_table[i][j]))

    return value_table, actions_table


def value_iteration_recursive(i:int, j:int, v:int, curr_trajectory:int, actions:int) -> tuple:
    """ Compute the value iteration algorithm recursively.

    Args:
        i (int): Row index  of the position in the grid.
        j (int): Column index of the position in the grid.
        v (int): The values matrix, where the expected return of each state is stored.
        curr_trajectory (int): All the states visited so far. Used to avoid infinite cycles.
        actions (int): The actions matrix, where the best action for each state is stored.

    Returns:
        tuple: The values matrix and the actions matrix.
    """    
    if i == 3 and j == 3:
        return (v, actions)

    all_possible_actions = possible_actions(i, j) 
    all_action_values = {}
    new_i, new_j = (i+1, j) if i < 3 else (i, j+1)
    for action in all_possible_actions:
        if (new_i, new_j) not in curr_trajectory:
            curr_trajectory += [(new_i, new_j)]
            new_v = PROBABILITY * (reward(new_i, new_j) + DISCOUNT_FACTOR * value_iteration_recursive(new_i, new_j, v, curr_trajectory, actions)[0][new_i][new_j])
            all_action_values[action] = new_v
            v[i][j] = max(new_v, v[i][j])

    if all_action_values != {}:
        actions[i][j] = max(all_action_values, key=all_action_values.get)
    return (v, actions)


#It represents the optimal action for each i,j state (each cell the agent is at).
#optimal_actions_table = [[-1] * 4] * 4
optimal_values = [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 0]]
optimal_actions =[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 0]] 
prev_optimal_actions = []
k = 1
while k <= 5:
    print(f'Iteration {k}')
    print('VALUES')
    for i in range(4):
        print(optimal_values[i])
    print()
    print('ACTIONS')
    for i in range(4):
        print(optimal_actions[i])
    print('---------')
    if optimal_actions == prev_optimal_actions:
        break
    else:
        prev_optimal_actions = optimal_actions

    optimal_values, optimal_actions = value_iteration_recursive(0, 1, optimal_values, [], optimal_actions)

    k += 1
