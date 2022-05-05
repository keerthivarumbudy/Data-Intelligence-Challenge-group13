from copy import deepcopy
from collections import OrderedDict

from utils.utility import *

materials =  {'cell_clean': 0,
'cell_wall': -1,
'cell_obstacle': -2, 
'cell_robot_n': -3, 
'cell_dirty': 1,
'cell_goal': 2,
'cell_death': 3,
'cell_robot_dead_body': 4,
}

# # a 4x4 grid to test with
# grid = [[-1, -1, -1, -1],
#         [-1, -3, 1, -1],
#         [-1, 1, 1, -1],
#         [-1, -1, -1, -1]]
        
        
def get_state_id(grid):
    """
    Given a 2D grid (list of lists), returns a flattened ID corresponding to that grid. 
    
    Parameters
    ----------
    state : [[int]]
        A 2D grid (list of lists) of integers.
    
    Returns
    -------
    str
        A flattened string ID corresponding to the input grid
    """
    
    # Good to note, length of strings does not impact dictionary performance.
    return ''.join([str(i) for i in grid])
    
def all_clean(grid):
    """
    Given a 2D grid (list of lists), returns True if there are no dirty cells.
    """
    for row in grid:
        for cell in row:
            if cell == 1:
                return False
    return True
    
def robot_dead(grid):
    """
    Given a 2D grid (list of lists), returns True if the robot is dead.
    """
    for row in grid:
        for cell in row:
            if cell == materials['cell_robot_n']:
                return False
    return True
    
def get_state_from_action(grid, action):
    """
    Given a 2D grid (list of lists), and an action, returns the grid that results from that action.
    
    Parameters
    ----------
    grid : [[int]]
        A 2D grid (list of lists) of integers.
        
    action : str
        A string representing the action to take.
        
    Returns
    -------
    [[int]]
        The grid that results from the action.
        
    Raises
    ------
    ValueError
        If the action is not one of the four cardinal dirs.
    """
    
    if action not in dirs:
        raise ValueError('Action must be one of the four cardinal dirs.')
        
    # Get the direction
    direction = dirs[action]
    # direction is a tuple of (x, y)
    
    # Get the new grid
    new_grid = deepcopy(grid)
    for y in range(len(grid)):
        # looping through rows (y)
        for x in range(len(grid[y])):
            # looping through columns (x)
            if grid[y][x] == materials['cell_robot_n']:
                if (grid[y+direction[1]][x+direction[0]] == materials['cell_wall'] or grid[y+direction[1]][x+direction[0]] == materials['cell_obstacle']):
                    return new_grid
                    
                elif(grid[y+direction[1]][x+direction[0]] == materials['cell_death']):
                    new_grid[y+direction[1]][x+direction[0]] = materials['cell_robot_dead_body']
                    new_grid[y][x] = materials['cell_clean']
                    
                else: 
                    new_grid[y+direction[1]][x+direction[0]] = materials['cell_robot_n']
                    new_grid[y][x] = materials['cell_clean']
                    
                break
    return new_grid
    
    
    
    
def generate_reachable_states(state, state_dict = OrderedDict()):
    """
    Given a 2D grid (list of lists), returns all reachable states with some metadata.
    
    Parameters
    ----------
    state : [[int]]
        A 2D grid (list of lists) of integers.
    
    Returns
    -------
    dict
        A dictionary mapping each state to a state_id, with each state_id
        corresponding to a state that is reachable from the input state. In addition to that, it shows if a state is a terminal state, and why is it terminal. 
    """
    
    state_id = get_state_id(state)
    state_dict[state_id] = {'grid': state, 'immediately_reachable_states': {} }
    
    # Check if state is all clean
    if (all_clean(state)):
        state_dict[state_id]['is_terminal'] = True
        state_dict[state_id]['terminal_reason'] = 'goal'
        return state_dict
        
    if (robot_dead(state)):
        state_dict[state_id]['is_terminal'] = True
        state_dict[state_id]['terminal_reason'] = 'death'
        return state_dict
        
    # Check all possible dirs
    for direction in dirs.keys(): 
        new_state = get_state_from_action(state, direction)
        new_state_id = get_state_id(new_state)
        if new_state_id not in state_dict:
            state_dict = generate_reachable_states(new_state, state_dict)
        
        if (direction not in state_dict[state_id]['immediately_reachable_states']):
            state_dict[state_id]['immediately_reachable_states'][direction] = new_state_id
        #     state_dict[state_id]['immediately_reachable_states'][new_state_id] = {'direction': [direction]}
        # else:
        #     state_dict[state_id]['immediately_reachable_states'][new_state_id]['direction'].append(direction)
    return state_dict


def add_location_to_grid(grid, location):
    """
    Given a 2D grid (list of lists), and a location, adds a robot to the grid.
    """
    grid[location[1]][location[0]] = materials['cell_robot_n']
    return grid



# # print number of keys in the dictionary
# print(generate_reachable_states(grid))