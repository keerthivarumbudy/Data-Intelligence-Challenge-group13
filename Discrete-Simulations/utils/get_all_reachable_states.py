from copy import deepcopy
from collections import OrderedDict

from utils.utility import *
# dirs = {'n': (0, 1), 's': (0, -1), 'e': (1, 0), 'w': (-1, 0)}

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
grid = [[-1, -1, -1, -1],
        [-1, -3, 1, -1],
        [-1, 2, 1, -1],
        [-1, -1, -1, -1]]
        
        
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
    
    
def get_state_from_action(grid, action):
    """
    Given a 2D grid (list of lists), and an action, returns a tuple of the following:
    (new_grid, is_goal_state)
    
    
    Parameters
    ----------
    grid : [[int]]
        A 2D grid (list of lists) of integers.
        
    action : str
        A string representing the action to take.
        
    Returns
    -------
    tuple
        A tuple of the following:
        grid: [[int]]
            The grid that results from the action.
        is_goal_state: bool
            True if the grid is a goal state (the robot picked up a goal tile), False otherwise.
        
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
    
    is_goal_state = False
    # Get the new grid
    new_grid = deepcopy(grid)
    for y in range(len(grid)):
        # looping through rows (y)
        for x in range(len(grid[y])):
            # looping through columns (x)
            
            # find robot location
            if grid[y][x] == materials['cell_robot_n']:
                
                # if the actions leads to a wall, do nothing
                if (grid[y+direction[1]][x+direction[0]] == materials['cell_wall'] or grid[y+direction[1]][x+direction[0]] == materials['cell_obstacle']):
                    return new_grid, is_goal_state
                    
                # if the actions leads to a death, mark it.
                elif(grid[y+direction[1]][x+direction[0]] == materials['cell_death']):
                    new_grid[y+direction[1]][x+direction[0]] = materials['cell_robot_dead_body']
                    new_grid[y][x] = materials['cell_clean']
                    
                # Mark the new location of the robot, and make the old location clean.
                else: 
                    new_grid[y+direction[1]][x+direction[0]] = materials['cell_robot_n']
                    new_grid[y][x] = materials['cell_clean']
                    
                # if the cell is a goal, change the boolean
                if(grid[y+direction[1]][x+direction[0]] == materials['cell_goal']):
                    is_goal_state = True
                    
                break
    return new_grid, is_goal_state
    

def get_state_info(grid):
    """
    Given a 2D grid (list of lists), returns a tuple of the following:
    (clean_tiles_number, is_state_terminal_bool, termination_reason, is_state_goal_bool)
    
    Parameters
    ----------
    grid : [[int]]
        A 2D grid (list of lists) of integers.
        
    Returns
    -------
    tuple
        A tuple of the following:
        clean_tiles_number: int
            The number of clean tiles in the grid.
        is_state_terminal_bool: bool
            True if the grid is a terminal state, False otherwise.
        termination_reason: str
            The reason the grid is a terminal state. Either 'goal' or 'death'.
            'goal' if the grid is a goal state (all tiles clean), 'death' if the grid is a death state (the robot died or is dead).
    """
    
    is_terminal = False
    termination_reason = None
    robot_alive = False
    all_tiles_clean = True
    clean_tiles_number = 0
    for row in grid:
        for cell in row:
            if cell == materials['cell_clean']:
                clean_tiles_number += 1
            if cell == materials['cell_robot_n']:
                robot_alive = True
            if cell == materials['cell_dirty'] or cell == materials['cell_goal']:
                all_tiles_clean = False
    if all_tiles_clean:
        is_terminal = True
        termination_reason = 'goal'
    elif not robot_alive:
        is_terminal = True
        termination_reason = 'death'
    return clean_tiles_number, is_terminal, termination_reason
                
    
def generate_reachable_states(state, state_dict = OrderedDict(), is_state_goal_bool = False):
    """
    Given a 2D grid (list of lists), returns all reachable states with some metadata.
    
    Parameters
    ----------
    state : [[int]]
        A 2D grid (list of lists) of integers.
        
    state_dict : OrderedDict
        A dictionary represeneting states and their metadata.
        looks like: 
        {
            'state_id': {
                'state': [[int]],
                'is_goal_state': bool,
                'clean_tiles_number': int,
                'is_terminal': bool,
                'termination_reason': str,
                'immediately_reachable_states': [str],
            }, 
            ....
        }
        
    
    Returns
    -------
    tuple
        A tuple of the following:
        state_dict: OrderedDict
            The dictionary represeneting states and their metadata.
            
        is_state_goal_bool: bool
            True if the grid is a goal state, False otherwise.
    """
    
    state_id = get_state_id(state)
    clean_tiles_number, is_terminal, termination_reason = get_state_info(state)
    
    state_dict[state_id] = {'grid': state, 'clean_tiles': clean_tiles_number, 'is_terminal': is_terminal, 'termination_reason': termination_reason,
    'is_goal': is_state_goal_bool,'immediately_reachable_states': {}}
    
        
    if (is_terminal):
        return state_dict
        
    # Check all possible dirs
    for direction in dirs.keys(): 
        
        new_state, is_state_goal = get_state_from_action(state, direction)
        new_state_id = get_state_id(new_state)
        
        if new_state_id not in state_dict:
            state_dict = generate_reachable_states(new_state, state_dict, is_state_goal)
            
        if (direction not in state_dict[state_id]['immediately_reachable_states']) & (state_id != new_state_id):
            state_dict[state_id]['immediately_reachable_states'][direction] = new_state_id

    return state_dict


def add_location_to_grid(grid, location):
    """
    Given a 2D grid (list of lists), and a location, adds a robot to the grid.
    """
    grid[location[1]][location[0]] = materials['cell_robot_n']
    return grid



# # print number of keys in the dictionary
# print(generate_reachable_states(grid))