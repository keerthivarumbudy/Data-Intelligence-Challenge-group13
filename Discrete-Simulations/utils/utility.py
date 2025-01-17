import numpy as np

orients = {'n': -3, 'e': -4, 's': -5, 'w': -6}
dirs = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0)}

def get_reward(state, action):
    # Get the possible values (dirty/clean) of the tiles we can end up at after a move:
    new_pos = tuple(np.array(state.pos) + dirs[action])

    reward_dict = {
        -3: -2,
        -2: -2,
        -1: -2,
        0: -1,
        1: 1,
        2: 10,
        3: -10
    }
    state_reward = reward_dict[state.grid.cells[new_pos]]

    # modified from environment.py
    # TODO: correct?
    expected_drain = state.battery_drain_p * np.random.exponential(state.battery_drain_lam)
    print("expected_drain:", expected_drain)

    # reward is reward of moving to new state - expected battery drain (negative constant)
    reward = state_reward - expected_drain

    return reward
    
def updated_get_reward(new_state):
    """
    Get the reward of the new state given an old state. 
    
    Parameters
    ----------
    new_state : dict
        the new state
        
    Returns
    -------
    reward : int
        the reward of the new state 
    """
    # get the reward of the new state
    # expected_drain = state.battery_drain_p * np.random.exponential(state.battery_drain_lam)
    
    terminal_reward = 0
    reward = 0
    
    if (new_state['termination_reason'] == 'death'):
        terminal_reward = -10000
    
    elif (new_state['termination_reason'] == 'goal'):
        terminal_reward = 10000
        
    if (new_state['is_goal']):
        reward += 100 # reward for reaching goal
    
    if (new_state['is_terminal']):
        return reward + terminal_reward
    
    elif(new_state['is_goal']): 
        return reward 
        
    else:
        return -1
                    

def get_reward_dict(state_dict, action):
    # Get the possible values (dirty/clean) of the tiles we can end up at after a move:
    robot_pos = next((((list(row).index(-3.)), y)
      for y, row in enumerate(state_dict['grid'])
      if -3. in row),
     None)
    new_pos = tuple(map(sum, zip(robot_pos, dirs[action])))

    reward_dict = {
        -3: -2,
        -2: -2,
        -1: -2,
        0: -1,
        1: 1,
        2: 10,
        3: -10
    }
    state_reward = reward_dict[state_dict['grid'][new_pos[1]][new_pos[0]]]

    # modified from environment.py
    # TODO: correct?
    # expected_drain = state_dict.battery_drain_p * np.random.exponential(state_dict.battery_drain_lam)
    # print("expected_drain:", expected_drain)

    # reward is reward of moving to new state - expected battery drain (negative constant)
    reward = state_reward# - expected_drain

    return reward