import numpy as np

orients = {'n': -3, 'e': -4, 's': -5, 'w': -6}
dirs = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0)}

def get_reward(state, action):
    # Get the possible values (dirty/clean) of the tiles we can end up at after a move:
    new_pos = tuple(np.array(state.pos) + dirs[action])

    reward_dict = {
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

def get_reward_dict(state_dict, action):
    # Get the possible values (dirty/clean) of the tiles we can end up at after a move:
    new_pos = tuple(np.array(state_dict['loc']) + dirs[action])

    reward_dict = {
        -3: -2,
        -2: -2,
        -1: -2,
        0: -1,
        1: 1,
        2: 10,
        3: -10
    }
    state_reward = reward_dict[state_dict['grid'].cells[new_pos]]

    # modified from environment.py
    # TODO: correct?
    # expected_drain = state_dict.battery_drain_p * np.random.exponential(state_dict.battery_drain_lam)
    # print("expected_drain:", expected_drain)

    # reward is reward of moving to new state - expected battery drain (negative constant)
    reward = state_reward# - expected_drain

    return reward