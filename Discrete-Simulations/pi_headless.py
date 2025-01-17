# Import our robot algorithm to use in this simulation:
import copy
import time
import numpy as np

from robot_configs.policy_iteration_robot import robot_epoch
import pickle
from policy_iteration import DumbRobot
import matplotlib.pyplot as plt

import pandas as pd

runs_df = pd.DataFrame()

randomness_move = [0, 0.25, 0.5,  0.75]
randomness = 0.25
gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
g = 0.9

grid_file = 'death.grid' #'example-random-house-0.grid'  # simple-random-house-0.grid'
# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

with open(f'grid_configs/{grid_file}', 'rb') as f:
    grid = pickle.load(f)

# Spawn the robot at (1,1) facing north with battery drainage enabled:
# print("The grid is:", grid.cells)
robot = DumbRobot(grid, (1, 1), orientation='n', p_move=randomness, battery_drain_p=0.0, battery_drain_lam=0, gamma=g)
# print("ROBOT.V=", robot.values)
# print("ROBOT.Policy=", robot.policy)


# def print_V_policy(robot):
#     for s_key in list(robot.all_states.keys())[:20]:

#         current_state = robot.all_states[s_key]
#         # print("GRID: \n", current_state.grid.cells)
#         # print("POS: ", current_state.pos)
#         # print("VALUE: ", robot.V[s_key])
#         # print("POLICY: ", robot.policy[s_key])

#         nb_states = current_state.get_neighbouring_states()
#         # print(robot.policy[s_key])
#         for s in nb_states:
#             # print("NEIGHBOR")
#             # print(s.grid.cells)
#             # print(s.pos)
#             # print(s.orientation)
#             grid_key, pos_key = get_state_key(s)
#             s_val = robot.V[(grid_key, pos_key)]
#             # print(s_val)


#print_V_policy(robot)

# Keep track of some statistics:
efficiencies = []
n_moves = []
deaths = 0
cleaned = []

#Run 100 times:
for i in range(1):
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    robot_i = copy.deepcopy(robot)
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    while True:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        robot_epoch(robot)
        
        # for state in range(len(robot.values)):
        #     # print("ROBOT.V=", robot.values[state])
        #     # print("ROBOT.Policy=", robot.policy[state])
        #     # print("ROBOT.states=", list(robot.S.keys())[state])
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths += 1
            break
        # Calculate some statistics:
        clean = (robot.grid.cells == 0).sum()
        dirty = (robot.grid.cells >= 1).sum()
        goal = (robot.grid.cells == 2).sum()
        death_tiles = (robot.grid.cells == 3).sum()
        # Calculate the cleaned percentage:
        clean_percent = (clean / (dirty + clean - death_tiles)) * 100
        # print(robot.grid.cells)
        # print('CLEANED: ', clean_percent)
        # See if the room can be considered clean, if so, stop the simulaiton instance:
        if clean_percent >= stopping_criteria and goal == 0:
            break
        # Calculate the effiency score:
        moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
        u_moves = set(moves)
        n_revisted_tiles = len(moves) - len(u_moves)
        efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
        # print("Move info:")
        # print("Move:", moves)
        # print("clean percent:", clean_percent)
        # print("efficiency:", efficiency)
        #time.sleep(5)
    # Keep track of the last statistics for each simulation instance:
    efficiencies.append(float(efficiency))
    n_moves.append(len(robot.history[0]))
    cleaned.append(clean_percent)
    
    print('run: ', i, 'clean_percent: ', clean_percent, 'efficiency: ', efficiency, 'n_moves: ', len(robot.history[0]))
    runs_df = runs_df.append(pd.Series([grid_file, efficiency, 0.0, len(robot.history[0]), 0.0, clean_percent, 0.0, randomness, g], index=runs_df.columns), ignore_index=True)

runs_df.to_csv(f'text/{randomness}_results.csv')
runs_df = pd.DataFrame()

# # Make some plots:
# plt.hist(cleaned)
# plt.title('Percentage of tiles cleaned.')
# plt.xlabel('% cleaned')
# plt.ylabel('count')
# plt.show()

# plt.hist(efficiencies)
# plt.title('Efficiency of robot.')
# plt.xlabel('Efficiency %')
# plt.ylabel('count')
# plt.show()