# Import our robot algorithm to use in this simulation:
import copy
import time
import numpy as np

from robot_configs.policy_iteration_robot import robot_epoch
import pickle
from policy_iteration import DumbRobot
import matplotlib.pyplot as plt

grid_file = 'death.grid' #'example-random-house-0.grid'  # simple-random-house-0.grid'
# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

with open(f'grid_configs/{grid_file}', 'rb') as f:
    grid = pickle.load(f)

# Spawn the robot at (1,1) facing north with battery drainage enabled:
# print("The grid is:", grid.cells)
robot = DumbRobot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2, gamma=0.9)
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
        try:
            robot_epoch(robot)
        except ValueError:
            break
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

# Make some plots:
plt.hist(cleaned)
plt.title('Percentage of tiles cleaned.')
plt.xlabel('% cleaned')
plt.ylabel('count')
plt.show()

plt.hist(efficiencies)
plt.title('Efficiency of robot.')
plt.xlabel('Efficiency %')
plt.ylabel('count')
plt.show()