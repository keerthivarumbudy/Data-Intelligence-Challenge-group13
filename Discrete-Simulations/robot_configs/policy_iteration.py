import numpy as np

from robot_configs.utility import *

class State:
    # state includes the robot's position and orientation
    def __init__(self, grid, pos, orientation, p_move=0, battery_drain_p=0, battery_drain_lam=0):
        # if grid.cells[pos[0], pos[1]] != 1:
        #     raise ValueError
        self.orientation = orientation
        self.pos = pos
        self.grid = grid
        self.grid.cells[pos] = orients[self.orientation]
        self.p_move = p_move
        self.alive = True
        self.battery_drain_p = battery_drain_p
        self.battery_drain_lam = battery_drain_lam
        self.policy = None

    # Policy Evaluation

    def init_policy(self):
        # self.policy = np.zeros(self.grid.cells.shape)
        # randomly assign policies
        orientations = [i for i in orients.keys()]
        self.policy = np.random.choice(orientations, self.grid.cells.shape)


    def calculate_values(self):
        # calculate the value of each state

        pass

    def sweep(self, no_sweeps=1):
        # sweep the grid and update values until convergence
        for _ in range(no_sweeps):
            self.calculate_values()


    # Policy Improvement

    def update_policy(self):
        # update the policy greedily based on values of accessible states
        pass

# Move
def move(self):
    # move the robot according to the policy

    pass

def robot_epoch(robot):
    # one epoch of the robot
    state = State(robot.grid, robot.pos, robot.orientation)
    state.init_policy()
    print(state.policy)
    print('robot orientation:', robot.orientation)
    print(get_reward(robot,robot.orientation))

    # state.sweep()

    # state.update_policy()
    # robot.orientation =
    # robot.move_to_position()

