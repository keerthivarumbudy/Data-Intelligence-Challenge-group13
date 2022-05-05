import numpy as np

from utils.utility import *

class State:
    # state includes the robot's position and orientation
    def __init__(self, grid, pos, orientation, p_move=0, battery_drain_p=0, battery_drain_lam=0):
        self.pos = pos
        self.grid = grid


# Move
def move(self):
    # move the robot according to the policy

    pass

def robot_epoch(robot):
    # one epoch of the robot
    state = State(robot.grid, robot.pos, robot.orientation, gamma=robot.gamma)
    print(state.policy)
    print('robot orientation:', robot.orientation)
    print()

    # state.sweep()

    # state.update_policy()
    # robot.orientation =
    # robot.move_to_position()

