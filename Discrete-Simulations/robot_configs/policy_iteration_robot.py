class State:
    # state includes the robot's position and orientation
    def __init__(self, grid, pos, orientation, p_move=0, battery_drain_p=0, battery_drain_lam=0):
        self.pos = pos
        self.grid = grid

# Policy Iteration based robot:
def robot_epoch(robot):
    robot.find_and_do_move()