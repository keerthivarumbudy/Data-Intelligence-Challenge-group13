
if __name__ == '__main__':
    # some_file.py
    import pickle
    from environment import DumbRobot

    with open(f'grid_configs/example-random-house-0.grid', 'rb') as f:
        grid = pickle.load(f)
    robot = DumbRobot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2)
    import robot_configs.policy_iteration as pi
    pi.robot_epoch(robot)

