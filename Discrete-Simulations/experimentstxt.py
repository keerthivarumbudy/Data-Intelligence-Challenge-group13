import os
import argparse
import multiprocessing
import itertools
import pickle
import random
from re import A
import matplotlib.pyplot as plt
import numpy as np

from environment import Robot

from robot_configs import *
from robot_configs.greedy_random_robot import robot_epoch

def run_grid(robot, grid_file, randomness_move, drain_prob, drain, vision, orientation):
    print(robot, grid_file, randomness_move, drain_prob, drain, vision, orientation)
    # robot_epoch = getattr(__import__('robot_configs', fromlist=[robot]), robot_epoch)

    # Cleaned tile percentage at which the room is considered 'clean':
    stopping_criteria = 100
        
    # Keep track of some statistics:
    efficiencies = []
    n_moves = []
    deaths = 0
    cleaned = []

    # Run 100 times:
    for i in range(100):
        # Open the grid file.
        # (You can create one yourself using the provided editor).
        with open(f'grid_configs/{grid_file}', 'rb') as f:
            grid = pickle.load(f)
        # Calculate the total visitable tiles:
        n_total_tiles = (grid.cells >= 0).sum()
        # Spawn the robot at (1,1) facing north with battery drainage enabled:
        robot = Robot(grid, (1, 1), orientation=orientation, battery_drain_p=drain_prob, battery_drain_lam=drain, vision=vision, p_move=randomness_move)
        # Keep track of the number of robot decision epochs:
        n_epochs = 0
        while True:
            n_epochs += 1
            # Do a robot epoch (basically call the robot algorithm once):
            robot_epoch(robot)
            # Stop this simulation instance if robot died :( :
            if not robot.alive:
                deaths += 1
                break
            # Calculate some statistics:
            clean = (grid.cells == 0).sum()
            dirty = (grid.cells >= 1).sum()
            goal = (grid.cells == 2).sum()
            # Calculate the cleaned percentage:
            clean_percent = (clean / (dirty + clean)) * 100
            # See if the room can be considered clean, if so, stop the simulaiton instance:
            if clean_percent >= stopping_criteria and goal == 0:
                break
            # Calculate the effiency score:
            moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
            u_moves = set(moves)
            n_revisted_tiles = len(moves) - len(u_moves)
            efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
        # Keep track of the last statistics for each simulation instance:
        efficiencies.append(float(efficiency))
        n_moves.append(len(robot.history[0]))
        cleaned.append(clean_percent)

    average_efficiencies = sum(efficiencies)/100
    average_n_moves = sum(n_moves)/100
    average_cleaned = sum(cleaned)/100
    print("average_efficiency : " + str(average_efficiencies))
    # # Print the final statistics:
    # fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(8,8)) # two axes on figure

    # # Make some plots:
    # ax1.hist(cleaned)
    # ax1.set_title('Percentage of tiles cleaned.')
    # ax1.set_xlabel('% cleaned')
    # ax1.set_ylabel('count')

    # ax2.hist(efficiencies)
    # ax2.set_title('Efficiency of robot.')
    # ax2.set_xlabel('Efficiency %')
    # ax2.set_ylabel('count')
    
    # plt.subplots_adjust(hspace=0.4,
    #                     wspace=0.4)
    # # plt savefig to unique dir in plots
    # save_dir = os.path.join('plots', f'{grid_file}', f'{orientation}')
    # print(save_dir)
    # print(os.path.join(save_dir, f'{randomness_move}_{drain_prob}_{drain}_{vision}_plot.png'))
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # fig1.savefig(os.path.join(save_dir, f'{randomness_move}_{drain_prob}_{drain}_{vision}_plot.png'))

    result = "average_efficiency : " + str(average_efficiencies) +  ", average_n_moves : " + str(average_n_moves) + ", average_cleaned : " + str(average_cleaned)
    save_dir = os.path.join('text', f'{grid_file}', f'{orientation}')
    print(os.path.join(save_dir, f'{randomness_move}_{drain_prob}_{drain}_{vision}_text.txt'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        file1 = open(os.path.join(save_dir, f'{randomness_move}_{drain_prob}_{drain}_{vision}_text.txt'), "w").write(result)
        file1.close()

    # .savefig(os.path.join(save_dir, f'{randomness_move}_{drain_prob}_{drain}_{vision}_text.txt'))
def run_experiment(robot):
    random.random()

    for grid_file in os.listdir('grid_configs'):
        if grid_file == 'empty.grid' or grid_file == 'death.grid' or grid_file == "example-random-house-0.grid" or grid_file == "example-random-house-1.grid" or grid_file == "example-random-house-2.grid" or grid_file == "example-random-house-3.grid" or grid_file == "example-random-house-4.grid":
            continue
        
        robot = [robot]
        grid_file = [grid_file]
        
        # evenly distributed list of floats between 0 and 1
        # evenly_floats = np.random.uniform(0.0, 1.0, size=10)
        evenly_floats = [0, 1]
        randomness_move = evenly_floats
        drain_prob = evenly_floats
        # evenly distributed list of integers between 0 and 10
        drain = [0, 10]
        vision = [1, 5]

        orientation = ['n', 'e', 's', 'w']
        print(robot, grid_file, randomness_move, drain_prob, drain, vision, orientation)

        args = list(itertools.product(*[robot, grid_file, randomness_move, drain_prob, drain, vision, orientation]))

        num_cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_cpus) as processing_pool:
            results = processing_pool.starmap(run_grid, args)
        
def main():
    cmdline_parser = argparse.ArgumentParser(description='Script for simulating a competitive sudoku game.')
    cmdline_parser.add_argument('--robot', help="the module name of the first player's SudokuAI class (default: greedy_random_robot)", default='greedy_random_robot')
    args = cmdline_parser.parse_args()
    
    run_experiment(args.robot)
    
    

if __name__ == '__main__':
    robot = 'greedy_random_robot'
    grid = 'example-random-house-0.grid'
    randomness_move = 0.04716257948841984
    drain_prob = 0.6800854602934419 
    drain = 1
    vision = 0
    orientation = 'n'
    main()