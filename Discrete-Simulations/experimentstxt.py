import copy
import os
import argparse
import multiprocessing
import itertools
import pickle
import random
from re import A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from environment import Robot, SmartRobot
from policy_iteration import DumbRobot

from robot_configs import *
from robot_configs.policy_iteration_robot import robot_epoch

runs_df = pd.DataFrame()

def run_grid(robot, grid_file, randomness_move, orientation, gamma):
    global runs_df
    
    print("start run_grid")
    print(robot, grid_file, randomness_move, orientation, gamma)
    # robot_epoch = getattr(__import__('robot_configs', fromlist=[robot]), robot_epoch)

    # Cleaned tile percentage at which the room is considered 'clean':
    stopping_criteria = 100

    # Keep track of some statistics:
    efficiencies = []
    n_moves = []
    deaths = 0
    cleaned = []
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    master_robot = DumbRobot(grid, (1, 1), orientation=orientation, p_move=randomness_move, gamma=gamma)

    nr_of_runs = 20
    # Run 100 times:
    for i in range(nr_of_runs):
        # Open the grid file.
        # (You can create one yourself using the provided editor).

        # Calculate the total visitable tiles:
        n_total_tiles = (grid.cells >= 0).sum()
        # Spawn the robot at (1,1) facing north with battery drainage enabled:
        robot = copy.deepcopy(master_robot)
        # Keep track of the number of robot decision epochs:
        n_epochs = 0
        while True:
            n_epochs += 1
            # Do a robot epoch (basically call the robot algorithm once):
                    
            try:
                robot_epoch(robot)
            except ValueError:
                break
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
            #clean_percent = (clean / (dirty + clean)) * 100
            # See if the room can be considered clean, if so, stop the simulaiton instance:

            if clean_percent >= stopping_criteria and goal == 0:
                break
            # Calculate the efficiency score:
            moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
            u_moves = set(moves)
            n_revisted_tiles = len(moves) - len(u_moves)
            efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
            # print("robot clean percent = ", clean_percent, "efficiency = ", efficiency)
            if efficiency < 1:
                print("Efficiency too low. Terminating robot. Robot configs:", str(grid_file) + ";" + str(len(moves)) + ";" + str(randomness_move) + ";" + str(gamma) + ";")
                break
            
        # Keep track of the last statistics for each simulation instance:
        efficiencies.append(float(efficiency))
        n_moves.append(len(robot.history[0]))
        cleaned.append(clean_percent)

    average_efficiencies = sum(efficiencies)/nr_of_runs
    average_n_moves = sum(n_moves)/nr_of_runs
    average_cleaned = sum(cleaned)/nr_of_runs
    print("average_efficiency : " + str(average_efficiencies))
    # # Print the final statistics:
    # fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(8,8)) # two axes on figure

    std_efficiences = np.std(efficiencies)
    std_n_moves = np.std(n_moves)
    std_cleaned = np.std(cleaned)

    result = str(grid_file) + ";" + str(average_efficiencies) + ";" + str(std_efficiences) +  ";" + str(average_n_moves) + ";" + str(std_n_moves) + ";" + str(average_cleaned) + ";" + str(std_cleaned) + ";" + str(randomness_move) + ";" + str(gamma) + "; \n"
    # add all parameters and results to a dataframe
    runs_df = runs_df.append(pd.Series([grid_file, average_efficiencies, std_efficiences, average_n_moves, std_n_moves, average_cleaned, std_cleaned, randomness_move, gamma], index=runs_df.columns), ignore_index=True)
    # save_dir = os.path.join("text")
    # with open("text/results.txt", "a") as f:
    #     f.write(result)


    print("end run_grid")

def run_experiment(robot):
    global runs_df
    random.random()
    with open("text/results.txt", "w") as f:
        header_line = "grid;average_efficiencies;std_efficiencies;average_n_moves;std_n_moves;average_cleaned;std_cleaned;randomness_move;gamma\n"
        f.write(header_line)

    for grid_file in os.listdir('grid_configs'): # grid_file == 'example-2x2-house-0.grid' or grid_file == 'death.grid' or grid_file == 'example-5x5-house-0.grid' or
        if grid_file == 'example-random-level.grid' or grid_file == 'empty.grid' \
                or grid_file == "example-random-house-0.grid" or grid_file == "example-random-house-1.grid" or grid_file == "example-random-house-2.grid" or grid_file == "example-random-house-3.grid" or grid_file == "example-random-house-4.grid":
            continue

        robot = [robot]
        grid_file = [grid_file]

        # evenly distributed list of floats between 0 and 1
        # evenly_floats = np.random.uniform(0.0, 1.0, size=10)
        evenly_floats = np.linspace(0,1,5)
        randomness_move = [0, 0.25, 0.5,  0.75]
        #drain_prob = evenly_floats
        gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # evenly distributed list of integers between 0 and 10
        #drain = [0, 10]
        #vision = [1]

        orientation = ['n']
        #print(robot, grid_file, randomness_move, drain_prob, drain, vision, orientation, gamma)
        print(robot, grid_file, randomness_move, orientation, gamma)
        #args = list(itertools.product(*[robot, grid_file, randomness_move, drain_prob, drain, vision, orientation, gamma]))
        args = list(itertools.product(*[robot, grid_file, randomness_move, orientation, gamma]))
        num_cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_cpus) as processing_pool:
            results = processing_pool.starmap(run_grid, args)
            
        runs_df.to_csv(f"text/{grid_file}_results.csv")
        runs_df = pd.DataFrame()
            
    # save the dataframe to a csv file in the text folder

def main():
    cmdline_parser = argparse.ArgumentParser(description='Script for simulating a competitive sudoku game.')
    cmdline_parser.add_argument('--robot', help="the module name of the first player's SudokuAI class (default: greedy_random_robot)", default='value_teration_robot') #'greedy_random_robot')
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