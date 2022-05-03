import copy

import numpy as np

orients = {'n': -3, 'e': -4, 's': -5, 'w': -6}
dirs = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0)}

# Value Iteration based robot:
# SMALL_ENOUGH is referred to by the mathematical symbol theta in equations
SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def best_action_value(robot, V, s):
    # finds the highest value action (max_a) from state s, returns the action and value
    best_a = None
    best_value = float('-inf')
    #orient according to new state action and move the robot
    robot.move_to_position() #but we are not sure
    # loop through all possible actions to find the best current action
    for a in ALL_POSSIBLE_ACTIONS:
        transititions = robot.grid.get_transition_probs(a) #binomial?
        #get transition probability
        #add reward function
        expected_v = 0
        expected_r = 0
        for (prob, r, state_prime) in transititions:
            expected_r += prob * r
            expected_v += prob * V[state_prime]
        v = expected_r + GAMMA * expected_v
        if v > best_value:
            best_value = v
            best_a = a
    return best_a, best_value

def initialize_random_policy(robot):
    # policy is a lookup table for state -> action
    # we'll randomly choose an action and update as we learn
    policy = {}
    for s in robot.grid.non_terminal_states():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    return policy

def calculate_greedy_policy(robot, V):
    policy = initialize_random_policy()
    # find a policy that leads to optimal value function
    for s in policy.keys():
        #orient and move #TODO
        robot.move_to_position()
        # loop through all possible actions to find the best current action
        best_a, _ = best_action_value(robot, V, s)
        policy[s] = best_a
    return policy

#
# if __name__ == '__main__':
#     # this grid gives you a reward of -0.1 for every non-terminal state
#     # we want to see if this will encourage finding a shorter path to the goal
#     grid = standard_grid(obey_prob=0.5, step_cost=-0.5)
#
#     # print rewards
#     print("rewards:")
#     print_values(grid.rewards, grid)
#
#     # calculate accurate values for each square
#     V = calculate_values(grid)
#
#     # calculate the optimum policy based on our values
#     policy = calculate_greedy_policy(grid, V)
#
#     # our goal here is to verify that we get the same answer as with policy iteration
#     print("values:")
#     print_values(V, grid)
#     print("policy:")
#     print_policy(policy, grid)


def calculate_values(robot):
    #initialize V(s)
    V = {}
    possible_tiles = robot.possible_tiles_after_move()
    # Get rid of any tiles outside a 1 step range (we don't care about our vision in this step):
    states = {move:possible_tiles[move] for move in possible_tiles if abs(move[0]) < 2 and abs(move[1]) < 2}

    print("robot.states:", states)
    for s in states:
        V[s] = 0
    #repeat until convergence
    # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
    while True:
        # biggest_change is referred to by the mathematical symbol delta in equations
        biggest_change = 0
        for s in states:
            old_v = V[s]
            _, new_v = best_action_value(robot, V, s)
            V[s] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - new_v))

        if biggest_change < SMALL_ENOUGH:
            break
    return V


def robot_epoch(robot):
    robot_copy = copy.deepcopy(robot)
    # calculate accurate values for each square
    V = calculate_values(robot_copy)

    # calculate the optimum policy based on our values
    policy = calculate_greedy_policy(robot_copy, V)
    move = policy[robot.pos]
    # Find out how we should orient ourselves:
    new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
    # Orient ourselves towards the dirty tile:
    while new_orient != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        # print('Rotating right once.')
        robot.rotate('r')
    # Move:
    robot.move()
    #handle negative cases