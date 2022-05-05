import numpy as np
import random

from utils.utility import *

SMALL_ENOUGH = 1e-3
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


class Robot:
    def __init__(self, grid, pos, orientation, p_move=0, battery_drain_p=0, battery_drain_lam=0, vision=1):
        if grid.cells[pos[0], pos[1]] != 1:
            raise ValueError
        self.orientation = orientation
        self.pos = pos
        self.grid = grid
        self.orients = {'n': -3, 'e': -4, 's': -5, 'w': -6}
        self.dirs = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0)}
        self.grid.cells[pos] = self.orients[self.orientation]
        self.history = [[], []]
        self.p_move = p_move
        self.battery_drain_p = battery_drain_p
        self.battery_drain_lam = battery_drain_lam
        self.battery_lvl = 100
        self.alive = True
        self.vision = vision

    def possible_tiles_after_move(self):
        moves = list(self.dirs.values())
        # Fool the robot and show a death tile as normal (dirty)
        data = {}
        for i in range(self.vision):
            for move in moves:
                to_check = tuple(np.array(self.pos) + (np.array(move) * (i + 1)))
                if to_check[0] < self.grid.cells.shape[0] and to_check[1] < self.grid.cells.shape[1] and to_check[
                    0] >= 0 and to_check[1] >= 0:
                    data[tuple(np.array(move) * (i + 1))] = self.grid.cells[to_check]
                    # Show death tiles as dirty:
                    if data[tuple(np.array(move) * (i + 1))] == 3:
                        data[tuple(np.array(move) * (i + 1))] = 1
        return data

    def move(self):
        # Can't move if we're dead now, can we?
        if not self.alive:
            return False
        random_move = np.random.binomial(1, self.p_move)
        do_battery_drain = np.random.binomial(1, self.battery_drain_p)
        if do_battery_drain == 1 and self.battery_lvl > 0:
            self.battery_lvl -= np.random.exponential(self.battery_drain_lam)
        # Handle empty battery:
        if self.battery_lvl <= 0:
            self.alive = False
            return False
        if random_move == 1:
            moves = self.possible_tiles_after_move()
            random_move = random.choice([move for move in moves if moves[move] >= 0])
            new_pos = tuple(np.array(self.pos) + random_move)
            # Only move to non-blocked tiles:
            if self.grid.cells[new_pos] >= 0:
                new_orient = list(self.dirs.keys())[list(self.dirs.values()).index(random_move)]
                tile_after_move = self.grid.cells[new_pos]
                self.grid.cells[self.pos] = 0
                self.grid.cells[new_pos] = self.orients[new_orient]
                self.pos = new_pos
                self.history[0].append(self.pos[0])
                self.history[1].append(self.pos[1])
                if tile_after_move == 3:
                    self.alive = False
                    return False
                return True
            else:
                return False
        else:
            new_pos = tuple(np.array(self.pos) + self.dirs[self.orientation])
            # Only move to non-blocked tiles:
            if self.grid.cells[new_pos] >= 0:
                tile_after_move = self.grid.cells[new_pos]
                self.grid.cells[self.pos] = 0
                self.grid.cells[new_pos] = self.orients[self.orientation]
                self.pos = new_pos
                self.history[0].append(self.pos[0])
                self.history[1].append(self.pos[1])
                # Death:
                if tile_after_move == 3:
                    self.alive = False
                    return False
                return True
            else:
                return False

    def move_to_position(self):
        # Can't move if we're dead now, can we?
        if not self.alive:
            return False
        new_pos = tuple(np.array(self.pos) + self.dirs[self.orientation])
        # Only move to non-blocked tiles:
        if self.grid.cells[new_pos] >= 0:
            tile_after_move = self.grid.cells[new_pos]
            self.grid.cells[self.pos] = 0
            self.grid.cells[new_pos] = self.orients[self.orientation]
            self.pos = new_pos
            self.history[0].append(self.pos[0])
            self.history[1].append(self.pos[1])
            # Death:
            if tile_after_move == 3:
                self.alive = False
                return False
            return True
        else:
            return False

    def rotate(self, dir):
        current = list(self.orients.keys()).index(self.orientation)
        if dir == 'r':
            self.orientation = list(self.orients.keys())[(current + 1) % 4]
        elif dir == 'l':
            self.orientation = list(self.orients.keys())[current - 1]
        self.grid.cells[self.pos] = self.orients[self.orientation]


class Grid:
    def __init__(self, n_cols, n_rows):
        self.n_rows = n_rows
        self.n_cols = n_cols
        # Building the boundary of the grid:
        self.cells = np.ones((n_cols, n_rows))
        print(self.cells)
        self.cells[0, :] = self.cells[-1, :] = -1
        print(self.cells)
        self.cells[:, 0] = self.cells[:, -1] = -1
        print(self.cells)

    def put_obstacle(self, x0, x1, y0, y1, from_edge=1):
        self.cells[max(x0, from_edge):min(x1 + 1, self.n_cols - from_edge),
        max(y0, from_edge):min(y1 + 1, self.n_rows - from_edge)] = -2

    def put_singular_obstacle(self, x, y):
        self.cells[x][y] = -2

    def put_singular_goal(self, x, y):
        self.cells[x][y] = 2

    def put_singular_death(self, x, y):
        self.cells[x][y] = 3


def generate_grid(n_cols, n_rows):
    # Placeholder function used to generate a grid.
    # Select an empty grid file in the user interface and add code her to automatically fill it.
    # Look at grid_generator.py for inspiration.
    grid = Grid(n_cols, n_rows)
    return grid

def best_action_value(V, s, gamma):
    # finds the highest value action (max_a) from state s, returns the action and value
    best_a = None
    best_value = float('-inf')

    # loop through all possible actions to find the best current action
    for a in ALL_POSSIBLE_ACTIONS:
        transititions = s.get_transition_probs(a)
        sum = 0
        for (prob, r, state_prime) in transititions:
            sum += prob * (r + (gamma * V[state_prime]))
        v = sum
        if v > best_value:
            best_value = v
            best_a = a
    return best_a, best_value


class SmartRobot(Robot):
    def __init__(self, grid, pos, orientation, p_move=0, battery_drain_p=0, battery_drain_lam=0, vision=1, gamma=0.9):
        if grid.cells[pos[0], pos[1]] != 1:
            raise ValueError
        self.orientation = orientation
        self.pos = pos
        self.grid = grid
        self.orients = {'n': -3, 'e': -4, 's': -5, 'w': -6}
        self.dirs = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0)}
        self.grid.cells[pos] = self.orients[self.orientation]
        self.history = [[], []]
        self.p_move = p_move
        self.battery_drain_p = battery_drain_p
        self.battery_drain_lam = battery_drain_lam
        self.battery_lvl = 100
        self.alive = True
        self.vision = vision
        self.gamma = gamma
        self.V = self.calculate_values()
        self.policy = self.calculate_policy()


    def calculate_values(self):
        from robot_configs.value_iteration_robot import State
        current_state = State(self.grid, self.pos, self.orientation, self.p_move, self.battery_drain_p,
                              self.battery_drain_lam)

        #initialize V(s)
        V = {}
        possible_states = current_state.get_possible_states()
        print("robot.states:", possible_states)
        for s in possible_states:
            V[s] = 0
        #repeat until convergence
        # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
        while True:
            # biggest_change is referred to by the mathematical symbol delta in equations
            biggest_change = 0
            for s in possible_states:
                old_v = V[s]
                _, new_v = best_action_value(V, s, self.gamma)
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - new_v))

            if biggest_change < SMALL_ENOUGH:
                break
        return V

    def initialize_random_policy(self):
        # policy is a lookup table for state -> action
        # we'll randomly choose an action and update as we learn
        policy = {}
        for s in self.grid.non_terminal_states():
            policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
        return policy

    def calculate_policy(self):
        from robot_configs.value_iteration_robot import State
        current_state = State(self.grid, self.pos, self.orientation, self.p_move, self.battery_drain_p,
                              self.battery_drain_lam)
        possible_states = current_state.get_possible_states()
        policy = {}
        # find a policy that leads to optimal value function
        for s in possible_states:
            # loop through all possible actions to find the best current action
            best_a, _ = best_action_value(self.V, s)
            policy[(s.grid.cells, s.pos)] = best_a
        return policy

#{(grid, pos): best_a, }


class DumbRobot(Robot):
    def __init__(self, grid, pos, orientation, p_move=0, battery_drain_p=0, battery_drain_lam=0, vision=1, gamma=0.9):
        if grid.cells[pos[0], pos[1]] != 1:
            raise ValueError
        self.orientation = orientation
        self.pos = pos
        self.grid = grid
        self.orients = {'n': -3, 'e': -4, 's': -5, 'w': -6}
        self.dirs = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0)}
        self.grid.cells[pos] = self.orients[self.orientation]
        self.history = [[], []]
        self.p_move = p_move
        self.battery_drain_p = battery_drain_p
        self.battery_drain_lam = battery_drain_lam
        self.battery_lvl = 100
        self.alive = True
        self.vision = vision
        self.gamma = gamma
        
        self.state = self.init_state()
        self.S = self.generate_reachable_states()
        
        self.policy = self.init_policy()
        self.values = self.init_values()
        
    def init_state(self):
        from robot_configs.policy_iteration import State
        
        # a state is the grid combined with a robot position
        initial_state = State(self.grid, self.pos, self.orientation)
        
        return initial_state
    
    def get_state_id(self):
        """
        Generate state id from grid and position
        :param grid:[[]]
        :param pos: (int, int)
        
        :return state_id: string
        """
        pass
        
    def generate_reachable_states(self):
        """
        Generate all reachable states in dict, where every
        entry has loc, grid and immediate_reachable_states
        
        :param state: {
            loc: (index_x, index_y),
            grid: [[]]
        }

        :return:
        {
            "state_id": {
                "loc": (index_x, index_y), 
                "grid": [[]],
                "immediately_reachable_states": {
                    "N": state_id,
                    "S": ....
                }
            }, 

            "state_id": ...

            ...

        }
        """
        from collections import OrderedDict
        new_dict = OrderedDict({
            "0": {
                "loc": (1,1), 
                "grid": self.grid,
                "immediately_reachable_states": {
                    "n": "1",
                    "e": "2",
                    "s": "3",
                    "w": "4"
                }
            },
            "1": {
                "loc": (1, 0), 
                "grid": self.grid,
                "immediately_reachable_states": {
                    "n": "1",
                    "s": "1"
                }
            },
            "2": {
                "loc": (2, 1), 
                "grid": self.grid,
                "immediately_reachable_states": {
                    "n": "1",
                    "s": "1"
                }
            },
            "3": {
                "loc": (1, 2), 
                "grid": [[]],
                "immediately_reachable_states": {
                    "n": "1",
                    "s": "1"
                }
            },
            "4": {
                "loc": (0, 1), 
                "grid": self.grid,
                "immediately_reachable_states": {
                    "n": "1",
                    "s": "1"
                }
            }
        })
        return new_dict
    
    def init_policy(self):
        # randomly assign policies
        orientations = [i for i in orients.keys()]
        return np.random.choice(orientations, len(self.S))

    def init_values(self):
        # init all values as 0
        return np.zeros(len(self.S))
    
    def stochastic_final_reward(self, action, immediate_aggs):
        # calculate P(s′,r|s,a)(r+γv(s′)) for all possible actions in state,
        # where intended action has prob (1-p_move) and other 3 moves have prob p_move/3
        immediate_final_rewards = [(self.p_move/3) * immediate_aggs[i] if not list(orients.keys())[i] == action else (1-self.p_move) * immediate_aggs[i] for i in range(len(immediate_aggs))]
        return sum(immediate_final_rewards)

    # Policy Evaluation

    def calculate_values(self, state_id):
        # calculate the value of each state
        
        state_ind = list(self.S).index(state_id)
        
        # get π(a|s)
        optimal_state_policy = self.policy[state_ind]
        
        # get ids of immediate reachable states
        immediate_state_ids = self.S[state_id]["immediately_reachable_states"]
        # get index of immediate_state_ids in self.S
        immediate_state_ids_index = [list(self.S).index(i) for i in immediate_state_ids.values()]
        
        # get v(s')
        state_value = self.values[immediate_state_ids_index[list(immediate_state_ids.keys()).index(optimal_state_policy)]]

        # get reward for each immediate state
        immediate_rewards = [get_reward_dict(self.S[state_id], orientation) for orientation in orients.keys()]
        
        # get the value of each immediate state
        immediate_values = [self.calculate_values(id) for id in list(immediate_state_ids.values())]
        
        # aggregate rewards and future values of state
        immediate_aggs = [reward + self.gamma * value for reward, value in zip(immediate_rewards, immediate_values)]
        
        immediate_final_rewards = self.stochastic_final_reward(optimal_state_policy, immediate_aggs)
        
        # sum up to get final future value for state
        state_future_value = sum(immediate_final_rewards)
        
        return state_value, state_future_value
            
    def sweep_until_convergence(self, convergence_threshold=0.01):
        # while |v′(s)−v(s)| < convergence_threshold
        
        old_values = self.values
        new_values = old_values
        # TODO: is s current state in for loop or is it initial state?
        while abs(new_values[self.state.pos] - old_values[self.state.pos]) < convergence_threshold:
            old_values = new_values
            
            for s in range(self.S.keys()):
                state_ind = self.S.index(s)
                old_values[state_ind], new_values[state_ind] = self.calculate_values(s)
        
        self.values = new_values

    # Policy Improvement

    def update_policy(self):
        # update the policy greedily based on values of accessible states
        
        # init new policy π′(a|s)
        old_policy = self.policy
        new_policy = old_policy
        for s in range(self.S.keys()):
            state_ind = self.S.index(s)
            self.values[state_ind]
            
            # get ids of immediate reachable states
            immediate_state_ids = self.S[s]["immediately_reachable_states"]
            # get index of immediate_state_ids in self.S
            immediate_state_ids_index = [self.S.index(i) for i in immediate_state_ids.values()]

            # get reward for each immediate state
            immediate_rewards = [get_reward(self.S[s], orientation) for orientation in orients.keys()]
            
            # get the value of each immediate state through already updated self.values
            immediate_values = [self.values[i] for i in immediate_state_ids_index]
            
            # aggregate rewards and future values of state
            immediate_aggs = [reward + self.gamma * value for reward, value in zip(immediate_rewards, immediate_values)]
            
            # calculate final rewards for all orientations
            possible_actions = orients.keys()
            possible_actions_final_rewards = [self.stochastic_final_reward(action, immediate_aggs) for action in possible_actions]
            
            # find max a
            max_action_ind = max(range(len(possible_actions_final_rewards)), key=possible_actions_final_rewards.__getitem__)
            
            new_policy[state_ind] = possible_actions[max_action_ind]
            
        # set new policy
        self.policy = new_policy
        # if π′ == π, then policy has converged
        if np.array_equal(old_policy, new_policy):
            return True
        # otherwise, rerun policy evaluation with new policy
        return False
    
if __name__ == '__main__':
    import pickle
    with open(f'grid_configs/simple-random-house-0.grid', 'rb') as f:
        grid = pickle.load(f)
        
    from utils.get_all_reachable_states import *
    import pprint
    
    starting_location = (1, 1)
    grid.cells = add_location_to_grid(grid.cells, starting_location)
    print(grid.cells)
    state_dict = generate_reachable_states(grid.cells)
    print('REACHABLE STATES')
    for key in state_dict.keys():
        print('\n',key)
        print('reachable states:')
        for sub_key in state_dict[key]['immediately_reachable_states'].keys():
            print(sub_key, state_dict[key]['immediately_reachable_states'][sub_key])
    robot = DumbRobot(grid, starting_location, orientation='n', battery_drain_p=0.5, battery_drain_lam=2)
    
    # get reward for each immediate state
    immediate_rewards = [1,1,1,1]
    
    # get the value of each immediate state
    immediate_values = [4,8,6,3]
    
    # aggregate rewards and future values of state
    immediate_aggs = [reward + robot.gamma * value for reward, value in zip(immediate_rewards, immediate_values)]
        
    # print(robot.stochastic_final_reward('e', immediate_aggs))
    
    # print(robot.policy)
    # print(robot.values)
    
    # print(robot.calculate_values("0"))