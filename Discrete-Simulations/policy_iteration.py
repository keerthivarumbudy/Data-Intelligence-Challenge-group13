import numpy as np
import random
import math

from utils.utility import *
from utils.get_all_reachable_states import *
from environment import Robot

SMALL_ENOUGH = 1e-3

class DumbRobot(Robot):
    def __init__(self, grid, pos, orientation, p_move=0, battery_drain_p=0, battery_drain_lam=0, vision=1, gamma=0.9):
        # if grid.cells[pos[0], pos[1]] != 1:
        #     raise ValueError
        self.orientation = orientation
        self.pos = pos
        self.grid = grid
        self.orients = {'n': -3, 'e': -4, 's': -5, 'w': -6}
        self.dirs = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0)}
        # self.grid.cells[pos] = self.orients[self.orientation]
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
        from robot_configs.policy_iteration_robot import State
        
        # a state is the grid combined with a robot position
        initial_state = State(self.grid, self.pos, self.orientation)
        
        return initial_state
    
    def get_state_id(self):
        """
        Generate state id from grid and position
        :param grid:[[]]
        
        :return state_id: string
        """
        return get_state_id(self.grid.cells)
        
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
                "grid": [[]],
                "immediately_reachable_states": {
                    state_id: {'direction': [dirs]},
                    state_id: ....
                }
            }, 

            "state_id": ...

            ...

        }
        """
        self.grid.cells = add_location_to_grid(self.grid.cells, self.pos)
        state_dict = generate_reachable_states(self.grid.cells)
        return state_dict
    
    def print_state_dict(self):
        print('REACHABLE STATES')
        for key in self.S.keys():
            print('\n',key)
            try:
                if self.S[key]['is_terminal']:
                    print('TERMINAL')
                    print('reason:',self.S[key]['terminal_reason'])
            except:
                print('NOT TERMINAL')
            print('reachable states:')
            for sub_key in self.S[key]['immediately_reachable_states'].keys():
                print(sub_key, self.S[key]['immediately_reachable_states'][sub_key])
                
    def possible_orients_per_state(self):
        # get all possible orientations that change to a new state for each state
        all_immediate_state_orients = []
        for state_id in list(self.S.keys()):
            immediate_state_orients = list(self.S[state_id]['immediately_reachable_states'].keys())
            immediate_state_orients = [immediate_state_orient for immediate_state_orient in immediate_state_orients 
                                        if self.S[state_id]['immediately_reachable_states'][immediate_state_orient] != state_id]
            all_immediate_state_orients.append(immediate_state_orients)
            
        return all_immediate_state_orients
    
    def init_policy(self):
        # randomly assign policies
        # orientations = [i for i in orients.keys()]
        # return np.random.choice(orientations, len(self.S))
        orientations_per_state = self.possible_orients_per_state()
        random_orientation_per_state = [random.choice(orientations_in_state) if not (len(orientations_in_state) == 0) else '' for orientations_in_state in orientations_per_state]
        return random_orientation_per_state

    def init_values(self):
        # init all values as 0
        return np.zeros(len(self.S))
    
    def calculate_state_value(self, state_id, action, list_S):
        # get ids of immediate reachable states
        immediate_state_ids = self.S[state_id]["immediately_reachable_states"]
        possible_actions = list(immediate_state_ids.keys())
        
        # get reward for each immediate state
        immediate_rewards = [updated_get_reward(self.S[self.S[state_id]["immediately_reachable_states"][orientation]]) for orientation in possible_actions]
        
        # get the value of each immediate state
        immediate_values = [self.values[list_S.index(id)] for id in list(immediate_state_ids.values())]
        
        # aggregate rewards and future values of state
        immediate_aggs = [reward + self.gamma * value for reward, value in zip(immediate_rewards, immediate_values)]
        nr_of_immediate_neighbors = len(immediate_aggs)

        # calculate P(s′,r|s,a)(r+γv(s′)) for all possible actions in state,
        # where intended action has prob (1-p_move) and other 3 moves have prob p_move/nr_of_immediate_neighbors-1
        if nr_of_immediate_neighbors > 1:
            move_probs = [self.p_move/(nr_of_immediate_neighbors-1) if not possible_actions[i] == action else (1-self.p_move) for i in range(nr_of_immediate_neighbors)]
        else:
            move_probs = [1]
        state_future_value = sum([move_prob*immediate_agg for move_prob, immediate_agg in zip(move_probs, immediate_aggs)])
        
        return state_future_value
    
    ### POLICY EVALUATION ###
    
    def calculate_optimal_value(self, state_id):
        # calculate the value of each state
        
        list_S = list(self.S)
        state_ind = list_S.index(state_id)
        
        # get π(a|s)
        optimal_state_policy = self.policy[state_ind]
        
        # get Σ_s'(P(s'|s,a){r + γV(s')})
        state_future_value = self.calculate_state_value(state_id, optimal_state_policy, list_S)
        
        return state_future_value
    
    def sweep_until_convergence(self):
        # while |v′(s)−v(s)| > convergence_threshold
        
        list_S = list(self.S)
        
        biggest_value_change = math.inf
        while biggest_value_change > SMALL_ENOUGH:
            # while biggest change in all states is bigger than SMALL_ENOUGH threshold
            old_values = deepcopy(self.values)
            new_values = deepcopy(old_values)
            biggest_value_change = 0
            for s in self.S.keys():
                temp_state_ind = list_S.index(s)
                
                new_value = self.calculate_optimal_value(s)
                
                biggest_value_change = max(biggest_value_change, abs(old_values[temp_state_ind] - new_value))
                new_values[temp_state_ind] = new_value
                
            self.values = deepcopy(new_values)

    ### POLICY IMPROVEMENT ###

    def update_policy(self):
        # update the policy greedily based on values of accessible states
        
        list_S = list(self.S)
        
        while True:
            self.sweep_until_convergence()
            
            # init new policy π′(a|s)
            old_policy = deepcopy(self.policy)
            new_policy = deepcopy(old_policy)
            for s in self.S.keys():
                # calculate values for all actions in state
                
                list_S = list(self.S)
                state_ind = list_S.index(s)
            
                # get possible actions in state
                possible_actions = list(self.S[s]["immediately_reachable_states"].keys())
                if len(possible_actions) == 0:
                    continue
                
                action_values = np.array([self.calculate_state_value(s, action, list_S) for action in possible_actions])
                best_action = possible_actions[np.argmax(action_values)]
                    
                new_policy[state_ind] = best_action
                
            # set new policy
            self.policy = new_policy
            # if π′ == π, then policy has converged
            if np.array_equal(old_policy, new_policy):
                return self.policy, self.values
            
    ### ROBOT EPOCH ###
    
    def do_move(self):
        # original robot self.move() with -3 at grid.cells[pos] instead of orientation
        
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
        # print(random_move, do_battery_drain, self.battery_lvl)
        if random_move == 1:
            moves = self.possible_tiles_after_move()
            random_move = random.choice([move for move in moves if moves[move] >= 0])
            new_pos = tuple(np.array(self.pos) + random_move)
            # Only move to non-blocked tiles:
            if self.grid.cells[new_pos] >= 0:
                new_orient = list(self.dirs.keys())[list(self.dirs.values()).index(random_move)]
                tile_after_move = self.grid.cells[new_pos]
                self.grid.cells[self.pos] = 0
                self.grid.cells[new_pos] = -100
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
            state_ind = list(self.S).index(self.get_state_id())
            # new_pos = (int(self.pos[0]) + int(self.policy[state_ind][0]), int(self.pos[1]) + int(self.policy[state_ind][1]))
            # new_pos = tuple(map(sum, zip(self.pos, self.policy[state_ind])))
            # print(self.policy[state_ind], self.pos)
            # print(self.get_state_id())
            if not self.policy[state_ind] == '':
                new_pos = tuple(np.array(self.pos) + self.dirs[self.policy[state_ind]])
            else:
                new_pos = self.pos
            # print(self.policy[state_ind], new_pos)
            # print('value of next pos:', self.grid.cells[new_pos[1]][new_pos[0]], 'bool:', self.grid.cells[new_pos[1]][new_pos[0]] >= 0)
            # Only move to non-blocked tiles:
            if self.grid.cells[new_pos[1]][new_pos[0]] >= 0 and not self.policy[state_ind] == '':
                tile_after_move = self.grid.cells[new_pos[1]][new_pos[0]]
                self.grid.cells[self.pos[1]][self.pos[0]] = 0
                self.grid.cells[new_pos[1]][new_pos[0]] = -100
                self.pos = new_pos
                # print(self.pos)
                self.history[0].append(self.pos[0])
                self.history[1].append(self.pos[1])
                # Death:
                if tile_after_move == 3:
                    self.alive = False
                    return False
                return True
            else:
                return False
    
    def find_and_do_move(self):
        # set self.policy to self.orientation at self.pos
        try:
            init_state_ind = list(self.S).index(self.get_state_id())
        except:
            return
        # self.policy[init_state_ind] = self.orientation
        
        # update to optimal values and policies
        self.update_policy()

        self.do_move()
        # attempt_complete = False
        # while not attempt_complete:
            # try:
            #     self.do_move()
            #     attempt_complete = True
            # except ValueError:
            #     # print("ERROR: Value error occured when moving. current position:", self.pos, "target action:", self.policy[init_state_ind], "grid:", self.grid.cells)
            #     attempt_complete = False
    
if __name__ == '__main__':
    import pickle
    grid_file = 'death.grid'
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
        
    starting_location = (1,1)
    
    # print(grid.cells)
    robot = DumbRobot(grid, starting_location, orientation='n', battery_drain_p=0.5, battery_drain_lam=2, p_move=0.2)
    robot.print_state_dict()
    
    # print('nr of reachable states:', len(robot.S))
    # print('old policy:', robot.policy)
    # print('old values:', robot.values)
    
    # state_ind = list(robot.S).index(get_state_id(robot.grid.cells))
    # _, robot.values[state_ind] = robot.calculate_values(get_state_id(robot.grid.cells))
    # # print('new values:', robot.values)
    # # print(robot.update_policy())
    # # print('new policy:', robot.policy)
    # _, robot.values[state_ind] = robot.calculate_values(get_state_id(robot.grid.cells))
    # # print('new values:', robot.values)
    
    # # print(robot.calculate_values(get_state_id(robot.grid.cells)))
    # print(robot.update_policy())