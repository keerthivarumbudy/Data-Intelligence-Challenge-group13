import copy
from itertools import combinations, chain
import numpy as np



# SMALL_ENOUGH is referred to by the mathematical symbol theta in equations

orients = {'n': -3, 'e': -4, 's': -5, 'w': -6}
dirs = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0)}
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')



class State:
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

    def move(self):
        # Can't move if we're dead now, can we?
        if not self.alive:
            return False
        new_pos = tuple(np.array(self.pos) + dirs[self.orientation])
        # Only move to non-blocked tiles:
        if self.grid.cells[new_pos] >= 0:
            tile_after_move = self.grid.cells[new_pos]
            self.grid.cells[self.pos] = 0
            self.grid.cells[new_pos] = orients[self.orientation]
            self.pos = new_pos
            # Death:
            if tile_after_move == 3:
                self.alive = False
                return False
            return True
        else:
            return False

    def rotate(self, dir):
        current = list(orients.keys()).index(self.orientation)
        if dir == 'r':
            self.orientation = list(orients.keys())[(current + 1) % 4]
        elif dir == 'l':
            self.orientation = list(orients.keys())[current - 1]
        self.grid.cells[self.pos] = orients[self.orientation]



    def get_possible_moves(self):
        data = {}
        moves = list(dirs.values())
        i = 1
        for move in moves:
            to_check = tuple(np.array(self.pos) + (np.array(move) * (i + 1)))
            if to_check[0] < self.grid.cells.shape[0] and to_check[1] < self.grid.cells.shape[1] and to_check[
                0] >= 0 and to_check[1] >= 0:
                data[tuple(np.array(move) * (i + 1))] = self.grid.cells[to_check]
                # Show death tiles as dirty:
                if data[tuple(np.array(move) * (i + 1))] == 3:
                    data[tuple(np.array(move) * (i + 1))] = 1
        return data

    def get_reward(self):
        reward_dict = {
            -2: -2,
            -1: -2,
            0: -1,
            1: 1,
            2: 10,
            3: -10
        }
        state_reward = reward_dict[self.grid.cells[self.pos]]

        # modified from environment.py
        # TODO: correct?
        expected_drain = self.battery_drain_p * np.random.exponential(self.battery_drain_lam)

        # reward is reward of moving to new state + expected battery drain (negative constant)
        reward = state_reward - expected_drain

        return reward

    def get_neighbouring_states(self): # Checks neighboring state is not a wall/obstacle
        moves = list(self.dirs.values())
        states = {}
        for move in moves:
            new_move = tuple(np.array(self.pos) + np.array(move))
            if new_move[0] < self.grid.cells.shape[0] and new_move[1] < self.grid.cells.shape[1] and \
                    new_move[0] >= 0 and new_move[1] >= 0 and self.grid.cells[new_move] >=0:
                new_pos = tuple(np.array(self.pos) + dirs[self.orientation])
                new_state = copy.deepcopy(self)
                # Find out how we should orient ourselves:
                new_orient = list(dirs.keys())[list(dirs.values()).index(self.grid.cells[new_move])]
                # Orient ourselves towards the dirty tile:
                while new_orient != new_state.orientation:
                    # If we don't have the wanted orientation, rotate clockwise until we do:
                    # print('Rotating right once.')
                    new_state.rotate('r')

                # Only move to non-blocked tiles:
                if new_state.grid.cells[new_pos] >= 0:
                    tile_after_move = new_state.grid.cells[new_pos]
                    new_state.grid.cells[new_state.pos] = 0
                    new_state.grid.cells[new_pos] = new_state.orients[new_state.orientation]
                    new_state.pos = new_pos
                    states.add(new_state)
                    if tile_after_move == 3:
                        self.alive = False
        return states

    def get_action_reward(self, action):
        state_primes = self.get_neighbouring_states() # Excludes obstacles and walls
        transitions = []
        new_pos = tuple(np.array(self.pos) + self.dirs[action])

        # Getting the transition probabilities
        for state_prime in state_primes:
            if state_prime.pos == new_pos:
                prob = 1 - state_prime.p_move
            elif state_prime.p_move == 0:
                prob = 0
            else:
                prob = state_prime.p_move/(len(state_primes)-1)
            transitions.append((prob, state_prime.get_reward(), state_prime))
        return transitions




def is_terminal(state):
    # TODO: consider unreachable situations
    g = state.grid.cells
    return not np.where(np.isin(g, [1, 2])).any()


def best_action_value(V, s, gamma):
    # finds the highest value action (max_a) from state s, returns the action and value
    best_a = None
    best_value = float('-inf')

    # loop through all possible actions to find the best current action
    for a in s.dirs.keys():
        state_primes = s.get_action_reward(a)
        sum_a = 0
        for (prob, r, state_prime) in state_primes:
            if not (state_prime.grid.cells, state_prime.pos) in V:
                V[(str(state_prime.grid.cells), state_prime.pos)] = 0

            sum_a += prob * (r + (gamma * V[(str(state_prime.grid.cells), state_prime.pos)]))

        v = sum_a
        if v > best_value:
            best_value = v
            best_a = a
    return best_a, best_value


def evaluate_state(state, V, gamma):

    if is_terminal(state):
        return V

    best_a, best_val = best_action_value(V, state, gamma)

    V[(str(state.grid.cells), state.pos)] = best_val

    for new_state in state.get_neighbouring_states():
        if not (str(new_state.grid.cells), new_state.pos) in V:
            V = evaluate_state(new_state, V, gamma)

    return V

def powerset(in_iter):
    "powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    in_iter = list(in_iter)
    return chain.from_iterable(combinations(in_iter, r) for r in range(len(in_iter) + 1))


def get_possible_states(s):

    start_grid = s.grid.cells

    travel_ind = np.where(np.isin(start_grid, [1, 2])) # Indices our robot could theoretically travel to
    travel_ind_tup = list(zip(travel_ind[0], travel_ind[1]))
    print("before powerset")
    print(travel_ind_tup)
    poss_comb_ind = powerset(travel_ind_tup)
    print(list(poss_comb_ind))




    print(np.where(travel_ind))
    states = set()



    # data = self.get_possible_moves()
    # for move in data:
    #     new_pos = tuple(np.array(self.pos) + dirs[self.orientation])
    #     new_state = copy.deepcopy(self)
    #     # Find out how we should orient ourselves:
    #     new_orient = list(dirs.keys())[list(dirs.values()).index(move)]
    #     # Orient ourselves towards the dirty tile:
    #     while new_orient != new_state.orientation:
    #         # If we don't have the wanted orientation, rotate clockwise until we do:
    #         # print('Rotating right once.')
    #         new_state.rotate('r')
    #
    #     # Only move to non-blocked tiles:
    #     if new_state.grid.cells[new_pos] >= 0:
    #         tile_after_move = new_state.grid.cells[new_pos]
    #         new_state.grid.cells[new_state.pos] = 0
    #         new_state.grid.cells[new_pos] = new_state.orients[new_state.orientation]
    #         new_state.pos = new_pos
    #         states.add(new_state)
    return states



# Value Iteration based robot:
def robot_epoch(robot):
    move = robot.policy[robot.pos]
    # Find out how we should orient ourselves:
    new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
    # Orient ourselves towards the dirty tile:
    while new_orient != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        # print('Rotating right once.')
        robot.rotate('r')
    # Move:
    robot.move()
