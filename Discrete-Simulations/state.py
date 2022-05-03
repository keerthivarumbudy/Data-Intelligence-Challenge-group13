import numpy as np


class State:
    def __init__(self, grid, pos, orientation, p_move=0, battery_drain_p=0, battery_drain_lam=0):
        if grid.cells[pos[0], pos[1]] != 1:
            raise ValueError
        self.orientation = orientation
        self.pos = pos
        self.grid = grid
        self.grid.cells[pos] = self.orients[self.orientation]
        self.p_move = p_move
        self.alive = True
        self.battery_drain_p = battery_drain_p
        self.battery_drain_lam = battery_drain_lam

    def move(self):
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