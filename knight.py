#! /usr/bin/env python
import numpy as np
import random
from time import sleep
from curses import wrapper



def two_dim(pos, width):
    '''
    Return 2d co-ordinates represented as 1d array.
    pos: position in one dimension
    width: total width of 2d array(columns)
    '''
    width += 1
    return pos//width, pos%width

def possible_choices(array):
    '''
    Return possible choices(within bounds) of directions from given state.
    array: 1x4 array of a single state's directions
    '''
    return [i for i in np.where(~np.isnan(array))[0]]

def next_state(state, direction, colsize):
    '''
    Returns next state given current state and direction
    state: current state(position)
    direction: direction of next move
    colsize: width of the board
    '''
    if direction == 0:
        next = state - (colsize + 1)
    elif direction == 1:
        next = state + 1
    elif direction == 2:
        next = state + colsize + 1
    else:
        next = state - 1
    return next


class Board:
    '''
    Stores board layout and rewards table.

    tiles: np.array, square matrix of tiles represented by chars
    costs: dict of costs of each tile type
    '''
    def __init__(self, tiles, costs):
        self.tiles = tiles
        # Build rewards table by taking costs
        self.reward = np.array(
            list([costs[val] for val in l] for l in self.tiles),
            dtype=np.int16)
        self.player_pos = (0, 0)

    def __str__(self):
        # Temporarily changing tile at player's position.
        # This can probably be done in a better way
        tile_under_player = self.tiles[self.player_pos[0], self.player_pos[1]]
        self.tiles[self.player_pos[0], self.player_pos[1]] = 'K'

        table_string = '\n'
        for r in self.tiles:
            table_string += ''.join(r) + '\n'

        self.tiles[self.player_pos[0], self.player_pos[1]] = tile_under_player

        return table_string


class Qtable:
    '''
    Stores Q values, methods to train Q-table using
    the Bellman-Ford algorithm.
    '''
    def __init__(self, board, cur_state=0):
        self.board = board
        self.cur_state = cur_state
        self.direction = 0
        self.acc_cost = 0
        # width of board (not qtable)
        self.colsize = board.tiles.shape[1] - 1
        self.table = np.zeros((board.tiles.size, 4))

        # setting illegal(outside bounds) choices to NaN
        # top edge
        self.table[:self.colsize+1, 0] = np.nan
        # right edge
        self.table[self.colsize::self.colsize+1, 1] = np.nan
        # bottom edge
        self.table[-(self.colsize+1):, 2] = np.nan
        # left edge
        self.table[::self.colsize+1, 3] = np.nan

    def calculate_q(self):
        '''
        Calculate Q value using the Bellman Ford algorithm
        '''
        self.table[self.cur_state, self.direction] += \
            self.learn_rate * (
                self.board.reward[
                    two_dim(self.cur_state, self.colsize)[0],
                    two_dim(self.cur_state, self.colsize)[1]] +
                self.discount *
                np.nanmax(self.table[self.next_state]) -
                self.table[self.cur_state, self.direction])

    def train(
            self,
            iterations,
            learn_rate=0.1,
            discount=0.9,
            epsilon=1.0,
            decay=True):
        '''
        Train the table.
        iterations: number of iterations
        learn_rate: the learning rate
        discount: the delta discount
        epsilon: starting value of epsilon parameter
        decay: boolean value indicating whether to enable epsilon 
            decay.(True by default)
        '''
        self.learn_rate = learn_rate
        self.discount = discount
        if decay:
            epsilon_decay = epsilon / iterations
        else:
            epsilon_decay = 0

        for i in range(iterations):
            if random.random() > epsilon:
                # exploit
                self.direction = np.nanargmax(self.table[self.cur_state])

            else:
                # explore
                self.direction = int(random.choice(
                    possible_choices(self.table[self.cur_state])))

            self.acc_cost += self.table[self.cur_state, self.direction]
            self.next_state = next_state(
                self.cur_state,
                self.direction,
                self.colsize)
            self.calculate_q()
            self.print_info(i)

            self.cur_state = self.next_state
            epsilon -= epsilon_decay

            # Reset to original state if game finishes(win or lose)
            if self.acc_cost < -100 or self.acc_cost > 90:
                self.cur_state = 0
                self.acc_cost = 0

    def print_info(self, i):
        print('Training iteration ', i)
        print('Current state: ', self.cur_state)
        print('Choice costs: ', self.table[self.cur_state])
        print('Chosen direction: ', self.direction)
        print('Accumulated cost: ', self.acc_cost)
        print('-' * 20)
        print('\n\n')
        pass

    def play(self):
        # TODO user-defined starting state
        state = 0
        cost = 0
        print('\n\nQ-Table:')
        print(self.table)
        print('Game:')
        print(self.board)
        while cost < 50:
            sleep(1)
            direction = np.nanargmax(self.table[state])
            cost += self.table.item(state, direction)
            state = next_state(state, direction, self.colsize)
            self.board.player_pos = two_dim(state, self.colsize)
            # if player gets stuck in loop
            if cost < -10:
                print('Ended game due to Bad performance,\n'
                      'Please retrain with different parameters.')
                break
            print(self.board)

