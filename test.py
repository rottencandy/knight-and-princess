import knight
import numpy as np

# Tiles can be numpy matrix of any size and can have any type of tiles
# Just all type of tiles to the costs dict.
tiles = np.array([
    ['O', 'O', 'O', 'O', 'O'],
    ['O', 'E' ,'O', 'E', 'O'],
    ['O', 'O', 'O', 'O', 'O'],
    ['E', 'E', 'O' ,'O', 'O'],
    ['O', 'O', 'P', 'O', 'O']])

# The algorithm will try to move towards the tiles with highest costs.
# You can add any number of new tiles.
# O : Nothing
# E : Enemy
# P : Princess
costs = {'O': -1, 'E': -100, 'P': 100}

# Initialize the board(tiles) and q-table
b = knight.Board(tiles, costs)
q = knight.Qtable(b)

# Train q table for 500 iterations
q.train(500, learn_rate=0.1, discount=0.9, epsilon=1)

# Start the game
q.play()

