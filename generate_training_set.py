#!/usr/bin/env python3
import os
import chess.pgn
from state import State
import numpy as np

# pgn files are in the data folder
def get_dataset(num_samples = None):
	gn = 0
	X, Y = [], []
	values = {"1/2-1/2": 0, "0-1": -1, "1-0": 1}
	for fn in os.listdir("data"):
		pgn = open(os.path.join("data", fn))
		#print(os.path.join("data", fn))
		while 1:
			try:
				game = chess.pgn.read_game(pgn)
			except Exception:
				break
			res = game.headers["Result"]
			if res not in values:
				continue
			value = values[res]
			board = game.board()		# Gets the starting position of the game
			for i, move in enumerate(game.main_line()):
				board.push(move)
				ser = State(board).serialize()	# state: [8, 8, 5]
				X.append(ser)
				Y.append(value)
				#print(np.array(X).shape)
			print("Parsing game {}, got {} examples".format(gn, len(X)))
			if num_samples is not None and len(X) > num_samples:
				return X, Y

			gn += 1
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

if __name__ == "__main__":
	X, Y = get_dataset(1000000)
	print(type(X))
	np.savez(os.path.join("processed", "dataset.npz"), X, Y)
	#h5 = h5py.File(os.path.join("processed", "trainme.h5"), "w")
	