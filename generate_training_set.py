#!/usr/bin/env python3
import os
import chess.pgn
import util
from state import State
import numpy as np
import argparse
import pickle

# pgn files are in the data folder
def get_dataset(num_samples = None):
	gn = 0
	board_positions = 0
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
			print(game)
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
				board_positions += 1
			#print("Parsing game {}, got {} examples".format(gn, len(X)))
			print("Parsing game {}, got {} examples".format(gn, board_positions))
			#if num_samples is not None and len(X) > num_samples:
			if num_samples is not None and board_positions > num_samples:
				return X, Y

			gn += 1
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

def get_dataset_keras(num_samples = None):
	gn = 0
	board_positions = 0
	X, Y = [], []
	input_planes, policy_list, value_list = [], [], []
	values = {"1/2-1/2": 0, "0-1": -1, "1-0": 1}

	for file in os.listdir("data"):
		print("Processing games in file ", file)
		pgn = open(os.path.join("data", file), errors='ignore')
		offsets = list(chess.pgn.scan_offsets(pgn))         # One pgn file contains many games
		print(file, "has {} games".format(len(offsets)))
		#gn += len(offsets)
		for offset in offsets:
			pgn.seek(offset)
			game = chess.pgn.read_game(pgn)
			#print(game)
			gn += 1
			result = game.headers["Result"]
			#value = values[result]
			if result == '1-0':
				black_win = -1
			elif result == '0-1':
				black_win = 1
			else:
				black_win = 0

			white_elo, black_elo = int(game.headers["WhiteElo"]), int(game.headers["BlackElo"])
			white_weight = util.clip_elo_policy(white_elo) 
			black_weight = util.clip_elo_policy(black_elo)

			actions = []
			while not game.is_end():
				game = game.variation(0)
				actions.append(game.move.uci())     # all the moves in uci format in the game
			#print("[INFO] Actions: ", len(actions), result, actions)

			board = chess.Board()
			for k in range(len(actions)):
				#print("main: ", board.fen(), board.turn, '\n', board, actions[k])
				input_planes.append(State(board).convert_to_input_planes())		# returns (18, 8, 8) representation of the game state
				
				policy_list.append(util.sl_action(board.fen(), actions[k]))
				
				move_number = int(board.fen().split(' ')[5])
				value_certainty = min(5, move_number) / 5
				#sl_value = (value * value_certainty) + util.valuefn(board.fen(), False)*(1 - value_certainty)
				if board.turn == chess.WHITE:
					sl_value = (-black_win * value_certainty) + util.valuefn(board.fen(), False)*(1 - value_certainty)
				else:
					sl_value = (black_win * value_certainty) + util.valuefn(board.fen(), False)*(1 - value_certainty)
				value_list.append(sl_value)

				#print(len(input_planes), len(policy_list), len(value_list), value_list[-1])
				board_positions += 1
				board.push_uci(actions[k])
				print("[INFO] Game: {}; Total board positions: {}".format(gn, board_positions))

			if (board_positions > num_samples) or (gn >= 10000):
				print("[INFO] Total games: ", gn, "Board positions: ", board_positions)
				return input_planes, policy_list, value_list, gn
		#break
	print("[INFO] Total games: ", gn, "Board positions: ", board_positions)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--type", type = str, default = "pytorch", help = "Creating dataset for pytorch model or keras model")
	parser.add_argument("--n", type = int, default = 100, help = "Number of board positions")
	args = parser.parse_args()

	if args.type == "pytorch":
		X, Y = get_dataset(args.n)
		np.savez(os.path.join("processed", "dataset.npz"), X, Y)
	elif args.type == "keras":
		state_list, policy_list, value_list, games = get_dataset_keras(args.n)
		temp = [state_list, policy_list, value_list]
		np.savez(os.path.join("processed", "dataset_keras_{}_1M.npz".format(games)), state_list, policy_list, value_list)
		#with open(os.path.join("processed", "dataset_keras.pickle"), 'wb') as file:
		#	pickle.dump(temp, file)
	else:
		print("Wrong value for the --type argument. It takes only 'keras' or 'pytorch'")
	#print(type(X))
	#np.savez(os.path.join("processed", "dataset.npz"), X, Y)
	#h5 = h5py.File(os.path.join("processed", "trainme.h5"), "w")