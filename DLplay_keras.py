#!/usr/bin/env python3
from __future__ import print_function
import os
import numpy as np
from state import State
import traceback
import base64
import time
import chess
import chess.svg
import matplotlib.pyplot as plt
from kerasModel import ChessModel
import tensorflow as tf
import util

# A simple chess Value function
MAXVAL = 10000

def random_move(s):
	t = time.time()
	moves = s.edges()
	print('No. of moves: ', len(moves))
	#if len(moves):
	#	return
	move = moves[np.random.randint(len(moves))]
	print('Random move', move,' after ', time.time() - t, 'sec', len(moves))
	s.board.push(move)

def predict_moves(model, fen):
	input_plane = util.canon_input_planes(fen)
	#print(input_plane.shape)
	input_plane = np.moveaxis(input_plane, 0, 2)
	#print(input_plane.shape)
	input_plane = np.expand_dims(input_plane, axis = 0)
	#print(input_plane.shape)
	policy, value = model.predict(x = input_plane)
	#print(policy.shape, value.shape)
	return policy, value

##############################################################################################
# A simple chess Value function
MAXVAL = 10000
class ClassicValuator(object):
	values = {	chess.PAWN: 1,
				chess.KNIGHT: 3,
				chess.BISHOP: 3,
				chess.ROOK: 5,
				chess.QUEEN: 9,
				chess.KING: 100}
	def __init__(self):
		self.reset()
		self.memo = {}

	def reset(self):
		self.count = 0

	# simple value function based on pieces
	def __call__(self, s):
		self.count += 1
		key = s.key()  # returns (board.board_fen(), board.turn, board.castling_rights, board.ep_square)
		if key not in self.memo:
			self.memo[key] = self.value(s)
		return self.memo[key]

	def value(self, s):
		b = s.board
		# Game over values
		if b.is_game_over():
			if b.result() == '1-0':
				return MAXVAL
			elif b.result() == '0-1':
				return -MAXVAL
			else:
				return 0

		# Piece Values
		pm = b.piece_map()
		val = 0.0
		for x in pm:
			tval = self.values[pm[x].piece_type]
			if pm[x].color == chess.WHITE:
				val += tval
			else:
				val -= tval

		# Add a number of legal moves term for the value
		bak = b.turn
		b.turn = chess.WHITE
		val += 0.1 * b.legal_moves.count()
		b.turn = chess.BLACK
		val -= 0.1 * b.legal_moves.count()

		b.turn = bak

		return val

def computer_minimax(s, v, depth, a, b, big = False):
	if depth >= 2 or s.board.is_game_over():
		return v(s)

	# white is maximizing player
	turn = s.board.turn
	if turn == chess.WHITE:
		ret = -MAXVAL
	else:
		ret = MAXVAL

	if big:
		bret = []

	isort = []
	for e in s.board.legal_moves:
		s.board.push(e)
		isort.append((v(s), e))
		s.board.pop()
	
	move = sorted(isort, key = lambda x: x[0], reverse = s.board.turn)

	if depth >= 3:
		move = move[:10]

	for e in [x[1] for x in move]:
		s.board.push(e)
		tval = computer_minimax(s, v, depth + 1, a, b)
		s.board.pop()
		if big:
			bret.append((tval, e))
		if turn == chess.WHITE:
			ret = max(ret, tval)
			a = max(a, ret)
			if a >= b:
				break	# b cut - off
		else:
			ret = min(ret, tval)
			b = min(b, ret)
			if a >= b:
				break	# a cut - off
	if big:
		return ret, bret
	else:
		return ret

def explore_leaves(s, v):
	# Function to determine all the possible next moves along with their value for current state s
	ret = []
	v.reset()
	#ret.append((computer_minimax(s, v), e))
	t = time.time()
	cval, ret = computer_minimax(s, v, depth = 0, a = -MAXVAL, b = MAXVAL, big = True)
	print("Explored {} nodes with {} nodes/sec".format(v.count, int(v.count / (time.time() - t))))
	return ret

def computer_move(s, v):
	t = time.time()
	move = sorted(explore_leaves(s, v), key = lambda x: x[0], reverse = s.board.turn)
	print('Computer moves after ', time.time() - t, 'sec')

	if len(move) == 0:
		return

	print("top 3 moves: ")
	for i, m in enumerate(move[:3]):
		print(" ", m)
	print({True: 'WHITE', False: 'BLACK'}[s.board.turn], 'moving ', move[0][1])
	s.board.push(move[0][1])

############################################################################################################
from keras.models import load_model
from keras.models import model_from_json

# Chess board and Engine
s = State()

global graph
graph = tf.get_default_graph()
# load json and create model
json_file = open(os.path.join("nets", "keras_model.json"), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(os.path.join("nets","keras_model_weights_516719_100e.h5"))

def to_svg(s):
	return base64.b64encode(chess.svg.board(board = s.board).encode('utf-8')).decode('utf-8')

from flask import Flask, Response, request

app = Flask(__name__)
@app.route("/")

def hello():
	ret = open("index.html").read()
	return ret.replace('start', s.board.fen())

@app.route("/selfplay")
def selfplay():
	s = State()
	ret = '<html><head>'

	# self play
	while not s.board.is_game_over():
		with graph.as_default():
			policy, value = predict_moves(model, s.board.fen())
		if util.is_black_turn(s.board.fen()):
			labels = util.flipped_uci_labels()
		else:
			labels = util.create_uci_labels()
		if chess.Move.from_uci(labels[np.argmax(policy)]) in s.board.legal_moves:
			pass
		s.board.push(chess.Move.from_uci(labels[np.argmax(policy)]))
		ret += '<img width = 700 height = 700 src = "data:image/svg+xml;base64,%s" /></img><br/>' % to_svg(s)
	print(s.board.result())
	return ret

@app.route("/move")
def move():
	#print("here!!!!!1")
	if not s.board.is_game_over():
		source = int(request.args.get('from', default=''))
		target = int(request.args.get('to', default=''))
		promotion = True if request.args.get('promotion', default='') == 'true' else False
		move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))
		#print("source: {}, target: {}, promotion: {}, move: {}".format(source, target, promotion, move))

		#move = request.args.get('move', default = "")
		if move is not None and move != "":
			print("Human moves: ", move)
			try:
				s.board.push_san(move)
				num_halfmoves = int(s.board.fen().split(' ')[-1])
				num_halfmoves = (2 * num_halfmoves) - 1
				if not s.board.is_game_over():
					with graph.as_default():
						policy, value = predict_moves(model, s.board.fen())
					if util.is_black_turn(s.board.fen()):
						#labels = util.flipped_uci_labels()
						policy = util.flip_policy(policy[0])
					labels = util.labels
					policy = util.cal_policy(labels, policy, list(s.board.legal_moves))
					action = int(np.random.choice(range(util.labels_n), p = policy))
					print("value: ", value[0], "Policy: ", chess.Move.from_uci(labels[action]))
					#s.board.push(chess.Move.from_uci(labels[np.argmax(policy)]))
					s.board.push(chess.Move.from_uci(labels[action]))
			except Exception:
				traceback.print_exc()
			
			response = app.response_class(
        					response=s.board.fen(),
        					status=200
      						)
			#print(s.board.fen())
			return response
	else:
		print("Game is Over")
		response = app.response_class(response = "game over", status = 200)
		return response
	print("hello ran")
	return hello()

@app.route("/newgame")
def newgame():
	s.board.reset()
	response = app.response_class(response=s.board.fen(), status=200)
	return response

if __name__ == '__main__':
	if os.getenv("SELFPLAY") is not None:
		s = State()
		while not s.board.is_game_over():
			print(s.board)
			with graph.as_default():
				policy, value = predict_moves(model, s.board.fen())
			if util.is_black_turn(s.board.fen()):
				labels = util.flipped_uci_labels()
			else:
				labels = util.create_uci_labels()
			policy = util.cal_policy(labels, policy[0], list(s.board.legal_moves))
			action = int(np.random.choice(range(util.labels_n), p = policy))
			s.board.push(chess.Move.from_uci(labels[action]))

		print('Result: ', {'1-0': 'WHITE', '0-1': 'BLACK', '1/2-1/2': 'Draw'}[s.board.result()], s.board.result())

	elif os.getenv("RANDPLAY") is not None:
		s = State()
		result = []
		for game in range(100):
			while not s.board.is_game_over():
				print(s.board)
				if not s.board.is_game_over():
					with graph.as_default():
						policy, value = predict_moves(model, s.board.fen())
					if util.is_black_turn(s.board.fen()):
						#labels = util.flipped_uci_labels()
						policy = util.flip_policy(policy[0])
					#else:
					#	labels = util.create_uci_labels()
					labels = util.labels
					policy = util.cal_policy(labels, policy[0], list(s.board.legal_moves))
					action = int(np.random.choice(range(util.labels_n), p = policy))
					s.board.push(chess.Move.from_uci(labels[action]))
				if s.board.is_game_over():
					#print(s.board.turn)
					break
				random_move(s)
				print('Game: ', game + 1)
			print('Result: ', {'1-0': 'WHITE', '0-1': 'BLACK', '1/2-1/2': 'DRAW'}[s.board.result()], s.board.result())
			result.append({'1-0': 'COMPUTER', '0-1': 'RANDOM', '1/2-1/2': 'DRAW'}[s.board.result()])
			s.board.reset()
			print(result, s.board.is_game_over())
		plt.hist(np.array(result), bins = 3)
		#plt.xticks(np.arange(3), ('COMPUTER', 'RANDOM', 'DRAW'))
		plt.show()

	elif os.getenv("CLASSICPLAY") is not None:
		s = State()
		v = ClassicValuator()
		result = []
		for game in range(100):
			while not s.board.is_game_over():
				if not s.board.is_game_over():
					with graph.as_default():
						policy, value = predict_moves(model, s.board.fen())
					if util.is_black_turn(s.board.fen()):
						#labels = util.flipped_uci_labels()
						policy = util.flip_policy(policy[0])
						print(s.board.fen())
					#else:
					#	labels = util.create_uci_labels()
					labels = util.labels
					policy = util.cal_policy(labels, policy[0], list(s.board.legal_moves))
					action = int(np.random.choice(range(util.labels_n), p = policy))
					s.board.push(chess.Move.from_uci(labels[action]))
					#s.board.push(chess.Move.from_uci(labels[np.argmax(policy)]))
				if s.board.is_game_over():
					#print(s.board.turn)
					break
				computer_move(s, v)
				print(s.board)
				print('Game: ', game)
			print('Result: ', {'1-0': 'WHITE', '0-1': 'BLACK', '1/2-1/2': 'DRAW'}[s.board.result()], s.board.result())
			result.append({'1-0': 'DL', '0-1': 'CLASSICAL', '1/2-1/2': 'DRAW'}[s.board.result()])
			s.board.reset()
			print(result, s.board.is_game_over())
		plt.hist(np.array(result), bins = 3)
		#plt.xticks(np.arange(3), ('COMPUTER', 'RANDOM', 'DRAW'))
		plt.show()

	else:
		app.run(debug = False)