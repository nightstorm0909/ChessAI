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
	if util.is_black_turn(fen):
		print(fen)
	#print(input_plane.shape)
	input_plane = np.moveaxis(input_plane, 0, 2)
	#print(input_plane.shape)
	input_plane = np.expand_dims(input_plane, axis = 0)
	#print(input_plane.shape)
	policy, value = model.predict(x = input_plane)
	#print(policy.shape, value.shape)
	return policy, value

############################################################################################################
from keras.models import load_model

# Chess board and Engine
s = State()
global graph
graph = tf.get_default_graph()
model = load_model(os.path.join("nets","keras_model_weights_516719_100e.h5"))

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
				if not s.board.is_game_over():
					with graph.as_default():
						policy, value = predict_moves(model, s.board.fen())
					if util.is_black_turn(s.board.fen()):
						labels = util.flipped_uci_labels()
					else:
						labels = util.create_uci_labels()
					policy = util.cal_policy(labels, policy[0], list(s.board.legal_moves))
					action = int(np.random.choice(range(util.labels_n), p = policy))
					print("value: ", value[0], "Policy: ", chess.Move.from_uci(labels[action]))
					#print("possible moves: ", chess.Move.from_uci(labels[np.argmax(policy)]) in (s.board.legal_moves))
					
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
		for game in range(10):
			while not s.board.is_game_over():
				print(s.board)
				if not s.board.is_game_over():
					with graph.as_default():
						policy, value = predict_moves(model, s.board.fen())
					if util.is_black_turn(s.board.fen()):
						labels = util.flipped_uci_labels()
					else:
						labels = util.create_uci_labels()
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
	else:
		app.run(debug = False)