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
import torch
from train import Net
import matplotlib.pyplot as plt

class Valuator(object):

	def __init__(self):
		vals = torch.load("nets/value_2M.pth", map_location = lambda storage, loc: storage)
		self.model = Net()
		self.model.load_state_dict(vals)
	
	def reset(self):
		self.count = 0

	def __call__(self, s):
		self.count += 1
		brd = s.serialize()[None]
		output = self.model(torch.tensor(brd).float())
		return float(output.data[0][0])

# A simple chess Value function
MAXVAL = 10000


def computer_minimax(s, v, depth, a, b, big = False):
	if depth >= 3 or s.board.is_game_over():
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

def random_move(s, v):
	t = time.time()
	moves = s.edges()
	print('No. of moves: ', len(moves))
	#if len(moves):
	#	return
	move = moves[np.random.randint(len(moves))]
	print('Random move', move,' after ', time.time() - t, 'sec', len(moves))
	s.board.push(move)


############################################################################################################

# Chess board and Engine
s = State()
v = Valuator()
#v = ClassicValuator()

def to_svg(s):
	return base64.b64encode(chess.svg.board(board = s.board).encode('utf-8')).decode('utf-8')

from flask import Flask, Response, request

app = Flask(__name__)
@app.route("/")

def hello():
	ret = open("index.html").read()
	return ret.replace('start', s.board.fen())

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

@app.route("/selfplay")
def selfplay():
	s = State()
	ret = '<html><head>'

	# self play
	while not s.board.is_game_over():
		computer_move(s, v)
		ret += '<img width = 700 height = 700 src = "data:image/svg+xml;base64,%s" /></img><br/>' % to_svg(s)
	print(s.board.result())
	return ret

@app.route("/move")
def move():
	print("here!!!!!1")
	if not s.board.is_game_over():
		source = int(request.args.get('from', default=''))
		target = int(request.args.get('to', default=''))
		promotion = True if request.args.get('promotion', default='') == 'true' else False
		move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))
		#move = request.args.get('move', default = "")
		if move is not None and move != "":
			print("Human moves: ", move)
			try:
				s.board.push_san(move)
				computer_move(s, v)
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
			computer_move(s, v)
		#print(s.board.result())
		print('Result: ', {'1-0': 'WHITE', '0-1': 'BLACK', '1/2-1/2': 'Draw'}[s.board.result()], s.board.result())

	elif os.getenv("RANDPLAY") is not None:
		s = State()
		result = []
		for game in range(100):
			while not s.board.is_game_over():
				print(s.board)
				computer_move(s, v)
				if s.board.is_game_over():
					#print(s.board.turn)
					break
				random_move(s, v)
				print('Game: ', game)
			print('Result: ', {'1-0': 'WHITE', '0-1': 'BLACK', '1/2-1/2': 'DRAW'}[s.board.result()], s.board.result())
			result.append({'1-0': 'COMPUTER', '0-1': 'RANDOM', '1/2-1/2': 'DRAW'}[s.board.result()])
			s.board.reset()
			print(result, s.board.is_game_over())
		plt.hist(np.array(result), bins = 3)
		#plt.xticks(np.arange(3), ('COMPUTER', 'RANDOM', 'DRAW'))
		plt.show()
	else:
		app.run(debug = True)