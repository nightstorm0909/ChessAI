#!/usr/bin/env python3
from state import State
from train import Net
import traceback
import base64
import time
import torch
import chess
import chess.svg

class Valuator(object):

	def __init__(self):
		vals = torch.load("nets/value.pth", map_location = lambda storage, loc: storage)
		self.model = Net()
		self.model.load_state_dict(vals)
	
	def __call__(self, s):
		brd = s.serialize()[None]
		output = self.model(torch.tensor(brd).float())
		return float(output.data[0][0])

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
		pass

	# simple value function based on pieces
	def __call__(self, s):
		if s.board.is_variant_win():
			if s.board.turn == chess.WHITE:
				return MAXVAL
			else:
				return -MAXVAL
		if s.board.is_variant_loss():
			if s.board.turn == chess.WHITE:
				return -MAXVAL
			else:
				return MAXVAL
		pm = s.board.piece_map()
		val = 0
		for x in pm:
			tval = self.values[pm[x].piece_type]
			if pm[x].color == chess.WHITE:
				val += tval
			else:
				val -= tval
		return val

def computer_minimax(s, v, depth = 3):
	if depth == 0 or s.board.is_game_over():
		return v(s)

	# white is maximizing player
	turn = s.board.turn
	if turn == chess.WHITE:
		ret = -MAXVAL
	else:
		ret = MAXVAL

	for e in s.edges():
		s.board.push(e)
		tval = computer_minimax(s, v, depth - 1)
		if turn == chess.WHITE:
			ret = max(ret, tval)
		else:
			ret = min(ret, tval)
		s.board.pop()
	return ret

def explore_leaves(s, v):
	# Function to determine all the possible next moves along with their value for current state s
	ret = []
	for e in s.edges():
		s.board.push(e)
		#ret.append((v(s), e))
		ret.append((computer_minimax(s, v), e))
		s.board.pop()
	return ret

############################################################################################################

# Chess board and Engine
s = State()
#v = Valuator()
v = ClassicValuator()

def to_svg(s):
	return base64.b64encode(chess.svg.board(board = s.board).encode('utf-8')).decode('utf-8')

from flask import Flask, Response, request

app = Flask(__name__)
@app.route("/")

def hello():
	board_svg = to_svg(s)
	ret = '<html><head>'
	ret += '<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>'
	ret += '<style>input {font-size: 30px;} button {font-size: 30px;}</style>'
	ret += '</head><body>'
	ret += '<a href = "/selfplay">Play vs itself</a><br/>'
	ret += '<img width=700 height=700 src="data:image/svg+xml;base64,%s" /></img><br/>' % board_svg 
	ret += '<form action = "/move"><input id = "move" name = "move" type = "text"></input><input type = "submit" value = "Human Move"></form><br?>'
	ret += '<script>$(function() {var input = document.getElementById("move"); console.log("selected");input.focus(); input.select();});</script>'
	
	return ret

def computer_move(s, v):
	t = time.time()
	move = sorted(explore_leaves(s, v), key = lambda x: x[0], reverse = s.board.turn)
	print("Time taken by computer to make move: ", time.time() - t, 'sec')
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
	if not s.board.is_game_over():
		move = request.args.get('move', default = "")
		if move is not None and move != "":
			print("Human moves: ", move)
			try:
				s.board.push_san(move)
				#hello()
				computer_move(s, v)
			except Exception:
				traceback.print_exc()
	else:
		print("Game is Over")
	return hello()

if __name__ == '__main__':
	app.run(debug = True)