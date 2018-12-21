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

def explore_leaves(s, v):
	# Function to determine all the possible next moves along with their value for current state s
	ret = []
	for e in s.edges():
		s.board.push(e)
		ret.append((v(s), e))
		s.board.pop()
	return ret

############################################################################################################

# Chess board and Engine
s = State()
v = Valuator()

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
	move = sorted(explore_leaves(s, v), key = lambda x: x[0], reverse = s.board.turn)
	print("top 3 moves: ")
	for i, m in enumerate(move[:3]):
		print(" ", m)
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
				computer_move(s, v)
			except Exception:
				traceback.print_exc()
	else:
		print("Game is Over")
	return hello()

if __name__ == '__main__':
	app.run(debug = True)