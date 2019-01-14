import os
import util
import numpy as np
from state import State
import copy
import chess
import chess.pgn
from collections import defaultdict
import tensorflow as tf
from keras.models import load_model
from keras.models import model_from_json

class ActionStats:
	def __init__(self):
		self.n = 0  # number of visits to this action by the algorithm
		self.w = 0  # every time a child of this action is visited by the algorithm, this accumulates the value (calculated from the value network) of that child.
		self.q = 0  # mean action value (total value from all visits to this action, divided by the total number of visits to this action) just w / n.
		self.p = 0  # prior probability of taking this action, given by the policy network.

class VisitStats:
	def __init__(self):
		self.a = defaultdict(ActionStats)
		self.sum_n = 0

class ChessPlayer:
	def __init__(self, model, s):
		self.model = model
		self.moves = []
		self.tree = defaultdict(VisitStats)
		self.labels_n = util.labels_n
		self.labels = util.labels
		self.move_lookup = {chess.Move.from_uci(move): i for move, i in zip(self.labels, range(self.labels_n))}
		#self.s = s
		self.simulation_per_move = 100

	def reset(self):
		self.tree = defaultdict(VisitStats)

	def action(self, s, num_halfmoves, graph = None, can_stop = True):
		self.reset()
		root_value, naked_value = self.search_moves(s, graph) # returns max value of all values predicted by each simulation and the first value that was predicted
		policy = self.calc_policy(s.board) # returns a list of probabilities of taking each action, calculated based on visit counts
		my_action = int(np.random.choice(range(self.labels_n), p = self.apply_temperature(policy, num_halfmoves)))
		self.moves.append([s.board.fen(), list(policy)])
		return self.labels[my_action]

	def search_moves(self, state, graph = None):
		vals = []
		board = state.board.copy()
		for i in range(self.simulation_per_move):
			s = State(board)
			#print('simulation no.: ', i + 1, '    ', s.board.fen())
			vals.append(self.search_my_moves(s, graph = graph, is_root_node=True))
		return np.max(vals), vals[0]

	def search_my_moves(self, s, graph = None, is_root_node = False):
		#print(s.board.fen())
		if s.board.is_game_over():
			if s.board.result() == '1/2-1/2':
				return 0
			return -1
		
		state = util.state_key(s.board.fen())
		if state not in self.tree:
			#print('hi2')
			leaf_p, leaf_v = self.expand_and_evaluate(s.board.fen(), graph = graph) # returns the policy and value predictions for this state
			#print(leaf_p.shape)
			leaf_p = np.squeeze(leaf_p)
			#print(leaf_p.shape)
			self.tree[state].p = leaf_p # policy
			return leaf_v
		
		# SELECT STEP
		#print('hi3')
		action_t = self.select_action_q_and_u(s.board, is_root_node) # return chess.Move.from_uci: the move to explore e.g: Move.from_uci('g1h3')
		virtual_loss = 3

		my_visit_stats = self.tree[state]
		my_stats = my_visit_stats.a[action_t]
		my_visit_stats.sum_n += virtual_loss
		my_stats.n += virtual_loss
		my_stats.w += -virtual_loss
		my_stats.q = my_stats.w / my_stats.n

		s.board.push_uci(action_t.uci())
		leaf_v = self.search_my_moves(s)  # next move from enemy POV
		leaf_v = -leaf_v
		s.board.pop()

		# BACKUP STEP
		# on returning search path update: N, W, Q
		#print('hi4')
		my_visit_stats.sum_n += -virtual_loss + 1
		my_stats.n += -virtual_loss + 1
		my_stats.w += virtual_loss + leaf_v
		my_stats.q = my_stats.w / my_stats.n

		return leaf_v


	def select_action_q_and_u(self, board, is_root_node):
		'''
		Picks the next action based on which action maximizes the 
		action value (ActionStats.q) + an upper confidence bound on that action.
		'''
		#print('poo1')
		state = util.state_key(board.fen())
		my_visitstats = self.tree[state]
		if my_visitstats.p is not None:  # This is used to copy tree[state].p to tree[state].a[move].p
			tot_p = 1e-8
			#print('my_visitstats.p.shape: ', my_visitstats.p.shape)
			for mov in board.legal_moves:
				#print(mov, self.move_lookup[mov], my_visitstats.p[self.move_lookup[mov]])
				mov_p = my_visitstats.p[self.move_lookup[mov]] # probability of the move according to the policy
				my_visitstats.a[mov].p = mov_p  # Prior probability for the given move according to the policy
				tot_p += mov_p
			for a_s in my_visitstats.a.values():
				a_s.p /= tot_p
			my_visitstats.p = None
		
		xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

		e = 0.25 # noise_eps added to the prior policy to ensure exploration
		c_puct = 1.5
		dir_alpha = 0.3

		#print('poo2')

		best_s = -999
		best_a = None
		if is_root_node:
			noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))
		i = 0
		for action, a_s in my_visitstats.a.items():
			p_ = a_s.p  # Prior probability for the given action
			if is_root_node:
				p_ = (1-e) * p_ + e * noise[i]
				i += 1
			b = a_s.q + (c_puct * p_ * xx_) / (1 + a_s.n) # q + upper confidence bound (c_puct * p_ * xx_ / (1 + a_s.n))
			if b > best_s:
				best_s = b
				best_a = action
		#print('here')
		return best_a


	def expand_and_evaluate(self, fen, graph = None):
		input_plane = util.canon_input_planes(fen)
		input_plane = np.moveaxis(input_plane, 0, 2)
		input_plane = np.expand_dims(input_plane, axis = 0)
		if graph is not None:
			with graph.as_default():
				policy, value = self.model.predict(x = input_plane)
		else:
			policy, value = self.model.predict(x = input_plane)
		if util.is_black_turn(fen):
			policy = util.flip_policy(policy[0])
		return policy, value

	def calc_policy(self, board):
		'''
		return list(float): a list of probabilities of taking each action, calculated based on visit counts.
		'''
		state = util.state_key(board.fen())
		my_visitstats = self.tree[state]
		policy = np.zeros(self.labels_n)
		for action, a_s in my_visitstats.a.items():
			policy[self.move_lookup[action]] = a_s.n
		policy /= np.sum(policy)
		return policy

	def apply_temperature(self, policy, turn):
		'''
		return: policy, randomly perturbed based on the temperature. High temp = more perturbation. Low temp
		'''
		tau = np.power(0.99, turn + 1)
		if tau < 0.1:
			tau = 0
		if tau == 0:
			action = np.argmax(policy)
			ret = np.zeros(self.labels_n)
			ret[action] = 1.0
			return ret
		else:
			ret = np.power(policy, 1/tau)
			ret /= np.sum(ret)
			return ret

	def finish_game(self, z):
		'''
		param z: win=1, lose=-1, draw=0
		'''
		for move in self.moves:  # add this game winner result to all past moves.
			move += [z]