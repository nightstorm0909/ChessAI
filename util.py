import chess
import chess.pgn
import numpy as np

pieces_order = 'KQRBNPkqrbnp' # 12x8x8
castling_order = 'KQkq'       # 4x8x8
ind = {pieces_order[i]: i for i in range(12)}

min_elo_policy = 500
max_elo_policy = 1800
def clip_elo_policy(elo):
    return min(1, max(0, elo - min_elo_policy) / max_elo_policy)	# 0 until min_elo, 1 after max_elo, linear in between

def create_uci_labels():
    """
    Creates the labels for the universal chess interface into an array and returns them
    :return:
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                           [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + '2' + l + '1' + p)
            labels_array.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + '2' + l_l + '1' + p)
                labels_array.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + '2' + l_r + '1' + p)
                labels_array.append(l + '7' + l_r + '8' + p)
    return labels_array

def flipped_uci_labels():
	# it flips the board and changes the policy from the perspective of black
	def repl(x):
		return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])
	return [repl(x) for x in create_uci_labels()]

############## Board representation helper functions###########################################
def maybe_flip_fen(fen, flip = False):
    if not flip:
        return fen
    foo = fen.split(' ')
    rows = foo[0].split('/')
    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a
    def swapall(aa):
        return "".join([swapcase(a) for a in aa])
    return "/".join([swapall(row) for row in reversed(rows)]) \
        + " " + ('w' if foo[1] == 'b' else 'b') \
        + " " + "".join(sorted(swapall(foo[2]))) \
        + " " + foo[3] + " " + foo[4] + " " + foo[5]


def all_input_planes(fen):
	current_aux_planes = aux_planes(fen)    # [6, 8, 8]: [castling(4, 8, 8), fifty_move(1, 8, 8), en_passant(1, 8, 8)]
	history_both = to_planes(fen)
	ret = np.vstack((history_both, current_aux_planes))
	assert ret.shape == (18, 8, 8)
	return ret

def aux_planes(fen):
	foo = fen.split(' ')
	en_passant = np.zeros((8, 8), dtype=np.float32)
	if foo[3] != '-':
		eps = alg_to_coord(foo[3])     # [rank, file]
		en_passant[eps[0]][eps[1]] = 1
	fifty_move_count = int(foo[4])
	fifty_move = np.full((8, 8), fifty_move_count, dtype=np.float32)
	castling = foo[2]
	auxiliary_planes = [np.full((8, 8), int('K' in castling), dtype=np.float32), np.full((8, 8), int('Q' in castling), dtype=np.float32),
						np.full((8, 8), int('k' in castling), dtype=np.float32), np.full((8, 8), int('q' in castling), dtype=np.float32),
						fifty_move, en_passant]
	ret = np.asarray(auxiliary_planes, dtype=np.float32)
	assert ret.shape == (6, 8, 8)
	return ret

def alg_to_coord(alg):
	rank = 8 - int(alg[1])        # 0-7
	file = ord(alg[0]) - ord('a') # 0-7
	return rank, file

def to_planes(fen):
	board_state = replace_tags_board(fen)
	pieces_both = np.zeros(shape=(12, 8, 8), dtype=np.float32)
	for rank in range(8):  # rank = 0-8
		for file in range(8):   # file: a-h
			v = board_state[rank * 8 + file]
			if v.isalpha():
				pieces_both[ind[v]][rank][file] = 1
	assert pieces_both.shape == (12, 8, 8)
	return pieces_both

def replace_tags_board(board_san):
	'''
	input: 'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1'; 
	output:'rnbqkbnrpppppppp11111111111111111111111111111N11PPPPPPPPRNBQKB1R'
	'''
	board_san = board_san.split(" ")[0]
	board_san = board_san.replace("2", "11")
	board_san = board_san.replace("3", "111")
	board_san = board_san.replace("4", "1111")
	board_san = board_san.replace("5", "11111")
	board_san = board_san.replace("6", "111111")
	board_san = board_san.replace("7", "1111111")
	board_san = board_san.replace("8", "11111111")
	return board_san.replace("/", "")

def is_black_turn(fen):
	return fen.split(" ")[1] == 'b'

def canon_input_planes(fen):
	#  return : (18, 8, 8) representation of the game state
	fen = maybe_flip_fen(fen, is_black_turn(fen)) # Flips the board if it is black's turn
	return all_input_planes(fen)    # returns input for model of shape (18, 8, 8)

#board = chess.Board()
#result = canon_input_planes(board.fen())

####################### Supervised learning actions ###################################
labels = create_uci_labels()
labels_n = len(labels)
move_lookup = {chess.Move.from_uci(move): i for move, i in zip(labels, range(labels_n))}
flipped_labels = flipped_uci_labels()
unflipped_index = [labels.index(x) for x in flipped_labels]

def flip_policy(pol):
	# Rearranges the given policy by flipping the board i.e considering black move from the white perspective
	return np.asarray([pol[ind] for ind in unflipped_index])

def sl_action(fen, my_action, weight=1):
	# Convert the action in uci format to NN output format
	policy = np.zeros(labels_n)
	k = move_lookup[chess.Move.from_uci(my_action)]
	policy[k] = weight

	if is_black_turn(fen):
		policy = flip_policy(policy)

	return policy

####################### Value Function ###############################################
def valuefn(fen, absolute = False):
	# it returns the value of the current board based on what pieces are present on the board
	piece_vals = {'K': 3, 'Q': 14, 'R': 5, 'B': 3.25, 'N': 3, 'P': 1}
	ans = 0.0
	tot = 0
	for c in fen.split(' ')[0]:
		if not c.isalpha():
			continue
		if c.isupper():
			ans += piece_vals[c]
			tot += piece_vals[c]
		else:
			ans -= piece_vals[c.upper()]
			tot += piece_vals[c.upper()]
	v = ans/tot
	if not absolute and is_black_turn(fen):
		v = -v
	assert abs(v) < 1
	#print(ans, tot, v, np.tanh(v * 3))
	return np.tanh(v * 3) # arbitrary

########################## Policy Renormalization #######################################
def cal_policy(labels, pol, legal_moves):
	move_lookup = {chess.Move.from_uci(move): i for move, i in zip(labels, range(labels_n))}
	policy = np.zeros(labels_n)
	for move in legal_moves:
		policy[move_lookup[move]] = pol[move_lookup[move]]
	policy /= np.sum(policy)
	return policy

############################ Self Play utilities #########################################
def state_key(board):
	fen = board.rsplit(' ', 1) # drop the move clock
	return fen[0]