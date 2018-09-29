
from adversarialsearchproblem import AdversarialSearchProblem
from gamedag import GameDAG, DAGState


def minimax(asp):
    """
	Implement the minimax algorithm on ASPs,
	assuming that the given game is both 2-player and constant-sum

	Input: asp - an AdversarialSearchProblem
	Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
	"""

    start_state = asp.get_start_state()

    action = None
    result = float('-inf')

    for a in asp.get_available_actions(start_state):
        value = min_value(asp, asp.transition(start_state, a))
        if value > result:
            result = value
            action = a

    return action

def max_value(asp, state):
    """
    Input: asp and state taking a maximum among state's children
    output: a value
    """
    if asp.is_terminal_state(state) == True:
        player = asp.get_start_state().player_to_move()
        return asp.evaluate_state(state)[player]
    v = float('-inf')
    for a in asp.get_available_actions(state):
        v = max(v, min_value(asp, asp.transition(state, a)))
    return v

def min_value(asp, state):
    """
    Input: asp and state taking a minimum among state's children
    output: a value
    """
    if asp.is_terminal_state(state) == True:
        player = asp.get_start_state().player_to_move()
        return asp.evaluate_state(state)[player]
    v = float('inf')
    for a in asp.get_available_actions(state):
        v = min(v, max_value(asp, asp.transition(state, a)))
    return v



def alpha_beta(asp):
    """
	Implement the alpha-beta pruning algorithm on ASPs,
	assuming that the given game is both 2-player and constant-sum.

	Input: asp - an AdversarialSearchProblem
	Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
	"""
    start_state = asp.get_start_state()

    action = None
    result = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    for a in asp.get_available_actions(start_state):
        value = alpha_beta_min(asp, asp.transition(start_state, a), alpha, beta)
        if value > result:
            result = value
            action = a
        if value >= beta:
            return action
        alpha = max(alpha, value)
    return action

def alpha_beta_max(asp, state, alpha, beta):
    """
    Input: asp, max state, current alpha, current beta
    Output: value
    """
    if asp.is_terminal_state(state) == True:
        player = asp.get_start_state().player_to_move()
        return asp.evaluate_state(state)[player]
    v = float('-inf')
    for a in asp.get_available_actions(state):
        v = max(v, alpha_beta_min(asp, asp.transition(state, a), alpha, beta))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v

def alpha_beta_min(asp, state, alpha, beta):
    """
    Input: asp, min state, current alpha, current beta
    Output: value
    """
    if asp.is_terminal_state(state) == True:
        player = asp.get_start_state().player_to_move()
        return asp.evaluate_state(state)[player]
    v = float('inf')
    for a in asp.get_available_actions(state):
        v = min(v, alpha_beta_max(asp, asp.transition(state, a), alpha, beta))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v


def alpha_beta_cutoff(asp, cutoff_ply, eval_func):
    """
	This function should:
	- search through the asp using alpha-beta pruning
	- cut off the search after cutoff_ply moves have been made.

	Inputs:
		asp - an AdversarialSearchProblem
		cutoff_ply- an Integer that determines when to cutoff the search
			and use eval_func.
			For example, when cutoff_ply = 1, use eval_func to evaluate
			states that result from your first move. When cutoff_ply = 2, use
			eval_func to evaluate states that result from your opponent's
			first move. When cutoff_ply = 3 use eval_func to evaluate the
			states that result from your second move.
			You may assume that cutoff_ply > 0.
		eval_func - a function that takes in a GameState and outputs
			a real number indicating how good that state is for the
			player who is using alpha_beta_cutoff to choose their action.
			You do not need to implement this function, as it should be provided by
			whomever is calling alpha_beta_cutoff, however you are welcome to write
			evaluation functions to test your implemention

	Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
	"""
    start_state = asp.get_start_state()
    player = start_state.player_to_move()

    action = None
    result = float('-inf')

    alpha = float('-inf')
    beta = float('inf')

    for a in asp.get_available_actions(start_state):
        value = cutoff_min(asp, asp.transition(start_state, a), alpha, beta, cutoff_ply, 1, eval_func, player)
        if value > result:
            result = value
            action = a
        if value >= beta:
            return action
        alpha = max(alpha, value)
    return action

def cutoff_max(asp, state, alpha, beta, cutoff, depth, eval_func, player):
    """
    Output: value and depth
    """
    if asp.is_terminal_state(state):
        return asp.evaluate_state(state)[player]
    if depth == cutoff:
        return (eval_func(state))
    v = float('-inf')
    for a in asp.get_available_actions(state):
        v = max(v, cutoff_min(asp, asp.transition(state, a), alpha, beta, cutoff, depth + 1, eval_func, player))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v


def cutoff_min(asp, state, alpha, beta, cutoff, depth, eval_func, player):
    """
    Output: value and depth
    """
    if asp.is_terminal_state(state):
        return asp.evaluate_state(state)[player]
    if depth == cutoff:
        return (eval_func(state))
    v = float('inf')
    for a in asp.get_available_actions(state):
        v = min(v, cutoff_max(asp, asp.transition(state, a), alpha, beta, cutoff, depth + 1, eval_func, player))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v

def general_minimax(asp):
    """
	Implement the generalization of the minimax algorithm that was
	discussed in the handout, making no assumptions about the
	number of players or reward structure of the given game.

	Input: asp - an AdversarialSearchProblem
	Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
	"""

    start_state = asp.get_start_state()
    player = start_state.player_to_move()
    max_action = None
    maximum = float('-inf')
    for action in asp.get_available_actions(start_state):
        opposite = asp.transition(start_state, action)
        if opposite.player_to_move() == player:
            s = general_max(opposite, player, asp)
            if s > maximum:
                maximum = s
                max_action = action
        else:
            s = general_min(opposite, player, asp)
            if s > maximum:
                maximum = s
                max_action = action
    return max_action
        # return maximum

def general_max(state, player, asp):
    if asp.is_terminal_state(state):
        return asp.evaluate_state(state)[player]
    v = float('-inf')
    for action in asp.get_available_actions(state):
        opposite = asp.transition(state, action)
        if opposite.player_to_move() == player:
            v = max(v, general_max(opposite, player, asp))
        else:
            v = max(v, general_min(opposite, player, asp))
    return v


def general_min(state, player, asp):
    if asp.is_terminal_state(state):
        return asp.evaluate_state(state)[player]
    v = float('inf')
    for action in asp.get_available_actions(state):
        opposite = asp.transition(state, action)
        if opposite.player_to_move() == player:
            v = min(v, general_max(opposite, player, asp))
        else:
            v = min(v, general_min(opposite, player, asp))
    return v
