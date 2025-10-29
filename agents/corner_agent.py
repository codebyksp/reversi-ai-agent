# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("second_agent")
class SecondAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(SecondAgent, self).__init__()
        self.name = "SecondAgent"
        
    def step(self, chess_board, player, opponent):
        """
        Enhanced step function for strategic play using heuristics.
        """
        start_time = time.time()
        valid_moves = get_valid_moves(chess_board, player)

        if not valid_moves:
            return None  # Pass turn if no valid moves

        best_move = None
        best_score = float('-inf')

        # Heuristic weights
        CORNER_WEIGHT = 10
        MOBILITY_WEIGHT = 2
        STABILITY_WEIGHT = 5

        for move in valid_moves:
            # Copy the board and simulate the move
            simulated_board = deepcopy(chess_board)
            execute_move(simulated_board, move, player)

            # Evaluate the move using a heuristic function
            move_score = self.evaluate_board(simulated_board, player, opponent, 
                                             CORNER_WEIGHT, MOBILITY_WEIGHT, STABILITY_WEIGHT)

            # Update the best move if the current move is better
            if move_score > best_score:
                best_score = move_score
                best_move = move

            # Break if close to the time limit
            if time.time() - start_time > 1.9:
                break

        print(f"My AI's turn took {time.time() - start_time:.2f} seconds.")
        return best_move if best_move else random_move(chess_board, player)

    def evaluate_board(self, board, player, opponent, corner_weight, mobility_weight, stability_weight):
        """
        Evaluate the board state using heuristics.
        """
        # Corners positions
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_score = sum(corner_weight for corner in corners if board[corner] == player)
        corner_penalty = sum(-corner_weight for corner in corners if board[corner] == opponent)

        # Mobility: difference in valid moves
        player_mobility = len(get_valid_moves(board, player))
        opponent_mobility = len(get_valid_moves(board, opponent))
        mobility_score = mobility_weight * (player_mobility - opponent_mobility)

        # Stability: count discs on edges
        edge_positions = [
            (0, i) for i in range(board.shape[1])
        ] + [
            (board.shape[0] - 1, i) for i in range(board.shape[1])
        ] + [
            (i, 0) for i in range(board.shape[0])
        ] + [
            (i, board.shape[1] - 1) for i in range(board.shape[0])
        ]
        stability_score = stability_weight * sum(1 for pos in edge_positions if board[pos] == player)

        # Combine scores
        total_score = corner_score + corner_penalty + mobility_score + stability_score
        return total_score

