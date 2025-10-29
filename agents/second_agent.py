# THIS IS VERSION 8.2 - changed the move sorting logic and added another heuristic
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
    A class implementing an AI agent for Reversi/Othello using a minimax-based algorithm
    with alpha-beta pruning. It also includes a heuristic that prioritizes corners,
    stability, mobility, and captured pieces.
    """

    def __init__(self):
        super(SecondAgent, self).__init__()
        self.name = "SecondAgent"
        self.time_limit = 1.90  # Ensure decisions complete within 2 seconds

    def step(self, chess_board, player, opponent):
        start_time = time.time()

        # Get all valid moves for the current player
        valid_moves = get_valid_moves(chess_board, player)

        if not valid_moves:
            return None  # No valid moves available, pass turn

        # Initialize variables for iterative deepening search
        best_move = None
        depth = 1

        while time.time() - start_time < self.time_limit:
            try:
                best_move = self.alpha_beta_search(chess_board, player, opponent, depth, start_time)
                depth += 1
            except TimeoutError:
                break

        if best_move is None:
            best_move = valid_moves[0]  # Fallback to a valid move

        # time_taken = time.time() - start_time
        # print(“My AI’s turn took”, time_taken, “seconds. Depth reached:”, depth)

        return best_move

    def alpha_beta_search(self, board, player, opponent, depth, start_time):
        valid_moves = get_valid_moves(board, player)
        if not valid_moves:
            return None

        # Simulate each move and store the resulting board state
        simulated_boards = []
        for move in valid_moves:
            simulated_board = deepcopy(board)
            execute_move(simulated_board, move, player)
            simulated_boards.append((simulated_board, move))

        # Sort the simulated boards based on the evaluation
        simulated_boards.sort(
            key=lambda x: self.evaluate_board(x[0], player, opponent),
            reverse=True
        )

        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        # Perform alpha-beta search using the sorted simulated boards
        for simulated_board, move in simulated_boards:
            if time.time() - start_time > self.time_limit:
                raise TimeoutError("Time limit exceeded")

            value = self.min_value(simulated_board, player, opponent, depth - 1, alpha, beta, start_time)
            if value > alpha:
                alpha = value
                best_move = move

        return best_move

    def max_value(self, board, player, opponent, depth, alpha, beta, start_time):
        if time.time() - start_time > self.time_limit:
            raise TimeoutError("Time limit exceeded")

        if depth == 0 or check_endgame(board, player, opponent)[0]:
            return self.evaluate_board(board, player, opponent)

        value = float('-inf')
        for move in get_valid_moves(board, player):
            simulated_board = deepcopy(board)
            execute_move(simulated_board, move, player)
            value = max(value, self.min_value(simulated_board, player, opponent, depth - 1, alpha, beta, start_time))
            if value >= beta:
                return value
            alpha = max(alpha, value)

        return value

    def min_value(self, board, player, opponent, depth, alpha, beta, start_time):
        if time.time() - start_time > self.time_limit:
            raise TimeoutError("Time limit exceeded")

        if depth == 0 or check_endgame(board, player, opponent)[0]:
            return self.evaluate_board(board, player, opponent)

        value = float('inf')
        for move in get_valid_moves(board, opponent):
            simulated_board = deepcopy(board)
            execute_move(simulated_board, move, opponent)
            value = min(value, self.max_value(simulated_board, player, opponent, depth - 1, alpha, beta, start_time))
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value

    def corner_control_evaluation(self, board, player, opponent):
        """
        Evaluate corner control, C-squares, X-squares, and stability.
        Returns a single combined score.
        """
        # Define corners and their associated C-squares and X-squares
        corners = [(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)]
        corner_to_csquares = {
            (0, 0): [(0, 1), (1, 0)],  # Top-left corner
            (0, len(board) - 1): [(0, len(board) - 2), (1, len(board) - 1)],  # Top-right corner
            (len(board) - 1, 0): [(len(board) - 2, 0), (len(board) - 1, 1)],  # Bottom-left corner
            (len(board) - 1, len(board) - 1): [(len(board) - 2, len(board) - 1), (len(board) - 1, len(board) - 2)]
        }
        corner_to_xsquares = {
            (0, 0): [(1, 1)],  # Top-left corner
            (0, len(board) - 1): [(1, len(board) - 2)],  # Top-right corner
            (len(board) - 1, 0): [(len(board) - 2, 1)],  # Bottom-left corner
            (len(board) - 1, len(board) - 1): [(len(board) - 2, len(board) - 2)]  # Bottom-right corner
        }

        # Calculate corner score
        corner_score = sum(10 if board[r][c] == player else -10 if board[r][c] == opponent else 0 for r, c in corners)

        # Adjust penalties for C-squares and X-squares based on corner control
        c_square_score = 0
        x_square_score = 0

        for corner, c_sqs in corner_to_csquares.items():
            if board[corner[0]][corner[1]] == player:
                # No penalty if the player controls the corner
                continue
            for r, c in c_sqs:
                if board[r][c] == player:
                    c_square_score -= 5
                elif board[r][c] == opponent:
                    c_square_score += 5

        for corner, x_sqs in corner_to_xsquares.items():
            if board[corner[0]][corner[1]] == player:
                # No penalty if the player controls the corner
                continue
            for r, c in x_sqs:
                if board[r][c] == player:
                    x_square_score -= 3
                elif board[r][c] == opponent:
                    x_square_score += 3

        # Calculate the stability score of the corners (a small bonus for each adjacent square that the player controls)
        stability_score = 0
        for r, c in corners:
            if board[r][c] == player:
                # Check adjacent squares for stability (horizontally or vertically)
                if r > 0 and board[r - 1][c] == player: stability_score += 2
                if r < len(board) - 1 and board[r + 1][c] == player: stability_score += 2
                if c > 0 and board[r][c - 1] == player: stability_score += 2
                if c < len(board) - 1 and board[r][c + 1] == player: stability_score += 2

        # Combine all scores into one final corner control score
        return corner_score + c_square_score + x_square_score + stability_score

    def mobility_evaluation(self, board, player, opponent):
        """
        Evaluate the mobility score based on the number of legal moves for the player
        and the opponent, scaled to the total number of squares.
        """
        total_squares = board.shape[0] * board.shape[1]
        player_mobility = len(get_valid_moves(board, player))
        opponent_mobility = len(get_valid_moves(board, opponent))

        if player_mobility + opponent_mobility == 0:
            # Return a default value when both players have no valid moves
            return 0

        mobility_score = (player_mobility - opponent_mobility) / (player_mobility + opponent_mobility) * 100
        # scaled_score = (mobility_score / 100) * total_squares
        return mobility_score

    def piece_count_evaluation(self, board, player, opponent):
        """
        Evaluate the piece count (parity) score, based on the number of pieces for each player,
        scaled to the total number of squares on the board.
        """
        total_squares = board.shape[0] * board.shape[1]  # Total number of squares on the board
        player_score = np.sum(board == player)
        opponent_score = np.sum(board == opponent)

        # Compute the raw piece count difference
        piece_count_diff = player_score - opponent_score

        # Scale the difference to the total number of squares, and convert it to a percentage
        scaled_score = (piece_count_diff / total_squares) * 100

        return scaled_score

    def count_capture_evaluation(self, board, player, opponent):
        """
        Evaluate the board based on the number of discs flipped in the current state.
        The weight of this heuristic increases in the late game.
        """
        player_flipped = sum(count_capture(board, move, player) for move in get_valid_moves(board, player))
        opponent_flipped = sum(count_capture(board, move, opponent) for move in get_valid_moves(board, opponent))

        # Check to avoid division by zero
        if player_flipped + opponent_flipped == 0:
            return 0
        capture_score = (player_flipped - opponent_flipped) / (player_flipped + opponent_flipped) * 100

        return capture_score

    def edge_control_with_wedging(self, board, player, opponent):
        """
        Evaluate edge control while checking for the possibility of creating wedges.
        - Adds points for edge squares controlled by the player.
        - Subtracts points for edge squares controlled by the opponent.
        - Checks for potential wedge formations where the opponent could create a wedge.
        """
        edge_score = 0
        wedge_score = 0
        edge_squares = []

        # Collect all edge squares (excluding corners for simplicity)
        for i in range(len(board)):
            # Top and bottom edges (excluding corners)
            edge_squares.append((0, i))  # Top row
            edge_squares.append((len(board) - 1, i))  # Bottom row
            # Left and right edges (excluding corners)
            edge_squares.append((i, 0))  # Left column
            edge_squares.append((i, len(board) - 1))  # Right column

        # Evaluate control of edge squares and check for wedges
        for r, c in edge_squares:
            if board[r][c] == player:
                edge_score += 1
            elif board[r][c] == opponent:
                edge_score -= 1

        # Check for potential wedge formations along the edges (horizontal and vertical)
        def check_for_wedge_in_line(line):
            nonlocal wedge_score
            # Scan the line for two same-colored discs with empty squares between them
            for i in range(len(line) - 1):
                for j in range(i + 1, len(line)):
                    if line[i] == line[j] != 0:  # Same color discs
                        empty_count = sum(1 for k in range(i + 1, j) if line[k] == 0)  # Count empty squares
                        if empty_count % 2 == 1:  # Odd number of empty squares
                            # Wedge detected; opponent can potentially create a wedge
                            wedge_score += 1

        # Check horizontal and vertical lines for potential wedges
        for r in range(len(board)):
            check_for_wedge_in_line([board[r][c] for c in range(len(board))])  # Horizontal line
            check_for_wedge_in_line([board[c][r] for c in range(len(board))])  # Vertical line

        # Combine edge control score with wedge penalty
        return edge_score - wedge_score

    def initialize_static_weights(self, board_size):
        """
        Initialize the static weights matrix for the given board size.
        """
        weights = np.zeros((board_size, board_size))
        high_weight = 4  # For corners
        low_weight = -3  # For near-corner positions
        edge_weight = 2  # For edges not near corners
        neutral_weight = 1  # For other positions

        # Assign static weights
        for r in range(board_size):
            for c in range(board_size):
                if (r, c) in [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]:
                    weights[r, c] = high_weight  # Corners
                elif (r, c) in [(0, 1), (1, 0), (0, board_size - 2), (1, board_size - 1),
                                (board_size - 1, 1), (board_size - 2, 0), (board_size - 2, board_size - 1),
                                (board_size - 1, board_size - 2)]:
                    weights[r, c] = low_weight  # Near corners
                elif r == 0 or r == board_size - 1 or c == 0 or c == board_size - 1:
                    weights[r, c] = edge_weight  # Edges
                else:
                    weights[r, c] = neutral_weight  # Center/neutral positions

        return weights

    def calculate_stability(self, board, player, opponent):
        """
        Calculate the stability of each disk on the board.
        The function classifies each disk into one of four categories:
        1. Super Stable (Corners)
        2. Stable (Lines and near-corners)
        3. Semi-Stable (Partially surrounded but not fully)
        4. Unstable (Can be flipped)
        """
        stability_map = np.zeros_like(board, dtype=int)  # 0 for empty, 1-4 for stability categories

        for r in range(len(board)):
            for c in range(len(board[r])):
                if board[r][c] == player or board[r][c] == opponent:
                    stability_score = self.get_disk_stability(board, r, c, player, opponent)
                    stability_map[r][c] = stability_score

        return stability_map

    def get_disk_stability(self, board, r, c, player, opponent):
        """
        Return the stability level of a disk based on its position.
        3 - Super Stable (Corners)
        2 - Stable (Edges and near-corners)
        1 - Semi-Stable (Partially surrounded)
        0 - Unstable (Can be flipped)
        """
        # Check if the disk is on the corner
        corners = [(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)]
        if (r, c) in corners:
            return 3  # Super Stable (Corners)

        # Check if the disk is near a corner (adjacent to a corner) or
        # Check if the disk is on an edge but not a corner
        elif self.is_near_corner(board, r, c) or (r == 0 or r == len(board) - 1 or c == 0 or c == len(board) - 1):
            return 2  # Stable (Near corners or Edge)

        # Check if the disk is semi-stable (partially surrounded by same player)
        elif self.is_semi_stable(board, r, c, player):
            return 1  # Semi-Stable

        return 0  # Default to Unstable if no other conditions apply

    def is_near_corner(self, board, r, c):
        """
        Check if the square (r, c) is adjacent to a corner (near-corner).
        """
        corners = [(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)]
        for corner in corners:
            # Check if (r, c) is adjacent to any corner
            if abs(r - corner[0]) <= 1 and abs(c - corner[1]) <= 1:
                return True
        return False

    def is_semi_stable(self, board, r, c, player):
        """
        Check if a disk at (r, c) is semi-stable (partially surrounded by same player's discs).
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(board) and 0 <= nc < len(board[0]):
                if board[nr][nc] != player:  # If adjacent to an opponent's disk or empty
                    return True
        return False

    def calculate_dynamic_weights(self, board, player, opponent):
        static_weights = self.initialize_static_weights(board.shape[0])  # Initialize static weights
        stability_matrix = self.calculate_stability(board, player, opponent)  # Calculate stability matrix for player

        # Calculate dynamic weights (static weights * stability)
        dynamic_board = static_weights * stability_matrix

        player_score = np.sum((board == player) * dynamic_board)
        opponent_score = np.sum((board == opponent) * dynamic_board)
        score = 0
        if (player_score != 0 and opponent_score != 0):
            score = (player_score + opponent_score) / (abs(player_score) + abs(opponent_score)) * 100
        return score

    def determine_game_phase(self, board, player, opponent):
        """
        Determine the current phase of the game: early, middle, or late.
        Based on the number of pieces on the board.
        """
        total_pieces = np.sum(board == player) + np.sum(board == opponent)
        total_squares = board.shape[0] * board.shape[1]

        if total_pieces < total_squares * 0.25:
            return "early"
        elif total_pieces < total_squares * 0.75:
            return "middle"
        else:
            return "late"

    def evaluate_board(self, board, player, opponent):
        """
        Evaluate the board state with a heuristic that considers corner control,
        C-squares, X-squares, stability, mobility, and piece count, adjusted for game phase.
        """
        # Get the game phase (early, middle, or late game)
        phase = self.determine_game_phase(board, player, opponent)

        # Get the combined corner-related score
        corner_control_score = self.corner_control_evaluation(board, player, opponent)

        # Get the mobility score
        mobility_score = self.mobility_evaluation(board, player, opponent)

        # Get the piece count score
        piece_count_score = self.piece_count_evaluation(board, player, opponent)

        #Get the capture score
        count_capture_score = self.count_capture_evaluation(board, player, opponent)

        # Get the edge control score
        edge_control_score = self.edge_control_with_wedging(board, player, opponent)

        dynamic_board_score = self.calculate_dynamic_weights(board, player, opponent)

        # Adjust heuristic scores based on the game phase
        if phase == "early":
            # In the early game, prioritize corner control less
            corner_control_score *= 0.7  # low to moderate
            mobility_score *= 2.0  # very high
            piece_count_score *= 0.5  # low
            count_capture_score *= 0.5 #low
            edge_control_score *= 1.0  # moderate

        elif phase == "middle":
            # In the middle game, balance corner control and mobility
            corner_control_score *= 3  # 3 high
            mobility_score *= 1.5  # high
            piece_count_score *= 1.3  # moderate to high
            count_capture_score *= 1.3 # moderate to high
            edge_control_score *= 3  # 3 high

        else:
            # In the late game, focus on mobility less and piece count and corner more
            corner_control_score *= 7  # very high corner_control_score since its not on the same scale as the rest
            mobility_score *= 0.7  # moderate to low
            piece_count_score *= 2.0  # high
            count_capture_score *= 2.0 # moderate to high
            edge_control_score *= 7  # high

        # Combine all components into one final evaluation score
        evaluation_score = corner_control_score + mobility_score + piece_count_score + count_capture_score + edge_control_score + dynamic_board_score
        return evaluation_score



