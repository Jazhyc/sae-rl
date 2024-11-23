class MoveChecker:
    
    def __init__(self):
        self.hash_table = {
            "X": {},
            "O": {}
        }
        
    def is_optimal_move(self, board, action, player):
        board_key = tuple(board)
        if board_key in self.hash_table[player]:
            optimal_moves = self.hash_table[player][board_key]
        else:
            optimal_moves = self.get_optimal_moves(board, player)
            self.hash_table[player][board_key] = optimal_moves
        
        return action in optimal_moves

    def get_optimal_moves(self, board, player):
        
        board_key = tuple(board)
        if board_key in self.hash_table[player]:
            return self.hash_table[player][board_key]
        
        best_score = None
        optimal_moves = []
        for move in self.available_moves(board):
            board_copy = board.copy()
            board_copy[move] = player
            score = self.minimax(board_copy, self.swap_player(player), False)
            if best_score is None or score > best_score:
                best_score = score
                optimal_moves = [move]
            elif score == best_score:
                optimal_moves.append(move)
                
        self.hash_table[player][board_key] = optimal_moves
        return optimal_moves

    def minimax(self, board, player, is_maximizing):
        original_player = self.swap_player(player) if not is_maximizing else player

        def minimax_recursion(board, player, is_maximizing, alpha, beta):
            winner = self.check_winner(board)
            if winner == original_player:
                return 1
            elif winner == self.swap_player(original_player):
                return -1
            elif self.is_board_full(board):
                return 0
            
            if is_maximizing:
                max_eval = float('-inf')
                for move in self.available_moves(board):
                    board[move] = player
                    eval = minimax_recursion(board, self.swap_player(player), False, alpha, beta)
                    board[move] = None
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return max_eval
            else:
                min_eval = float('inf')
                for move in self.available_moves(board):
                    board[move] = player
                    eval = minimax_recursion(board, self.swap_player(player), True, alpha, beta)
                    board[move] = None
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval

        return minimax_recursion(board, player, is_maximizing, float('-inf'), float('inf'))

    def available_moves(self, board):
        return [i for i, spot in enumerate(board) if spot not in ['X', 'O']]

    def check_winner(self, board):
        winning_combinations = [
            [0,1,2], [3,4,5], [6,7,8],  # rows
            [0,3,6], [1,4,7], [2,5,8],  # columns
            [0,4,8], [2,4,6]            # diagonals
        ]
        for combo in winning_combinations:
            a, b, c = combo
            if board[a] == board[b] == board[c] and board[a] in ['X', 'O']:
                return board[a]
        return None

    def is_board_full(self, board):
        return all(spot in ['X', 'O'] for spot in board)

    def swap_player(self, player):
        return 'O' if player == 'X' else 'X'