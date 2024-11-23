class TicTacToeEnv():
    
    def __init__(self, move_checker=None):
        self.reset()
        
        self.reward_magnitude = 2
        self.reward_draw = 1
        self.reward_optimal_move = 0.1
        self.winner = None
        
        self.move_checker = move_checker
        
    def reset(self):
        self.board = [x for x in range(1, 10)]
        self.current_player = 'X'
        return self._obs()
    
    def _obs(self):
        """
        Returns the board state
        """
        return self.board
    
    def check_winner(self):
        """
        Checks if the state of the board is a winning state
        Returns the winner if there is one, None otherwise
        """
        # Check rows
        for i in range(3):
            if self.board[i*3] == self.board[i*3 + 1] == self.board[i*3 + 2]:
                return self.board[i*3]
        
        # Check columns
        for i in range(3):
            if self.board[i] == self.board[i + 3] == self.board[i + 6]:
                return self.board[i]
        
        # Check diagonals
        if self.board[0] == self.board[4] == self.board[8]:
            return self.board[0]
        
        if self.board[2] == self.board[4] == self.board[6]:
            return self.board[2]
        
        # Check for draw
        if all([x in ['X', 'O'] for x in self.board]):
            return 'Draw'
        
        return None
    
    def step(self, action, player):
        """
        Take a step in the environment by placing a mark on the board
        Args:
            action: int 0-8 indicating position on board
            player: int 1 or 2 indicating which player
        Returns:
            tuple: (observation, reward_p1, reward_p2, done)
        """
        # Validate action
        if not 0 <= action <= 8:
            raise ValueError("Invalid action. Must be between 0 and 8")
        if self.board[action] not in range(1, 10):
            raise ValueError("Invalid action. Position already taken")
        if player not in ['X', 'O']:
            raise ValueError("Invalid player. Must be X or O")

        # Place mark on board
        self.board[action] = player
        
        # Check for winner
        winner = self.check_winner()
        
        if winner:
            # Ideally, the reset will be called by the controller
            self.winner = winner
            done = True
        else:
            done = False
        
        # Calculate rewards
        reward_X = 0
        reward_O = 0
        if winner == 'X':
            reward_X = self.reward_magnitude
            reward_O = -self.reward_magnitude
        elif winner == 'O':
            reward_X = -self.reward_magnitude
            reward_O = self.reward_magnitude
        elif winner == 'Draw':
            reward_X = self.reward_draw
            reward_O = self.reward_draw
        else:
            if action in self.move_checker.get_optimal_moves(self.board, player):
                # Give the player who made the optimal move a small reward
                reward_X = self.reward_optimal_move if player == 'X' else 0
                reward_O = self.reward_optimal_move if player == 'O' else 0
            else:
                # Penalize the player who made the suboptimal move
                reward_X = -self.reward_optimal_move if player == 'X' else 0
                reward_O = -self.reward_optimal_move if player == 'O' else 0
        
        return self._obs(), reward_X, reward_O, done
        
        
        
            
if __name__ == '__main__':
    env = TicTacToeEnv()