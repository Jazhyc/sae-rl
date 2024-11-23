import random

def display_board(board, print_board=False):
        """
        Minimal representation of the board
        """
        
        text = ''
        for i in range(3):
            text += ' '.join(map(str, board[i*3:i*3+3]))
            
            if i < 2:
                text += '\n'
            
        if print_board:
            print(text)
            
        return text

class BaseAgent():
    
    def __init__(self, player):
        self.player = player
    
    def act(self, state):
        pass
    
    def learn(self, state, action, reward, next_state):
        pass
    
class RandomAgent(BaseAgent):
    
    def __init__(self, player):
        super().__init__(player)
        
    def act(self, state):
        return random.choice([i for i, spot in enumerate(state) if spot not in ['X', 'O']])
    
    def learn(self, state, action, reward, next_state):
        pass
    
class OptimalAgent(BaseAgent):
    
    def __init__(self, player, move_checker):
        super().__init__(player)
        self.move_checker = move_checker
        
    def act(self, state):
        optimal_moves = self.move_checker.get_optimal_moves(state, self.player)
        
        # Randomly select a move from the optimal moves
        return random.choice(optimal_moves)
    
    def learn(self, state, action, reward, next_state):
        pass
    
class LLMAgent(BaseAgent):
    
    def __init__(self):
        pass