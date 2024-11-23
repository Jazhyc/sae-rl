import random
from dotenv import load_dotenv
import goodfire

from copy import deepcopy

from utils import add_statistic, get_valid_move, get_top_features, get_base_api_format, display_board

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
    
class OptimalAgent(BaseAgent):
    
    def __init__(self, player, move_checker):
        super().__init__(player)
        self.move_checker = move_checker
        
    def act(self, state):
        optimal_moves = self.move_checker.get_optimal_moves(state, self.player)
        
        # Randomly select a move from the optimal moves
        return random.choice(optimal_moves)
    
class LLMAgent(BaseAgent):
    
    def __init__(self, player):
        self.player = player
        
        self.model = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")
        
        self.stats = {'top_features': {}}
        self.get_context = False
        
        self.api_template = get_base_api_format()
        
    def act(self, state):
        
        # Create copy of the template
        api_format = deepcopy(self.api_template)
        api_format['user']['content'] = self.api_template['user']['content'].format(board=display_board(state), player_type=self.player)
        
        move, response = get_valid_move(self, state, api_format)
        
        # Increase counter for this move
        add_statistic(self.stats, f'move_{move+1}')
        
        api_format['assistant']['content'] = response
        
        if self.get_context:
            get_top_features(self, state, move, api_format)
        
        return move
    
class RLAgent(LLMAgent):
    
    def __init__(self, player):
        super().__init__(player)
        self.model = goodfire.Variant("meta-llama/Meta-Llama-3-8B-RL")