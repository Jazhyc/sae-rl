import random
from dotenv import load_dotenv
import goodfire
import torch.nn as nn

from copy import deepcopy
from utils import add_statistic, get_valid_move, get_top_features, get_base_api_format, display_board
from constants import MODEL
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from stable_baselines3.sac import MlpPolicy, SAC

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
    
    def __init__(self, player, get_context=False):
        self.player = player
        
        self.model = goodfire.Variant(MODEL)
        
        self.stats = {'top_features': {}}
        self.get_context = get_context
        
        self.api_template = get_base_api_format()
        
    def act(self, state):
        
        # Create copy of the template
        api_format = deepcopy(self.api_template)
        api_format['user']['content'] = self.api_template['user']['content'].format(board=display_board(state), player_type=self.player)
        
        # Skip punishment since this agent is not learning
        move, response = get_valid_move(self, state, api_format)
        
        # Increase counter for this move
        add_statistic(self.stats, f'move_{move+1}')
        add_statistic(self.stats, 'step') # We cannot access this in main loop I think
        
        api_format['assistant']['content'] = response
        
        if self.get_context:
            get_top_features(self, state, move, api_format)
        
        return move
    
class RLAgent(BaseAgent):
    
    def __init__(self, player, test_mode=False, use_checkpoint=False):
        super().__init__(player)
        self.stats = {}
        self.test_mode = test_mode
        self.use_checkpoint = use_checkpoint
        
    def setup_model(self, env):
        
        # Use sac algorithm
        if not self.test_mode:
            self.model = SAC(
                MlpPolicy,
                env,
                verbose=1,
                batch_size=256,
                tensorboard_log="output/tensorboard/",
                device='cpu',
                learning_rate=3e-4,
            )
        
        if self.test_mode or self.use_checkpoint:
            print("Loading trained model from disk")
            self.model = SAC.load("output/saerl_model_load_fix.zip", env=env)
            self.model.load_replay_buffer("output/saerl_replay_buffer_load_fix.pkl")
    
    # Used during testing
    def act(self, state):
        actions, _ = self.model.predict(state)
        return actions