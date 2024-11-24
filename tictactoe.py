from agents import display_board
import gymnasium as gym
from constants import STUDENT, NUM_ACTIONS_SAE, STEERING_BOUND, ERROR_PUNISHMENT_MAGNITUDE
from utils import get_base_api_format, get_valid_move, convert_board_to_observation, add_statistic
from copy import deepcopy
import goodfire
import dotenv
import pickle

dotenv.load_dotenv()

class TicTacToeEnv(gym.Env):
    
    def __init__(self, move_checker, teacher):
        self.move_checker = move_checker
        self.teacher = teacher
        
        self.reward_magnitude = 2
        self.reward_draw = 1
        self.reward_optimal_move = 0.1
        self.winner = None
        
        # These would be used in regular RL
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(9,), dtype=int)
        
        self.reset()
        
    def reset(self, seed=None):
        self.board = [x for x in range(1, 10)]
        self._step(self.teacher.act(self.board), self.teacher.player)
        return self._obs(), {} # Return observation and empty info
    
    def render(self):
        display_board(self.board, print_board=True)
        
    def close(self):
        pass
    
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
    
    def _step(self, action, current_player):
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
        if current_player not in ['X', 'O']:
            raise ValueError("Invalid player. Must be X or O")

        # Place mark on board
        self.board[action] = current_player
        
        # Check for winner
        winner = self.check_winner()
        
        if winner:
            # Ideally, the reset will be called by the controller
            self.winner = winner
            done = True
        else:
            done = False
        
        # Calculate rewards
        reward = 0
        if winner == 'X':
            reward = -self.reward_magnitude
        elif winner == 'O':
            reward = self.reward_magnitude
        elif winner == 'Draw':
            reward = self.reward_draw
        else:
            if action in self.move_checker.get_optimal_moves(self.board, current_player):
                # Give the player who made the optimal move a small reward
                reward = self.reward_optimal_move if current_player == 'O' else 0
            else:
                # Penalize the player who made the suboptimal move
                reward = -self.reward_optimal_move if current_player == 'O' else 0
        
        # Immediately compute the teacher's move if the game is not done
        if not done and current_player != 'X':
            _, _, done, _, _ = self._step(self.teacher.act(self.board), self.teacher.player)
        
        # The player will always be O
        # Truncation is always False, and empty dict is returned for now
        return self._obs(), reward, done, False, {}
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self._step(action, STUDENT) # Player is always O
        return obs, reward, terminated, truncated, info
    
class TicTacToeSAE(TicTacToeEnv):
    
    def __init__(self, move_checker, teacher):
        super().__init__(move_checker, teacher)
        
        # Loads a Counter from Collections
        action_candidates = pickle.load(open('output/results.pkl', 'rb'))
        
        # Get the top NUM_ACTIONS_SAE actions
        self.action_features = [x[0] for x in action_candidates.most_common(NUM_ACTIONS_SAE)]
        true_action_length = len(self.action_features)
        
        # Each action corresponds to a continuous space bounded by STEERING_BOUND for each element in action_features
        self.action_space = gym.spaces.Box(low=-STEERING_BOUND, high=STEERING_BOUND, shape=(true_action_length,), dtype=float)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(9,), dtype=int)
        
        # Used when the agent makes an invalid move, set by get_valid_move
        self.will_punish = False
        
        self.model = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")
        self.api_template = get_base_api_format()
        
        self.stats = {}
        
    def reset(self, seed=None):
        self.board = [x for x in range(1, 10)]
        self._step(self.teacher.act(self.board), self.teacher.player)
        return convert_board_to_observation(self.board), {}
        
    def step(self, action):
        
        self.model.reset()
        
        # Zip the action features with the action values
        action_values = zip(self.action_features, action)
        
        for feature, value in action_values:
            self.model.set(feature, value)
            
        # Create copy of the template
        api_format = deepcopy(self.api_template)
        
        api_format['user']['content'] = self.api_template['user']['content'].format(board=display_board(self.board), player_type=STUDENT)
        
        move, _ = get_valid_move(self, self.board, api_format, is_sae_rl=True)
        add_statistic(self.stats, f"move_{move+1}")
        
        obs, reward, terminated, truncated, info = self._step(move, STUDENT)
        
        obs = convert_board_to_observation(obs)
        
        if self.will_punish:
            reward = -ERROR_PUNISHMENT_MAGNITUDE
            self.will_punish = False
        
        return obs, reward, terminated, truncated, info
         
if __name__ == '__main__':
    env = TicTacToeEnv()
    display_board(env.board, print_board=True)