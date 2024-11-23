import random
from dotenv import load_dotenv
import goodfire
import os
import re

RETRY_COUNT = 3

load_dotenv()

with open('prompts/system_prompt.txt', 'r') as f:
    system_prompt = f.read()
    
with open('prompts/user_prompt.txt', 'r') as f:
    user_prompt = f.read()
    
api_format = {
    'system' : {"role": "system", "content": system_prompt},
    'user' : {"role": "user", "content": None} # Placeholder for user input
}

client = goodfire.Client(
    os.getenv('GOODFIRE_API_KEY'),
)

def get_completion(model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            api_format['system'],
            api_format['user']
        ]
    )
    
    return completion.choices[0].message['content']

def extract_move(text):
    """
    It's possible that the model will output the move number in different formats.
    We should use regex to extract the number regardless of the format.
    """
        
    match = re.search(r'\d', text)
    if match:
        move = int(match.group())
        
        # Check if move is valid, moves are 1-indexed
        if 1 <= move <= 9:
            return move - 1
        else:
            raise ValueError("Invalid move")
    
    raise ValueError("Could not extract move from text")

def get_valid_move(agent, state):
    for _ in range(RETRY_COUNT):
        try:
            completion_text = get_completion(agent.model)
            move = extract_move(completion_text)
            
            # Check if move is already taken
            if state[move] in ['X', 'O']:
                raise ValueError("Invalid move")
            
            return move
        except:
            agent.error_move_count += 1
            return random.choice([i for i, spot in enumerate(state) if spot not in ['X', 'O']])

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
        self.error_move_count = 0
        
        self.model = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")
        
    def act(self, state):
        
        current_prompt = user_prompt.format(board=display_board(state), player_type=self.player)
        api_format['user']['content'] = current_prompt
        
        move = get_valid_move(self, state)
        
        return move
    
class RLAgent(LLMAgent):
    
    def __init__(self, player):
        super().__init__(player)
        self.model = goodfire.Variant("meta-llama/Meta-Llama-3-8B-RL")