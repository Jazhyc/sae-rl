import os, time, random, re
import numpy as np
from constants import RETRY_COUNT, SLEEP_TIME
import goodfire
import dotenv
import tenacity

dotenv.load_dotenv()

client = goodfire.Client(
    os.getenv('GOODFIRE_API_KEY'),
)

def get_top_features(agent, state, move, api_format):
    context = client.features.inspect(
        [
            api_format['system'],
            api_format['user'],
            api_format['assistant']
        ],
        model=agent.model
    )
    
    for token in context.tokens:
        token_text = token._token.strip()
        
        # check if token is an int
        if token_text.isdigit():
            
            if move == int(token_text) - 1:             
                top_features = token.inspect()
                append_statistic(agent.stats['top_features'], tuple(state), top_features)
                break

def get_base_api_format():
    
    with open('prompts/system_prompt.txt', 'r') as f:
        system_prompt = f.read()
        
    with open('prompts/user_prompt.txt', 'r') as f:
        user_prompt = f.read()
        
    api_format = {
        'system' : {"role": "system", "content": system_prompt},
        'user' : {"role": "user", "content": user_prompt}, # Needs to be filled in
        'assistant' : {"role": "assistant", "content": None} # Placeholder for assistant response
    }
        
    return api_format
    
    

def add_statistic(stats, key):
    if key not in stats:
        stats[key] = 1
    else:
        stats[key] += 1
    return stats

def append_statistic(stats, key, value):
    if key not in stats:
        stats[key] = [value]
    else:
        stats[key].append(value)
    return stats

def get_completion(model, api_format):
    """Wrapper function to handle retry errors"""
    try:
        return _get_completion_with_retry(model, api_format)
    except tenacity.RetryError:
        print("Gave up after 3 retries (60 seconds) due to rate limiting")
        raise

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=2, min=15, max=60), retry=tenacity.retry_if_exception_type(goodfire.api.exceptions.RateLimitException))
def _get_completion_with_retry(model, api_format):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
            api_format['system'],
            api_format['user']
        ],
            max_completion_tokens=25
        )
    except Exception as e:
        if not isinstance(e, goodfire.api.exceptions.RateLimitException):
            print("Error getting completion", e)
        raise
    
    return completion.choices[0].message['content']

def extract_move(text, verbose=False):
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
        
    if verbose:
        print(text)
        
    raise ValueError("Could not extract move from text")

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
            print(text, end='\n')
            
        return text

def get_valid_move(agent, state, api_format, verbose=False, is_sae_rl=False):
    for _ in range(RETRY_COUNT):
        
        minor_punish = False
        
        try:
            completion_text = get_completion(agent.model, api_format)
            move = extract_move(completion_text)
            
            if verbose:
                print(state)
                print(move)
            
            # Check if move is already taken
            if state[move] in ['X', 'O']:
                minor_punish = True
                raise ValueError("Move already taken")
            return move, completion_text
        
        except Exception as e: 
            add_statistic(agent.stats, 'invalid_move')
            if verbose:
                print(f"Error: {e}")
            time.sleep(SLEEP_TIME)
        
    add_statistic(agent.stats, 'fail_safe')
    completion_text = "Error"
    
    if is_sae_rl:
        if minor_punish:
            agent.minor_punish = True
        else:
            agent.will_punish = True
    
    return random.choice([i for i, spot in enumerate(state) if spot not in ['X', 'O']]), completion_text

def convert_board_to_observation(board):
    """
    Converts the current board state into a format recognizable by Stable Baselines.
    
    Args:
        board (list): The current board state as a list of length 9.
                    Each element should be 'X', 'O', or None.
    
    Returns:
        np.ndarray: A numpy array of shape (9,) with values 0, 1, or 2.
    """
    observation = np.zeros(9, dtype=int)
    for i, cell in enumerate(board):
        if cell == 'X':
            observation[i] = 1
        elif cell == 'O':
            observation[i] = 2
        else:
            observation[i] = 0
    return observation