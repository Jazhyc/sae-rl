import os, time, random, re
from constants import RETRY_COUNT, SLEEP_TIME
import goodfire
import dotenv

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
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            api_format['system'],
            api_format['user']
        ],
        max_completion_tokens=25
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

def get_valid_move(agent, state, api_format, verbose=True):
    for _ in range(RETRY_COUNT):
        try:
            completion_text = get_completion(agent.model, api_format)
            move = extract_move(completion_text)
            
            # Check if move is already taken
            if state[move] in ['X', 'O']:
                raise ValueError("Move already taken")
            return move, completion_text
        
        except Exception as e: 
            add_statistic(agent.stats, 'invalid_move')
            if verbose:
                print(f"Error: {e}")
            time.sleep(SLEEP_TIME)
        
    add_statistic(agent.stats, 'fail_safe')
    return random.choice([i for i, spot in enumerate(state) if spot not in ['X', 'O']]), completion_text