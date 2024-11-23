from tictactoe import TicTacToeEnv
from agents import OptimalAgent, RandomAgent, LLMAgent, RLAgent
from move_checker import MoveChecker
from agents import display_board
import concurrent.futures

from tqdm import tqdm

NUM_GAMES = 10
NUM_WORKERS = 4 # API seems to be rate limited to 100 requests per minute

def single_thread(results, teacher, agent, move_checker):
    for _ in tqdm(range(NUM_GAMES)):
        winner = play_game(teacher, agent, move_checker)
        results[winner] += 1
        
    return results

def multi_thread(results, teacher, agent, move_checker):
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(play_game, teacher, agent, move_checker) for _ in range(NUM_GAMES)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=NUM_GAMES):
            winner = future.result()
            results[winner] += 1
    
    return results
    

def play_game(teacher, student, move_checker, verbose=False):
    env = TicTacToeEnv(move_checker)
    done = False
    
    # X and O in order
    agents = [teacher, student]
    
    # Check that both agents are different
    if teacher.player == student.player:
        raise ValueError("Both agents cannot be the same player")
    
    state = env.reset()
    
    while not done:
        for agent in agents:
            action = agent.act(state)
            
            if verbose:
                display_board(state, print_board=True)
                print(f"Player {agent.player} selects {action + 1}")
            
            new_state, reward_X, reward_O, done = env.step(action, agent.player)
            
            # Update agents, teacher does not learn
            if agent != teacher:
                student.learn(state, action, reward_O, new_state)
            
            if done:
                break
            
            state = new_state
        
    return env.winner

if __name__ == '__main__':
    move_checker = MoveChecker()
    teacher = OptimalAgent('X', move_checker)
    student = LLMAgent('O')
    
    results = {'X': 0, 'O': 0, 'Draw': 0}
    
    # If agent is RLAgent, always use single_thread as multiple actors are not currently supported
    if isinstance(student, RLAgent):
        print("Using single thread")
        results = single_thread(results, teacher, student, move_checker)
    else:
        print("Using multi thread, games will be played concurrently")
        results = multi_thread(results, teacher, student, move_checker)
    
    print(results)