from tictactoe import TicTacToeEnv
from agents import OptimalAgent, RandomAgent, LLMAgent, RLAgent, add_statistic
from move_checker import MoveChecker
from agents import display_board
import concurrent.futures

from tqdm import tqdm
from constants import TEACHER, STUDENT, NUM_GAMES, NUM_WORKERS

def single_thread(results, teacher, agent, move_checker, num_games=NUM_GAMES):
    for _ in tqdm(range(num_games)):
        winner = play_game(teacher, agent, move_checker)
        results[winner] += 1
        
    return results

# I did not want to deal with race conditions, so this is shelved for now
def multi_thread(results, teacher, agent, move_checker, num_games=NUM_GAMES):
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(play_game, teacher, agent, move_checker) for _ in range(num_games)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_games):
            winner = future.result()
            results[winner] += 1
    
    return results
    

def play_game(teacher, student, move_checker, verbose=False):
    env = TicTacToeEnv(move_checker, teacher)
    done = False
    
    # Check that both agents are different
    if teacher.player == student.player:
        raise ValueError("Both agents cannot be the same player")
    
    state, _ = env.reset()
    
    while not done:
        
        action = student.act(state)
        
        if verbose:
            display_board(state, print_board=True)
            print(f"Player {student.player} selects {action + 1}")
        
        new_state, reward, done, _, _ = env.step(action)
        
        add_statistic(student.stats, 'step')
        student.learn(state, action, reward, new_state)
        
        if done:
            break
        
        state = new_state
        
    return env.winner

def run_experiment(num_games=NUM_GAMES, get_context=False):
    move_checker = MoveChecker()
    teacher = OptimalAgent(TEACHER, move_checker)
    student = LLMAgent(STUDENT)
    
    # Determines whether to use the context or not
    # The context takes a long time to generate
    student.get_context = get_context
    
    results = {'X': 0, 'O': 0, 'Draw': 0}
    
    # If agent is RLAgent, always use single_thread as multiple actors are not currently supported
    results = single_thread(results, teacher, student, move_checker, num_games)

    
    print(results)
    print(student.stats)
    
    return student, results

if __name__ == '__main__':
    run_experiment()