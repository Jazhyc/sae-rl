from tictactoe import TicTacToeEnv, TicTacToeSAE
from agents import OptimalAgent, RandomAgent, LLMAgent, RLAgent, add_statistic
from move_checker import MoveChecker
from utils import display_board
import concurrent.futures

from tqdm import tqdm
from constants import TEACHER, STUDENT, NUM_GAMES, NUM_WORKERS

def baseline_experiment(agent, env, num_games=NUM_GAMES):
    for _ in tqdm(range(num_games)):
        regular_game(agent, env)

def regular_game(student, env, verbose=False):
    
    state, _ = env.reset()
    done = False
    
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
        
def sae_rl_game(student, env, verbose=False):
    pass

def run_experiment(num_games=NUM_GAMES, get_context=False, use_rl_agent=False):
    
    move_checker = MoveChecker()
    teacher = OptimalAgent(TEACHER, move_checker)
    
    if use_rl_agent:
        student = RLAgent(STUDENT)
        env = TicTacToeSAE(move_checker, teacher)
    else:
        student = LLMAgent(STUDENT, get_context=True)
        env = TicTacToeEnv(move_checker, teacher)
        baseline_experiment(student, env, num_games)
    
    # Determines whether to use the context or not
    # The context takes a long time to generate
    student.get_context = get_context
    
    results = env.results
        
    # if use_rl_agent:
    #     return student, 
    
    return student, env, results

if __name__ == '__main__':
    run_experiment()