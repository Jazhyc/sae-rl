from tictactoe import TicTacToeEnv, TicTacToeSAE
from agents import OptimalAgent, RandomAgent, LLMAgent, RLAgent, add_statistic
from move_checker import MoveChecker
from utils import display_board

from tqdm import tqdm
from constants import TEACHER, STUDENT, NUM_GAMES

from stable_baselines3.common.callbacks import CheckpointCallback

def baseline_experiment(agent, env, num_games=NUM_GAMES):
    for _ in tqdm(range(num_games)):
        regular_game(agent, env)
        
def saerl_learning(agent, env, num_steps):
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path="./output/checkpoints/",
        name_prefix="saerl_in_progress",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    agent.setup_model(env)
    agent.model.learn(total_timesteps=num_steps, progress_bar=True, callback=checkpoint_callback)
    agent.model.save("output/saerl_model")

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

def run_experiment(num_games=NUM_GAMES, get_context=False, use_rl_agent=False):
    
    move_checker = MoveChecker()
    teacher = OptimalAgent(TEACHER, move_checker)
    
    if use_rl_agent:
        student = RLAgent(STUDENT)
        env = TicTacToeSAE(move_checker, teacher)
        saerl_learning(student, env, num_games)
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