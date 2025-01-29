from tictactoe import TicTacToeEnv, TicTacToeSAE
from agents import OptimalAgent, RandomAgent, LLMAgent, RLAgent, add_statistic
from move_checker import MoveChecker
from utils import display_board
import pickle

from tqdm import tqdm
from constants import TEACHER, STUDENT, NUM_GAMES, NUM_ENVS

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def baseline_experiment(agent, env, num_games=NUM_GAMES):
    for _ in tqdm(range(num_games)):
        regular_game(agent, env)
        
def saerl_learning(agent, env, num_steps):
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path="./output/checkpoints/exp10/",
        name_prefix="saerl_in_progress",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    agent.setup_model(env)
    
    if not agent.test_mode:
        agent.model.learn(total_timesteps=num_steps, callback=checkpoint_callback, progress_bar=True)
        agent.model.save("output/saerl_model_load_fix")
        agent.model.save_replay_buffer("output/saerl_replay_buffer_load_fix")
    else:
        for _ in tqdm(range(num_steps)):
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
        
        if done:
            break
        
        state = new_state

def run_experiment(num_games=NUM_GAMES, get_context=False, use_rl_agent=False, test_agent=False, use_checkpoint=False):
    
    move_checker = MoveChecker()
    teacher = OptimalAgent(TEACHER, move_checker)
    
    if use_rl_agent:
        student = RLAgent(STUDENT, test_mode=test_agent, use_checkpoint=use_checkpoint)

        # Create X parallel environments
        if NUM_ENVS == 1 or test_agent:
            env = TicTacToeSAE(move_checker, teacher, test_agent)
        else:
            env = SubprocVecEnv([
                lambda i=i: Monitor(TicTacToeSAE(move_checker, teacher, test_agent), filename=f"monitor_{i}.csv")
                for i in range(NUM_ENVS)  # Creates X parallel environments
            ])
        
        saerl_learning(student, env, num_games)
    else:
        student = LLMAgent(STUDENT, get_context=get_context)
        env = TicTacToeEnv(move_checker, teacher)
        baseline_experiment(student, env, num_games)
    
    # Determines whether to use the context or not
    # The context takes a long time to generate
    student.get_context = get_context
    
    results = None
    
    # Expect to only use single environment for testing
    if test_agent:
        results = env.results
    
    return student, env, results

if __name__ == '__main__':
    results_tuple = run_experiment(num_games=10000, get_context=False, use_rl_agent=True, use_checkpoint=True)