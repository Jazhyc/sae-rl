from tictactoe import TicTacToeEnv
from agents import OptimalAgent, RandomAgent, LLMAgent
from move_checker import MoveChecker

from tqdm import tqdm

def play_game(agent1, agent2, move_checker):
    env = TicTacToeEnv(move_checker)
    done = False
    
    # X and O in order
    agents = [agent1, agent2]
    
    # Check that both agents are different
    if agent1.player == agent2.player:
        raise ValueError("Both agents cannot be the same player")
    
    while not done:
        for agent in agents:
            action = agent.act(env.board)
            _, reward_X, reward_O, done = env.step(action, agent.player)
            
            # Update agents
            agent1.learn(env.board, action, reward_X, env.board)
            agent2.learn(env.board, action, reward_O, env.board)
            
            if done:
                break
        
    return env.winner

if __name__ == '__main__':
    move_checker = MoveChecker()
    agent1 = OptimalAgent('O', move_checker)
    agent2 = RandomAgent('X')
    
    results = {'X': 0, 'O': 0, 'Draw': 0}
    for _ in tqdm(range(100)):
        winner = play_game(agent1, agent2, move_checker)
        results[winner] += 1
    
    print(results)