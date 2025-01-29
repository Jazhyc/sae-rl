TEACHER = 'X'
STUDENT = 'O'

NUM_GAMES = 1
NUM_ACTIONS_SAE = 20 # This is not really a constant and depends on detected features
NUM_WORKERS = 4 # API seems to be rate limited to 100 requests per minute
NUM_ENVS = 6

RETRY_COUNT = 2
SLEEP_TIME = 3
STEERING_BOUND = 0.2
ERROR_PUNISHMENT = -10

MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
