TEACHER = 'X'
STUDENT = 'O'

NUM_GAMES = 1
NUM_ACTIONS_SAE = 30 # This is not really a constant and depends on detected features
NUM_WORKERS = 4 # API seems to be rate limited to 100 requests per minute
NUM_ENVS = 12

RETRY_COUNT = 2
SLEEP_TIME = 3
STEERING_BOUND = 0.25
ERROR_PUNISHMENT = -10