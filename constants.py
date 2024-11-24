TEACHER = 'X'
STUDENT = 'O'

NUM_GAMES = 1
NUM_ACTIONS_SAE = 30 # This is not really a constant and depends on detected features
NUM_WORKERS = 4 # API seems to be rate limited to 100 requests per minute

RETRY_COUNT = 1
SLEEP_TIME = 1
STEERING_BOUND = 0.5
ERROR_PUNISHMENT_MAGNITUDE = -10