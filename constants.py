TEACHER = 'X'
STUDENT = 'O'

NUM_GAMES = 1
NUM_ACTIONS_SAE = 30 # This is not really a constant and depends on detected features
NUM_WORKERS = 4 # API seems to be rate limited to 100 requests per minute
NUM_TRAINING_STEPS = 10
NUM_ENVS = 8

RETRY_COUNT = 1
SLEEP_TIME = 8
STEERING_BOUND = 0.5
ERROR_PUNISHMENT = -10