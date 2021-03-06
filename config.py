BLANK = 0
HUMAN = 1
COMPUTER = 2
TIE = 3

WIN_DEPTH = 8
MIN_MOVES = 3
CHECK_WIN = False
WIN_MOVES = []
NUM_NODES = 0.0
NUM_NODES_ROLLOUTS = 0.0
SOLVED = 0.0
SIM = 0

OPPONENT_THREAT = 10
PLAYER_THREAT = 50

NOISE_COMPUTER = 0.0
NOISE_HUMAN = 0.0
WIN_SCORE = 100000
LOSE_SCORE = -WIN_SCORE
POS_INF = 1000000
NEG_INF = -POS_INF
TERM_SCORE = 100000000
TIME = '000'

BOARD = None

BOARDS_MINUS_1 = []

SCORES_DICT = None
SCORES_DICT_ALL_BOARDS = {'6_easy': {}, '6_hard': {}, '10_easy': {}, '10_hard': {}, '10_medium': {}}

HITS = 0
NO_HIT = 0
PATH = "boardsFilled/"

DIMENSION = 6
PATHS_DICT = []
NODE_LIMIT = 0