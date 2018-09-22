## Generalized Tic-Tac-Toe
import board as b
import config as c
import math
# import unittest
import sys
import random
import time
import json
import os
import copy
from scipy import stats
import csv


class Game:
  ## MORE LIKE POOP METHODS ##

  def __init__(self, num_spaces, winning_paths, board={}, whos_turn=c.COMPUTER, other_player=c.HUMAN, reduced_opponent=True, noise=c.NOISE_HUMAN, heuristic="paths", exp=1, interaction=False, opponent=False, neighborhood=2, prune = False, potential='square'):
    ''' Initalizes a Game object '''
    self.num_spaces = num_spaces
    self.winning_paths = winning_paths
    self.node_count = 0
    # The board will be represented via a dictionary,
    # {space: [c.BLANK|c.HUMAN|c.COMPUTER]}
    if board:
      self.board = board
    else:
      self.board = b.Board(self.num_spaces, self.winning_paths)
    self.whos_turn = whos_turn
    self.other_player = other_player
    self.num_turns = 0
    self.last_socre = 0
    self.last_depth = 0
    # self.random_moves = random.randint(6,18)
    self.random_moves = 0
    self.spaces = []
    self.dist_between_spaces_on_path = 0.0
    # self.dist_between_spaces_undo = 0.0
    self.dist_between_spaces_reset = 0.0
    self.count_between_spaces_on_path = 0.0
    # self.count_between_spaces_undo = 0.0
    self.count_between_spaces_reset = 0.0
    self.on_same_win_path = 0.0
    self.node_count_x = 0
    self.move_matrix = []
    self.move_counter = 0.0
    self.prev_move_x = None
    self.prev_move_x_depth = 0
    self.max_depth = None
    self.max_moves = 35
    self.heuristic = heuristic
    self.prune = prune
    self.exp = exp
    self.interaction = interaction
    self.potential = potential
    self.opponent = opponent
    self.neighborhood_size = neighborhood
    self.reduced_opponent = reduced_opponent


  def make_move(self, space, player):
    """Puts <player>'s token on <space>
    If the game is over, returns the winner (c.HUMAN, c.COMPUTER, or, c.TIE).
    If the game is not over, it returns False.
    """
    ## First, change the state of the 'board' map
    self.num_turns += 1

    if space not in self.board.get_board():
      print space
      raise Exception("Space not in board")
    elif self.board.get_player(space) is not c.BLANK:
      raise Exception("Incorrect move")
    else:
      self.board.add_marker(space, player)

    winning_player = self.board.is_terminal() # False if there is no winning_player
    if winning_player:
      return winning_player
    else:
      return False

  def change_player(self):
    ''' Switches the current player '''
    if self.whos_turn == c.COMPUTER:
      self.whos_turn = c.HUMAN
      self.other_player = c.COMPUTER
    else:
      self.whos_turn = c.COMPUTER
      self.other_player = c.HUMAN

  def play_game(self, max_depth = False):
    ''' Returns the winner if there is one (or returns TIE)'''
    winning_player = False
    space = 100
    # while not winning_player:
    while self.num_turns<1:
      print self.num_turns
    # while not winning_player:
      self.node_count = 0
      self.display_game()
      self.max_depth = max_depth-self.num_turns
      space = self.get_next_move(max_depth)
      if space is None:
        print 'darn'
      print self.node_count
      winning_player = self.make_move(space, self.whos_turn)
      self.change_player()
    self.display_game()
    print self.node_count
    return (self.node_count, space)
    # return winning_player


  def play_random_game(self, num_markers = 8):
    ''' Returns the winner if there is one (or returns TIE)'''
    winning_player = False
    space = 100
    while not winning_player and self.num_turns<num_markers:
      self.display_game()
      space = self.get_random_move()
      if space == -1:
        print 'save board!'
        return
      winning_player = self.make_move(space, self.whos_turn)
      self.change_player()
    if space == -1:
      print 'save board!'
    return winning_player

  def get_random_move(self):
    board_copy = self.board.get_board_copy()
    spaces = board_copy.get_free_spaces()
    return random.choice(spaces)


  def get_next_move(self, max_depth=False):
    '''Gets the next move, by listening to the self.whos_turn class variable.
    If it is human's turn, it asks for input from the user.
    If it is the computer's turn, it performs a minimax search with alpha-beta pruning,
    with depth-limited search if requested.
    '''

    if self.whos_turn == c.HUMAN:
      board_copy = self.board.get_board_copy()
      if max_depth:
        self.max_depth = max_depth
        (score, space) = self.minimax_min_alphabeta_DL(c.NEG_INF, c.POS_INF, board_copy, max_depth)
      else:
        (score, space) = self.minimax_min_alphabeta(c.NEG_INF, c.POS_INF, board_copy)
      # if (c.CHECK_WIN and self.num_turns>c.MIN_MOVES):
      #   self.change_player()
      #   if self.check_win_at_depth(c.WIN_DEPTH):
      #     # self.make_move(space, self.whos_turn)
      #     self.save_board_to_file(space)
      #   self.change_player()
          # max_depth=25
      print 'score = '+str(score)
      print 'space = '+str(space)
      self.last_socre = score
      self.last_depth = max_depth
      # self.save_board_to_file(space, score)
      return space

    ######---------for human input------------------######
    #   try:
    #     space = int(raw_input("Enter a next move: "))
    #   except:
    #     print "Invalid input. Enter a positive integer."
    #     return self.get_next_move(max_depth)
    #
    #   if space in xrange(1, self.num_spaces+1) and self.board.get_player(space) is c.BLANK:
    #     return space
    #   else:
    #     print "Invalid input. Pick an available space."
    #     return self.get_next_move(max_depth)
    # elif self.board.is_empty() and max_depth > 4: # First move if large max_depth
    #   return self.first_move()
    ######---------for human input------------------######

    else:
      board_copy = self.board.get_board_copy()
      if max_depth:
        (score, space) = self.minimax_max_alphabeta_DL(c.NEG_INF, c.POS_INF, board_copy, max_depth)
      else:
        (score, space) = self.minimax_max_alphabeta(c.NEG_INF, c.POS_INF, board_copy)

      # self.save_board_to_file(space, score)
      self.last_socre = score
      self.last_depth = max_depth

          # max_depth = 25
        # self.change_player()
      print 'score = '+str(score)
      print 'space = '+str(space)

      return space

  def check_win_at_depth(self, depth):
    board_copy = self.board.get_board_copy()
    (score, space) = self.minimax_max_alphabeta_DL(c.NEG_INF, c.POS_INF, board_copy, depth)
    print 'in check win score = ' + str(score)
    if self.whos_turn == c.COMPUTER:
      return score >= c.WIN_SCORE-c.WIN_DEPTH
    else:
      return score == c.LOSE_SCORE



  def display_game(self):
    '''Presents a visual representation of the board. If the board size is square, it calls
    display_game_square() to represent a visual output. (see below). If the board size
    is not square, it simply informs you of the available spaces, the spaces you occupy, and
    the spaces your opponent (the computer) occupies.
    '''
    if self.num_spaces in [4, 9, 16, 25, 36,100]: # Reasonable sizes
      self.display_game_square()
    else:
      if self.whos_turn == c.HUMAN:
        print "Winning paths: ", self.board.winning_paths
        available_spaces = self.board.get_free_spaces()
        print "Available Spaces: ", available_spaces
        spaces_HUMAN_owns = self.board.get_HUMAN_spaces()
        print "Spaces you own: ", spaces_HUMAN_owns
        spaces_COMPUTER_owns = self.board.get_COMPUTER_spaces()
        print "Spaces COMPUTER owns: ", spaces_COMPUTER_owns

  def display_game_square(self):
    ''' If the board size is square, creates a visual representation of the board. '''
    num_rows = int(math.sqrt(self.num_spaces))
    if self.whos_turn == c.HUMAN:
      print "Your turn"
    else:
      print "COMPUTER's turn"
    for row in range(num_rows):
        row_str = ""
        for col in range(num_rows):
          space = num_rows*row + col + 1
          token = self.board.get_player(space)
          if token == c.BLANK:
            row_str += "_\t"
          elif token == c.HUMAN:
            row_str += "X\t"
          else:
            row_str += "O\t"
        print row_str

  def save_board_to_file(self, winning_move, score):
    ''' If the board size is square, creates a visual representation of the board. '''
    # timestamp = str(time.time())
    filename = c.PATH + "board_" + str(c.WIN_DEPTH) + "_" + str(game.random_moves) + "_" + c.TIME[:-3] + ".txt"
    print filename
    num_rows = int(math.sqrt(self.num_spaces))
    with open(filename, "a") as text_file:
      for row in range(num_rows):
        row_str = ""
        for col in range(num_rows):
          space = num_rows * row + col + 1
          token = self.board.get_player(space)
          if token == c.BLANK:
            row_str += "_\t"
          elif token == c.HUMAN:
            if space==winning_move:
              row_str += "Xw\t"
            else:
              row_str += "X\t"
          else:
            if space == winning_move:
              row_str += "Ow\t"
            else:
              row_str += "O\t"
        text_file.write(row_str)
        text_file.write("\n")
      text_file.write('next move = ' + str(winning_move) + '\n')
      text_file.write('score = ' + str(score) + '\n')



            ## ---------AI METHODS----------- ##

  def ai_dummy(self):
    """Returns the first of all available spaces (not used in prouduction, just testing)"""
    for space, player in self.board.iteritems():
      if player == c.BLANK:
        return space



  def random_play(self,num_markers):
    self.get_random_move();


  def minimax_max(self, board):
    ''' Minimax algorithm without alpha-beta pruning (not used in production)'''
    if board.is_terminal():
      return (board.obj(), None) # Space is none, because the board is terminal
    else:
      children = []
      for space in board.get_free_spaces():
        new_board = board.get_board_copy()
        new_board.add_marker(space, self.whos_turn)
        children.append((self.minimax_min(new_board)[0], space))
      return max(children)

  def minimax_min(self, board):
    ''' Minimax algorithm without alpha-beta pruning (not used in production)'''
    if board.is_terminal():
      return (board.obj(), None) # Space is none, because the board is terminal
    else:
      children = []
      for space in board.get_free_spaces():
        new_board = board.get_board_copy()
        new_board.add_marker(space, self.other_player)
        children.append((self.minimax_max(new_board)[0], space))
      return min(children)

  def minimax_max_alphabeta(self, alpha, beta, board):
    ''' Minimax algorithm with alpha-beta pruning. '''
    if board.is_terminal():
      return (board.obj(), None) # Space is none, because the board is terminal
    else:
      max_child = (c.NEG_INF, None)
      for space in board.get_free_spaces():
        new_board = board.get_board_copy()
        new_board.add_marker(space, self.whos_turn)
        score = self.minimax_min_alphabeta(alpha, beta, new_board)[0]
        if score > max_child[0]:
          max_child = (score, space)
        if max_child[0] >= beta:
          return max_child # Shouldn't help anyway
        alpha = max(alpha, score)
      return max_child

  def minimax_min_alphabeta(self, alpha, beta, board):
    ''' Minimax algorithm with alpha-beta pruning.'''
    if board.is_terminal():
      return (board.obj(), None)
    else:
      min_child = (c.POS_INF, None)
      for space in board.get_free_spaces():
        new_board = board.get_board_copy()
        new_board.add_marker(space, self.other_player)
        score = self.minimax_max_alphabeta(alpha, beta, new_board)[0]
        if score < min_child[0]:
          min_child = (score, space)
        if min_child[0] <= alpha:
          return min_child # Shouldn't help anyway
        beta = min(beta, score)
      return min_child

  def minimax_max_alphabeta_DL(self, alpha, beta, board, depth, prev_space_x = None):
    '''Minimax algorithm with alpha-beta pruning and depth-limited search. '''
    # or ((depth==self.max_depth-1) & (beta==c.LOSE_SCORE))
    if (board.is_terminal()) or (depth <= 0) or (self.node_count >= self.max_moves):
      # return (board.obj(c.WIN_DEPTH-depth), None) # Terminal (the space will be picked up via recursion)
      return (board.obj_interaction(c.COMPUTER,remaining_turns_x=math.ceil(depth/2.0),exp=self.exp, interaction=self.interaction), None)  # Terminal (the space will be picked up via recursion)
    else:
      self.node_count += 1
      max_child = (c.NEG_INF, None)
      # print 'depth = '+ str(depth) + ', free =' +str
      # for space in board.get_free_spaces_ranked_neighbors(self.whos_turn):
      # moves = board. get_free_spaces_ranked_neighbors(player=c.COMPUTER, remaining_turns_x=math.ceil(depth/2.0))
      # moves = board.get_free_spaces_ranked_paths(player=c.COMPUTER, remaining_turns_x=math.ceil(depth/2.0), depth=depth)
      moves = board.get_free_spaces_ranked_heuristic(player=c.COMPUTER,reduced_opponent=self.reduced_opponent, heuristic=self.heuristic, remaining_turns_x=math.ceil(depth/2.0), depth=depth, interaction=self.interaction, other_player=self.opponent, prune=self.prune, exp=self.exp, neighborhood=self.neighborhood_size, potential=self.potential)
      # top_moves = moves
      # print top_moves
      # for space in board.get_free_spaces_ranked_paths(player=c.COMPUTER, remaining_turns_x=math.ceil(depth/2.0), depth=depth)[:5]:
      for space in moves:
        # if space in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,31,41,51,61,71,30,40,50,60,70,80,90,100]:
        #   continue
        # print 'depth = ' + str(depth) + ', space = ' + str(space)
        new_board = board.get_board_copy()
        new_board.add_marker(space, c.COMPUTER)
        score = self.minimax_min_alphabeta_DL(alpha, beta, new_board, depth - 1, prev_space_x = prev_space_x)[0]
        if score > max_child[0]:
          max_child = (score, space)
        if max_child[0] >= beta:
          return max_child # Shouldn't help anyway

        # if max_child[1] is None:
        #     # print depth
        #     print 'max none_' +str(depth)
        # if max_child[0] >= c.LOSE_SCORE:
        #   return max_child # Shouldn't help anyway
        alpha = max(alpha, score)
      return max_child

  def minimax_min_alphabeta_DL(self, alpha, beta, board, depth, prev_space_x = None):
    '''Minimax algorithm with alpha-beta pruning and depth-limited search. '''
    # if ((depth==self.max_depth-2) & (beta==c.LOSE_SCORE)):
    #   print 'happened'
    if (board.is_terminal()) or (depth <= 0) or (self.node_count >= self.max_moves) or  ((depth==self.max_depth-2) & (beta==c.LOSE_SCORE)):
      # return (board.obj(c.WIN_DEPTH-depth), None) # Terminal (the space will be picked up via recursion)
      return (board.obj_interaction(c.COMPUTER,remaining_turns_x=math.ceil(depth/2.0),exp=self.exp, interaction=self.interaction),None)

    # if board.obj_interaction(c.HUMAN,remaining_turns_x=(math.ceil(depth/2.0)))==-20000000:
    #   # print 'cutting'
    #   return (board.obj_interaction(c.HUMAN,remaining_turns_x=(math.ceil(depth/2.0))),None)
    #

    else:
      self.node_count += 1
      min_child = (c.POS_INF, None)
      # for space in board.get_free_spaces_ranked():
      # for space in board.get_free_spaces_ranked_neighbors(self.whos_turn):
      # top_moves = board.get_free_spaces_ranked_paths(player=c.HUMAN)
      # if (self.whos_turn==c.COMPUTER):
      # moves = board. get_free_spaces_ranked_neighbors(player=c.COMPUTER, remaining_turns_x=math.ceil(depth/2.0))
      moves = board.get_free_spaces_ranked_heuristic(player=c.HUMAN,reduced_opponent=self.reduced_opponent, heuristic=self.heuristic, remaining_turns_x=math.ceil(depth/2.0), depth=depth, interaction=self.interaction, other_player=self.opponent, prune=self.prune, exp=self.exp, neighborhood=self.neighborhood_size, potential=self.potential)

      # top_moves = moves
      # print top_moves
      # for space in board.get_free_spaces_ranked_paths(player=c.HUMAN, remaining_turns_x=math.ceil(depth/2.0), depth=depth):
      for space in moves:

        # if depth==8:
        #   print space
        # if space==24:
        #   print '24' + '_' + str(depth)
        # if space==56:
        #   print '56' + '_' + str(depth)
        new_board = board.get_board_copy()
        new_board.add_marker(space, c.HUMAN)
        col = ((space - 1) % len(self.move_matrix))
        row = (float(space)/float(len(self.move_matrix)))-1
        row = math.ceil(row)

        if (new_board.check_possible_win(remaining_turns_O=math.ceil(depth/2.0))):

          if prev_space_x!=None:
            self.node_count_x+=1
            distance = board.get_manhattan_dist(space, prev_space_x)
            # on_same_path =

            if (depth==self.max_depth-2):
              self.dist_between_spaces_reset += distance
              self.count_between_spaces_reset += 1
            # elif (depth == self.prev_move_x_depth):
            #   # self.dist_between_spaces_undo += distance
            else:
              if space == prev_space_x:
                print 'oopsy'
                print space
              self.count_between_spaces_on_path += 1
              self.dist_between_spaces_on_path += distance
              if board.get_is_on_same_path(space, prev_space_x):
                self.on_same_win_path += 1

          # self.prev_move_x = space
          # self.prev_move_x_depth = depth
          # if (depth==self.max_depth) & (space ==4):
          #   print 'space  = 4'
          score = self.minimax_max_alphabeta_DL(alpha, beta, new_board, depth - 1, prev_space_x = space)[0]
          # if depth == 8:
          #   print score
          self.move_matrix[int(row)][int(col)] += 1
          self.move_counter += 1
          # if (depth==self.max_depth):
          #   print str(space) + ':' + str(score)
          if score < min_child[0]:
            min_child = (score, space)
          # if min_child[0] <= alpha:
          #   return min_child # Shouldn't help anyway
          if min_child[0] <= alpha:
            return min_child # Shouldn't help anyway
          # if min_child[1] is None:
          #   print 'min none_' +str(depth)
          beta = min(beta, score)


      return min_child



  def init_move_matrix(self, board_positions):
    self.move_matrix = copy.deepcopy(board_positions)
    for r in range(len(self.move_matrix)):
      for j in range(len(self.move_matrix[r])):
        if self.move_matrix[r][j] == 1:
          self.move_matrix[r][j] = 'X'
        elif self.move_matrix[r][j] == 2:
          self.move_matrix[r][j] = 'O'

  def first_move(self):
    '''Picks the space that appears in the most winning paths first.
    This is used for large board sizes.
    '''
    all_spaces = [space for path in self.winning_paths for space in path]
    counts = {}
    for space in all_spaces:
      if space in counts:
        counts[space] += 1
      else:
        counts[space] = 1
    return max([(count, space) for space, count in counts.iteritems()])[1]

'''
Load initial positions from json file
'''
def fill_board_from_file(json_file,game):
  json1_file = open(json_file)
  json1_str = json1_file.read()
  json1_data = json.loads(json1_str)
  board_positions = json1_data['position']
  game.whos_turn = int(json1_data['turn'])
  c.BOARD = copy.deepcopy(board_positions)
  # board_size = len(board_positions)*len(board_positions)
  space = 1
  for row in range(len(board_positions)):
    for col in range(len(board_positions)):

      mark = board_positions[row][col]

      if mark==1:
        game.board.add_marker(space,c.HUMAN)
      elif mark==2:
        game.board.add_marker(space,c.COMPUTER)

      space+=1
  game.board.prune_squares_density()
  game.init_move_matrix(board_positions)

  # game.move_matrix = copy.deepcopy(board_positions)
  # for r in range(len(game.move_matrix)):
  #   for c in range(len(game.move_matrix[r])):
  #     if game.move_matrix[r][c] == 1:
  #       game.move_matrix[r][c] = 'X'
  #     elif game.move_matrix[r][c] == 2:
  #       game.move_matrix[r][c] = 'O'

  # print board_positions
  c.WIN_MOVES = json1_data['winMove']
  c.SOLVED = json1_data['solved']
  c.SIM = json1_data['sim']
  return int(json1_data['win_depth'])





def start_game(file_path, configuration = None):
  '''Opens the file, processes the input, and initailizes the Game() object'''
  try:
    input_file = open(file_path ,'r')
  except:
    raise Exception("File not found!!! Make sure you didn't make a spelling error.")
  num_spaces = int(input_file.readline())
  winning_paths = []
  for line in input_file:
    path = map(int, line.split())
    winning_paths.append(path)
  if configuration == None:
    game = Game(num_spaces, winning_paths)
  else:
    game = Game(num_spaces, winning_paths, reduced_opponent=configuration['reduced_opponent'],heuristic=configuration['heuristic'], interaction=configuration['interaction'], neighborhood=configuration['neighborhood'], exp=configuration['exp'], opponent=configuration['opponent'], potential=configuration['potential'], prune=configuration['prune'])


  return game


def write_matrices_to_file(data_matrices, filename):
  with open(filename, 'w') as fp:
      json.dump(data_matrices, fp)

def get_game_configs(config_file):
  json1_file = open(config_file)
  json1_str = json1_file.read()
  json1_data = json.loads(json1_str)
  return json1_data['configs']


def get_heatmaps_alpha_beta():
  data_matrices = {}
  for filename in os.listdir("predefinedBoards/"):
    if filename.startswith("6"):
      file_path = "examples/board_6_4.txt"
      # continue
      if not(filename.startswith("6_easy")):
        continue

    else:
      # if filename.startswith("10by10_easy"):
      # if not(filename.startswith("10by10_medium")):
      #   continue
      file_path = "examples/board_10_5.txt"
      # continue



    game = start_game(file_path)
    print filename
    win_depth = fill_board_from_file("predefinedBoards/"+filename,game)
    print 'depth = '+ str(win_depth)
    game.play_game(win_depth)
    # print game.dist_between_spaces_on_path
    # print game.count_between_spaces_on_path
    move_matrix = copy.deepcopy(game.move_matrix)
    for r in range(0,len(move_matrix)):
        for j in range(0,len(move_matrix[r])):
            if (move_matrix[r][j]=='X'):
                move_matrix[r][j] = -0.00001
            elif (move_matrix[r][j]=='O'):
                move_matrix[r][j] = -0.00002
    normalized = True
    if normalized:
      for r in range(len(move_matrix)):
          for j in range(len(move_matrix[r])):
              if (move_matrix[r][j]!=-0.00001) & (move_matrix[r][j]!=-0.00002):
                  move_matrix[r][j] = move_matrix[r][j]/game.move_counter

    print move_matrix
    #compute entropy
    pk = []
    cell_counter = 0
    for i in range(len(move_matrix)):
        for j in range(len(move_matrix[i])):
            if ((move_matrix[i][j]!=-0.00001) & (move_matrix[i][j]!=-0.00002)):
                pk.append(move_matrix[i][j]/game.move_counter)
    # print pk
    ent = stats.entropy(pk)
    pk_uniform = []
    for i in range(len(pk)):
        pk_uniform.append(1.0/len(pk))

    # print pk_uniform

    # ent_norm = ent/free_cells
    ent_norm_max = ent/stats.entropy(pk_uniform)
    data_matrices[filename[:-5]] = (move_matrix, ent_norm_max)
    print ent_norm_max

  return data_matrices


def write_results(filename, results, header = None):

    res_file = open(filename, 'wb')
    res_writer = csv.writer(res_file, delimiter=',')
    if header!=None:
      res_writer.writerow(header)
    for res_line in results:
        res_writer.writerow(res_line)




if __name__ == "__main__":
    # data = get_heatmaps_alpha_beta()

    results = []
    header = ['board','heuristic_name','heuristic','layers','interaction','exponent','potential','neighborhood','opponent','numberOfNodes','answer','correct','exploredNodes']
    game_configs_file = "ab_config1.json"
    configs = get_game_configs(game_configs_file)
    for conf in configs:
      data_matrices = {}
      for filename in os.listdir("predefinedBoards/"):
        if filename.startswith("6"):
          file_path = "examples/board_6_4.txt"
          # continue
          # if not(filename.startswith("6_easy")):
          #   continue

        else:
          # if filename.startswith("10by10_easy"):
          # if (filename.startswith("10_medium")):
          #   continue
          file_path = "examples/board_10_5.txt"
          # continue

        game = start_game(file_path, conf)
        board_results = []
        board_results.append(filename[:-5])
        board_results.append(conf['name'])
        board_results.append(game.heuristic)
        board_results.append(game.prune)
        board_results.append(game.interaction)
        board_results.append(game.exp)
        board_results.append(game.potential)
        board_results.append(game.neighborhood_size)
        board_results.append(game.opponent)

        print filename
        win_depth = fill_board_from_file("predefinedBoards/"+filename,game)
        print 'depth = '+ str(win_depth)
        nodes, solution = game.play_game(win_depth)
        board_results.append(nodes)
        board_results.append(solution)
        if solution in c.WIN_MOVES:
          board_results.append(1)
        else:
          board_results.append(0)

        # print game.dist_between_spaces_on_path
        # print game.count_between_spaces_on_path
        move_matrix = copy.deepcopy(game.move_matrix)
        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[r])):
                if (move_matrix[r][j]=='X'):
                    move_matrix[r][j] = -0.00001
                elif (move_matrix[r][j]=='O'):
                    move_matrix[r][j] = -0.00002
        normalized = True
        if normalized:
          for r in range(len(move_matrix)):
              for j in range(len(move_matrix[r])):
                  if (move_matrix[r][j]!=-0.00001) & (move_matrix[r][j]!=-0.00002):
                      move_matrix[r][j] = move_matrix[r][j]/game.move_counter

        print move_matrix
        board_results.append(copy.deepcopy(move_matrix))
        data_matrices[filename] = move_matrix
        results.append(copy.deepcopy(board_results))


      # write_matrices_to_file(data_matrices, 'data_matrices/'+conf['name']+'_potentialBlock100000.json')

    for i in range(len(results)):
      print results[i]

    output_name = 'stats/' + game_configs_file[:-5] + '_cogsci_' + str(game.max_moves) + '.csv'
    write_results(output_name, results, header)
      #
      # print game.dist_between_spaces_on_path/game.count_between_spaces_on_path
      # print game.on_same_win_path
      # print game.on_same_win_path/game.count_between_spaces_on_path
      # print '----'
      # # print game.dist_between_spaces_reset
      # # print game.count_between_spaces_reset
      # print game.dist_between_spaces_reset/game.count_between_spaces_reset
      # print game.node_count_x

    # for filename in os.listdir("predefinedBoards/"):
    #   if filename.startswith("6"):
    #     file_path = "examples/board_6_4.txt"
    #     continue
    #   else:
    #     if (filename.startswith("10by10_hard")) | (filename.startswith("10by10_easy")):
    #       continue
    #     file_path = "examples/board_10_5.txt"
    #
    #
    #   game = start_game(file_path)
    #   print filename
    #   win_depth = fill_board_from_file("predefinedBoards/"+filename,game)
    #   print 'depth = '+ str(win_depth)
    #   game.play_game(win_depth)

    #
    # win_depth = fill_board_from_file("predefinedBoards/6by6.json",game)
    #
    # winning_player = game.play_game(max_depth)
    # # winning_player = game.play_random_game(9)
    # #
    # # filename = c.PATH + "board_" + str(c.WIN_DEPTH) + "_" + str(game.random_moves) + "_" + c.TIME[:-3] + ".txt"
    # # with open(filename, "a") as text_file:
    # if winning_player == c.HUMAN:
    #   print "HUMAN is the winner!"
    # elif winning_player == c.COMPUTER:
    #   print "COMPUTER is the winner!"
    # else:
    #   print "Tie game!"

    # for i in range(30):
    #   c.TIME = str(time.time())
    #
    #   game = start_game(file_path)
    #   # game.save_board_to_file(1)
    #
    #
    #
    #   winning_player = game.play_game(max_depth)
    #   # winning_player = game.play_random_game(9)
    #
    #   filename = c.PATH + "board_" + str(c.WIN_DEPTH) + "_" + str(game.random_moves) + "_" + c.TIME[:-3] + ".txt"
    #   with open(filename, "a") as text_file:
    #     if winning_player == c.HUMAN:
    #       print "HUMAN is the winner!"
    #       text_file.write("HUMAN is the winner!")
    #     elif winning_player == c.COMPUTER:
    #       print "COMPUTER is the winner!"
    #       text_file.write("COMPUTER is the winner!")
    #     else:
    #       print "Tie game!"
    #       text_file.write("Tie game!")






