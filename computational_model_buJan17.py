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

class Game:
  ## MORE LIKE POOP METHODS ##

  def __init__(self, num_spaces, winning_paths, board = {}, whos_turn = c.COMPUTER, other_player = c.HUMAN, noise = c.NOISE_HUMAN):
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
    # while self.num_turns<1:
    while not winning_player:
      self.node_count = 0
      self.display_game()
      space = self.get_next_move(max_depth)
      if space is None:
        print 'darn'
      print self.node_count
      winning_player = self.make_move(space, self.whos_turn)
      self.change_player()
    self.display_game()
    print self.node_count
    return winning_player


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

  def minimax_max_alphabeta_DL(self, alpha, beta, board, depth):
    '''Minimax algorithm with alpha-beta pruning and depth-limited search. '''
    if board.is_terminal() or depth <= 0:
      # return (board.obj(c.WIN_DEPTH-depth), None) # Terminal (the space will be picked up via recursion)
      return (board.obj_interaction(c.COMPUTER,remaining_turns_x=math.ceil(depth/2.0)),None)  # Terminal (the space will be picked up via recursion)
    else:
      self.node_count += 1
      max_child = (c.NEG_INF, None)
      # print 'depth = '+ str(depth) + ', free =' +str(board.get_free_spaces())
      # for space in board.get_free_spaces_ranked():
      # for space in board.get_free_spaces_ranked_neighbors(self.whos_turn):
      # top_moves = board.get_free_spaces_ranked_paths(player=c.COMPUTER)[:20]
      for space in board.get_free_spaces_ranked_paths(player=c.COMPUTER, remaining_turns_x=math.ceil(depth/2.0)):
      # for space in top_moves:
        # print 'depth = ' + str(depth) + ', space = ' + str(space)
        new_board = board.get_board_copy()
        new_board.add_marker(space, c.COMPUTER)
        score = self.minimax_min_alphabeta_DL(alpha, beta, new_board, depth - 1)[0]
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

  def minimax_min_alphabeta_DL(self, alpha, beta, board, depth):
    '''Minimax algorithm with alpha-beta pruning and depth-limited search. '''
    if board.is_terminal() or depth <= 0:
      # return (board.obj(c.WIN_DEPTH-depth), None) # Terminal (the space will be picked up via recursion)
      return (board.obj_interaction(c.HUMAN,remaining_turns_x=(math.ceil(depth/2.0))),None)
    else:
      self.node_count += 1
      min_child = (c.POS_INF, None)
      # for space in board.get_free_spaces_ranked():
      # for space in board.get_free_spaces_ranked_neighbors(self.whos_turn):
      # top_moves = board.get_free_spaces_ranked_paths(player=c.HUMAN)[:20]
      for space in board.get_free_spaces_ranked_paths(player=c.HUMAN, remaining_turns_x=math.ceil(depth/2.0)):
      # for space in top_moves:
        new_board = board.get_board_copy()
        new_board.add_marker(space, c.HUMAN)
        score = self.minimax_max_alphabeta_DL(alpha, beta, new_board, depth - 1)[0]
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

  print board_positions
  return int(json1_data['win_depth'])


def start_game(file_path):
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
  game = Game(num_spaces, winning_paths)


  return game
#
# class TicTacToeTest(unittest.TestCase):
#   ''' Test suite, only used for debugging. I wanted to experiment with
#   test-driven development.
#   '''
#   def setUp(self):
#     self.game = start_game('examples/tictactoe.txt')
#     self.board = board.Board(self.game.num_spaces, self.game.winning_paths)
#
#   def tearDown(self):
#     self.game = None
#
#   def test_board_properties(self):
#     assert self.board.get_free_spaces() == [1,2,3,4,5,6,7,8,9]
#     assert len(self.board.get_children(c.COMPUTER)) == 9
#     self.board.add_marker(1, c.COMPUTER)
#     assert self.board.get_free_spaces() == [2,3,4,5,6,7,8,9]
#     assert len(self.board.get_children(c.HUMAN)) == 8
#
#   def test_basic_game_play(self):
#     assert self.game.board.obj() == 0
#     assert self.game.board.obj() == 0
#
#     self.game.make_move(2, c.COMPUTER)
#     assert self.game.board.get_player(2) == c.COMPUTER
#     assert self.game.board.obj() == 10*2
#
#     self.game.make_move(3, c.HUMAN)
#     assert self.game.board.get_player(3) == c.HUMAN
#     assert self.game.board.obj() == -10
#
#     self.game.make_move(5, c.COMPUTER)
#
#     self.game.make_move(4, c.HUMAN)
#
#     self.game.make_move(8, c.COMPUTER)
#     assert self.game.board.obj() == c.WIN_SCORE


if __name__ == "__main__":

  # if len(sys.argv) != 2:
  #   print "Invalid input. Make sure to include a file path."
  #   print "Example: $ python gentictactoe.py <file_path>"
  # else:
  #   max_depth = raw_input("Do you want the AI to perform a depth-limited search? \
  #   \nIf yes, enter a positive integer representing the desired depth. \
  #   \nIf no, enter 'n': ")

    max_depth = 4
    # Ask user for maximum depth
    # if max_depth == 'n':
    #   max_depth = False
    # else:
    #   try:
    #     if int(max_depth) > 0:
    #       max_depth = int(max_depth)
    #     else:
    #       raise Exception("Please enter a positive integer for max_depth, or an 'n'")
    #   except:
    #     raise Exception("Please enter a positive integer for max_depth, or a 'n'")

    # Get file path, and start playing!
    # file_path = sys.argv[1]
    # file_path = "examples/board_6_4.txt"
    #
    # game = start_game(file_path)

    for filename in os.listdir("predefinedBoards/"):
      if filename.startswith("6"):
        file_path = "examples/board_6_4.txt"
        continue
      else:
        # if filename.startswith("10by10_easy"):
        if not(filename.startswith("10by10_easy")):
          continue
        file_path = "examples/board_10_5.txt"
        # continue


      game = start_game(file_path)
      print filename
      win_depth = fill_board_from_file("predefinedBoards/"+filename,game)
      print 'depth = '+ str(win_depth)
      game.play_game(win_depth)


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






