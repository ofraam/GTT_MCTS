import config as c
import math
import copy
from model import *
from utils import *
from replay import *
from board import *
from bisect import bisect_right


class Board:
  def __init__(self, num_spaces, winning_paths, board = None, pruned_spaces = None):
    ''' Initializes a Board object '''
    self.size = num_spaces
    self.winning_paths = winning_paths
    if board:
      self.board = board
    else:
      self.board = {}
      for space in xrange(1, num_spaces+1):
        self.board[space] = c.BLANK
    self.last_space = 0
    if pruned_spaces!=None:
      self.pruned_spaces_x = pruned_spaces
    else:
      self.pruned_spaces_x = []
    # self.prune_squares_density()
    # self.pruned_spaces_o = []

  def get_board(self):
    ''' Returns a dictionary representing the state of the board'''
    return self.board
    
    
  def get_board_copy(self):
    ''' Returns a copy of the Board object '''
    board_copy = self.board.copy()
    return Board(self.size, self.winning_paths, board_copy, pruned_spaces=self.pruned_spaces_x)
    
  def get_player(self, space):
    ''' Returns the player who occupies <space>. If no player occupies
    <space>, it returns BLANK. 
    '''
    return self.board[space]
    
  def is_empty(self):
    ''' Returns True if the board is empty, False if not. '''
    for space in self.board:
      if self.board[space] != c.BLANK:
        return False
    return True
    
  def add_marker(self, space, player):
    self.last_space = space
    ''' If <space> is unoccupied, add <player>'s marker to <space>.
    Else, throw an exception, indicating that either the human or the AI
    is making an invalid move. '''
    if self.get_player(space) is c.BLANK:
      self.board[space] = player
    else:
      raise Exception("Space has to be c.BLANK!!")

  def get_HUMAN_spaces(self):
    ''' Return a list of spaces that human occupies. '''
    list_of_spaces = []
    for space in self.board:
      if self.board[space] == c.HUMAN:
        list_of_spaces.append(space)
    return list_of_spaces
    
  def get_COMPUTER_spaces(self):
    ''' Return a list of spaces that COMPUTER occupies. '''
    list_of_spaces = []
    for space in self.board:
      if self.board[space] == c.COMPUTER:
        list_of_spaces.append(space)
    return list_of_spaces
    
  def get_free_spaces(self):
    ''' Return a list of unoccupied spaces. '''
    list_of_spaces = []
    for space in self.board:
      if self.board[space] == c.BLANK:
        list_of_spaces.append(space)
    return list_of_spaces   

  def get_manhattan_dist(self,space1, space2):
    dimension = math.sqrt(self.size)
    row1 = float(space1)/float(dimension)
    row1 = math.ceil(row1)
    row2 = float(space2)/float(dimension)
    row2 = math.ceil(row2)

    col1 = ((space1 - 1) % dimension) + 1
    col2 = ((space2 - 1) % dimension) + 1
    dist = abs(row1-row2) + abs(col1-col2)
    return dist

  def get_path_dist(self,space1, space2):
    if self.get_is_on_same_path(space1, space2):
      return 0

    else:
      min_manhattan_dist = 1000
      for space in self.board:
        if self.board[space] == 0:
          if self.get_is_on_same_path(space,space2):
            new_dist = self.get_manhattan_dist(space1,space)
            if new_dist == 1:
              return new_dist
            if new_dist < min_manhattan_dist:
              min_manhattan_dist = new_dist
      return min_manhattan_dist

  def get_is_on_same_path(self,space1, space2):
    if self.get_manhattan_dist(space1, space2) > ((len(self.winning_paths[0])-1)*2):
      return False
    for path in self.winning_paths:
      if space1 in path:
        len_path = len(path)
        c.COMPUTER_count, c.HUMAN_count = 0, 0
        free_on_path = []
        for space in path:
          if self.board[space] == c.COMPUTER:
            c.COMPUTER_count += 1
          elif self.board[space] == c.HUMAN:
            c.HUMAN_count += 1
          else:
            free_on_path.append(space)
        if c.COMPUTER_count == 0:
          if space2 in path:
            return True
    return False

  def compute_avg_distance(self, space, list_of_occupied):
    sum_distances = 0.0
    for occupied_space in list_of_occupied:
      sum_distances = sum_distances + self.get_manhattan_dist(space,occupied_space)
    avg_dist = 0
    if len(list_of_occupied)>0:
      avg_dist = sum_distances/float(len(list_of_occupied))
    return avg_dist


  def prune_squares_density(self, neighborhood_size=2, threshold=0.2):
    spaces = self.get_free_spaces()
    scores = []
    max_score = -100
    for space in spaces:
      score = self.compute_square_score_density(space, c.HUMAN, neighborhood_size)
      scores.append(score)
      if score > max_score:
        max_score = score

    for i in range(len(spaces)):
      score = scores[i]
      # print score
      # print max_score
      # print threshold*max_score
      if score < threshold*max_score:
        self.pruned_spaces_x.append(spaces[i])
      # else:
      #   print 'not pruned'
    # print self.pruned_spaces_x


  def compute_square_score_density(self, space, player, neighborhood_size, remaining_turns_x=None):
    # print space
    if (player == c.HUMAN) & (remaining_turns_x != None):
      if not(self.check_possible_win(remaining_turns_x)):
        return -20000000

    dimension = math.sqrt(self.size)
    col = ((space - 1) % dimension)
    row = (float(space)/float(dimension))-1
    row = math.ceil(row)
    # print row
    # print col
    x_count = 0.0
    density_score = 0.0
    for i in range(-1*neighborhood_size,neighborhood_size+1):
      for j in range(-1*neighborhood_size,neighborhood_size+1):
        if (i != 0) | (j != 0):
          curr_row = row + i
          curr_col = col + j
          space_neighbor = curr_row*dimension + (curr_col+1)
          if (curr_row < dimension) & (curr_row >= 0) & (curr_col < dimension) & (curr_col >= 0):
            # print r
            # print c
            if self.board[space_neighbor] == player:
              x_count += 1.0
              density_score += 1.0/(8*max(abs(i), abs(j)))

    return density_score


  def neighbors_score(self,space, neighborhood_size, player):
    nighbor_count = 0.0
    density_score = 0.0
    dimension = math.sqrt(self.size)

    col = ((space - 1) % dimension)
    row = (float(space)/float(dimension))-1
    row = math.ceil(row)
    for i in range(-1*neighborhood_size,neighborhood_size+1):
      for j in range(-1*neighborhood_size,neighborhood_size+1):
        if (i != 0) | (j != 0):
          r = row + i

          c = col + j
          space_neighbor = r*dimension + (c+1)
          if (space_neighbor>0) & (space_neighbor<dimension*dimension+1):
            # print r
            # print c
            if self.board[space_neighbor] == player:
              nighbor_count += 1.0
              density_score += 1.0/(8*max(abs(i), abs(j)))
    return density_score


  def get_free_spaces_ranked_heuristic_with_scores(self, player, remaining_turns_x = None, depth = 0, heuristic = 'paths', interaction = True, exp = 2, neighborhood = 2, other_player=True, potential='full', prune = False, reduced_opponent = True):
    ''' Return a list of unoccupied spaces. '''
    list_of_spaces = []
    list_of_occupied = []
    list_of_spaces_with_dist = []
    list_of_spaces_with_dist2 = []
    for space in self.board:
      if self.board[space] == c.BLANK:
        list_of_spaces.append(space)
      else:
        list_of_occupied.append(space)
    if player is None:
      print 'problem'

    forced_move = self.win_or_forced_move(player)
    if forced_move:
      return forced_move

    for free_space in list_of_spaces:
      # if (free_space not in self.pruned_spaces_x):
      #   print 'pruning'
      if (free_space not in self.pruned_spaces_x) | (player == c.COMPUTER) | (not(prune)):
        if heuristic == 'paths':
          # list_of_spaces_with_dist.append((free_space,self.compute_square_score_paths_clean2(free_space, player=player,remaining_turns_x=remaining_turns_x,depth=depth, exp=exp, interaction=interaction, other_player=other_player, potential=potential)))
          list_of_spaces_with_dist.append((free_space,self.compute_square_score_pathsJan31(free_space, player=player,reduced_opponent=reduced_opponent,remaining_turns_x=remaining_turns_x,depth=depth, exp=exp, interaction=interaction, other_player=other_player, potential=potential)))
          # list_of_spaces_with_dist2.append((free_space,self.compute_square_score_paths_potential(free_space, player=player,remaining_turns_x=remaining_turns_x,depth=depth, exp=exp, interaction=interaction, other_player=other_player, potential='full')))
        elif heuristic == 'block':
          # list_of_spaces_with_dist.append((free_space,self.compute_square_score_block_clean(free_space, player=player,remaining_turns_x=remaining_turns_x,depth=depth, exp=exp, interaction=interaction, other_player=other_player, potential=potential)))
          list_of_spaces_with_dist.append((free_space,self.compute_square_score_blockJan31(free_space, player=player,reduced_opponent=reduced_opponent,remaining_turns_x=remaining_turns_x,depth=depth, exp=exp, interaction=interaction, other_player=other_player, potential=potential)))

        else:
          list_of_spaces_with_dist.append((free_space,self.compute_square_score_density(free_space, player,remaining_turns_x=remaining_turns_x, neighborhood_size=neighborhood)))

    # if player==c.COMPUTER:
    sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1], reverse=True)

    return sorted_list

  def get_free_spaces_ranked_heuristic_model(self, player, remaining_turns_x = None, depth = 0, heuristic = 'paths', interaction = True, exp = 2, neighborhood = 2, other_player=True, potential='full', prune = False, reduced_opponent = True, shutter=False, shutter_size=2, k=3, prev_move_x=None, stochastic_order=True, noise=0):
    ''' Return a list of unoccupied spaces. '''
    list_of_spaces = []
    list_of_occupied = []
    list_of_spaces_with_dist = []
    for space in self.board:
      if self.board[space] == c.BLANK:
        list_of_spaces.append(space)
      else:
        list_of_occupied.append(space)
    if player is None:
      print 'problem'

    board_matrix = convert_ab_board_to_matrix(self.board)
    player_type = 'X'
    if player == 2:
      player_type = 'O'
    # probs = compute_transition_probs_heuristic('blocking', board_matrix, player_type, normalized=True)
    prev_move_x_row_col = None
    if prev_move_x != None:
      prev_move_x_row_col = convert_position_to_row_col(prev_move_x,math.sqrt(self.size))

    # probs, nodes_computed, winning_moves = compute_paths_scores_for_matrix(board_matrix, player=player_type, normalized=True, o_weight=0.5, exp=2, block=False, interaction=False, board_obj=self, shutter=shutter, shutter_size=shutter_size, prev_x_move=prev_move_x_row_col, noise=noise)

    # probs, nodes_computed, winning_moves = compute_paths_scores_for_matrix(board_matrix, player=player_type, normalized=True, o_weight=0.5, exp=2, block=True, interaction=True, board_obj=self, shutter=shutter, shutter_size=shutter_size, prev_x_move=prev_move_x_row_col, noise=noise)
    probs, nodes_computed, winning_moves = compute_paths_scores_for_matrix(board_matrix, player=player_type, normalized=True, o_weight=0.5, exp=2, block=False, interaction=True, board_obj=self, shutter=shutter, shutter_size=shutter_size, prev_x_move=prev_move_x_row_col, noise=noise)


    # note to self: now done within the heuristic computation
    # if (shutter) & (prev_move_x != None):
    #   pruned_list_of_moves = []
    #   for space in list_of_spaces:
    #     dist = self.get_path_dist(space, prev_move_x)
    #     list_of_spaces_with_dist.append((space, dist))
    #   sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1], reverse=False)
    #   list_of_spaces = []
    #   for sp in sorted_list:
    #     list_of_spaces.append(sp[0])
    #   list_of_spaces = list_of_spaces[ :k]

    list_of_spaces_with_dist = []
    for space in list_of_spaces:
      row, col = convert_position(space, math.sqrt(self.size))
      list_of_spaces_with_dist.append((space, probs[row][col]))
    sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1], reverse=True)
    ranked_scores = []  # for mcts
    if stochastic_order:
      ranked_list = self.reorder_list(sorted_list)
    else:
      ranked_list = []
      for sp in sorted_list:
          ranked_list.append(sp[0])
          ranked_scores.append(sp[1])
    missed_win = -1
    if len(ranked_list) == 0:
      return [],[],False
    top_move = ranked_list[0]
    top_move_row_col = convert_position_to_row_col(top_move, math.sqrt(self.size))
    if len(winning_moves) > 0:
      missed_win = 1.0
      for win_move in winning_moves:
        if (win_move[0] == top_move_row_col[0]) & (win_move[1] == top_move_row_col[1]):
          missed_win = 0.0
          break
    if len(ranked_list) == 0:
      print 'problem'
    # return ranked_list[ :k], nodes_computed, missed_win
    # for mcts
    return ranked_list[ :k], ranked_scores, missed_win


  def weighted_choice(self,weights):
      rnd = random.random() * sum(weights)
      for i, w in enumerate(weights):
          rnd -= w
          if rnd < 0:
              return i

  def reorder_list(self, move_list):
    a = [i[0] for i in move_list]
    w = [i[1]+0.000001 for i in move_list]
    # for item_weight in w:
    #   if item_weight == 0:
    #     item_weight += 0.0001
    w = list(w) # make a copy of w
    if len(a) != len(w):
        print("weighted_shuffle: Lenghts of lists don't match.")
        return

    r = [0]*len(a) # contains the random shuffle
    for i in range(len(a)):
        j = self.weighted_choice(w)
        if (i is None) | (j is None):
          print 'problem'
        r[i]=a[j]
        w[j] = 0
    return r
    # r = np.empty_like(a)
    # cumWeights = np.cumsum(w)
    # for i in range(len(a)):
    #      rnd = random.random() * cumWeights[-1]
    #      j = bisect_right(cumWeights,rnd)
    #      #j = np.searchsorted(cumWeights, rnd, side='right')
    #      r[i]=a[j]
    #      cumWeights[j:] -= w[j]
    return r
    # move_list.sort(key = lambda item: random.expovariate(lambd=0.5) * item[1], reverse=True)

    # reordered_list = []
    # added_indices = []
    #
    # while len(reordered_list) < len(move_list):
    #   rand = random.random()
    #   sum_probs = 0.0
    #   i = 0
    #   while rand > sum_probs:
    #     sum_probs += move_list[i][1]
    #     i += 1
    #   if i == len(move_list):
    #     i -= 1
    #
    #   # if i not in added_indices:
    #   added_indices.append(i)
    #   reordered_list.append(copy.deepcopy(move_list[i]))
    #   move_list
    # return reordered_list

  def get_free_spaces_ranked_heuristic(self, player, remaining_turns_x = None, depth = 0, heuristic = 'paths', interaction = True, exp = 2, neighborhood = 2, other_player=True, potential='full', prune = False, reduced_opponent = True, shutter=False, k=10):
    ''' Return a list of unoccupied spaces. '''
    list_of_spaces = []
    list_of_occupied = []
    list_of_spaces_with_dist = []
    list_of_spaces_with_dist2 = []
    for space in self.board:
      if self.board[space] == c.BLANK:
        list_of_spaces.append(space)
      else:
        list_of_occupied.append(space)
    if player is None:
      print 'problem'


    # TODO: bring back (6/8/18, removed for shutter)
    # forced_move = self.win_or_forced_move(player)
    # if forced_move:
    #   return forced_move

    for free_space in list_of_spaces:
      # if (free_space not in self.pruned_spaces_x):
      #   print 'pruning'
      if (free_space not in self.pruned_spaces_x) | (player == c.COMPUTER) | (not(prune)):
        if heuristic == 'paths':
          # list_of_spaces_with_dist.append((free_space,self.compute_square_score_paths_clean2(free_space, player=player,remaining_turns_x=remaining_turns_x,depth=depth, exp=exp, interaction=interaction, other_player=other_player, potential=potential)))
          list_of_spaces_with_dist.append((free_space,self.compute_square_score_pathsJan31(free_space, player=player,reduced_opponent=reduced_opponent,remaining_turns_x=remaining_turns_x,depth=depth, exp=exp, interaction=interaction, other_player=other_player, potential=potential)))
          # list_of_spaces_with_dist2.append((free_space,self.compute_square_score_paths_potential(free_space, player=player,remaining_turns_x=remaining_turns_x,depth=depth, exp=exp, interaction=interaction, other_player=other_player, potential='full')))
        elif heuristic == 'block':
          # list_of_spaces_with_dist.append((free_space,self.compute_square_score_block_clean(free_space, player=player,remaining_turns_x=remaining_turns_x,depth=depth, exp=exp, interaction=interaction, other_player=other_player, potential=potential)))
          list_of_spaces_with_dist.append((free_space,self.compute_square_score_blockJan31(free_space, player=player,reduced_opponent=reduced_opponent,remaining_turns_x=remaining_turns_x,depth=depth, exp=exp, interaction=interaction, other_player=other_player, potential=potential)))
        else:
          list_of_spaces_with_dist.append((free_space,self.compute_square_score_density(free_space, player,remaining_turns_x=remaining_turns_x, neighborhood_size=neighborhood)))

    # if player==c.COMPUTER:


    sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1], reverse=True)
    # sorted_list2 = sorted(list_of_spaces_with_dist2, key=lambda x: x[1], reverse=True)
    # for i in range(len(sorted_list)):
    #   if sorted_list[i][0] != sorted_list2[i][0]:
    #     print 'boo'
    # else:
    #   sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1], reverse=False)
    ranked_list = []
    for sp in sorted_list:
      if (sp[1] > -20000000):
        ranked_list.append(sp[0])
      else:
        return ranked_list

    # save matrix of scores:
    # score_matrix = []
    # dimension = int(math.sqrt(len(self.board)))
    # for row in range(dimension):
    #   score_matrix.append([])
    #   for col in range(dimension):
    #     if self.board[row*dimension + (col+1)] == 1:
    #       score_matrix[row].append('X')
    #     elif self.board[row*dimension + (col+1)] == 2:
    #       score_matrix[row].append('O')
    #     else:
    #       score_matrix[row].append(0)
    # for i in range(len(list_of_spaces_with_dist)):
    #   position = list_of_spaces_with_dist[i][0]
    #   col = ((position - 1) % dimension)
    #   row = (float(position)/float(dimension))-1
    #   row = int(abs(math.ceil(row)))
    #   score_matrix[row][col] = list_of_spaces_with_dist[i][1]
    #
    # print score_matrix
    return ranked_list


  def get_free_spaces_ranked_neighbors(self, player, remaining_turns_x = None, depth = 0, neighborhood_size = 2):
    ''' Return a list of unoccupied spaces. '''
    list_of_spaces = []
    list_of_occupied = []
    list_of_spaces_with_dist = []
    for space in self.board:
      if self.board[space] == c.BLANK:
        list_of_spaces.append(space)
      else:
        list_of_occupied.append(space)

    for free_space in list_of_spaces:
      list_of_spaces_with_dist.append((free_space,self.compute_square_score_density(free_space, player,remaining_turns_x=remaining_turns_x, neighborhood_size=2)))

    # if player==c.COMPUTER:
    sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1], reverse=True)
    # else:
    #   sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1], reverse=False)
    ranked_list = []
    for sp in sorted_list:
      if (sp[1] > -20000000):
        ranked_list.append(sp[0])
      else:
        return ranked_list

    return ranked_list


  def get_free_spaces_ranked_paths(self, player, remaining_turns_x = None, depth = 0):
    ''' Return a list of unoccupied spaces. '''
    list_of_spaces = []
    list_of_occupied = []
    list_of_spaces_with_dist = []
    for space in self.board:
      if self.board[space] == c.BLANK:
        list_of_spaces.append(space)
      else:
        list_of_occupied.append(space)
    if player is None:
      print 'problem'
    for free_space in list_of_spaces:
      list_of_spaces_with_dist.append((free_space,self.compute_square_score_paths(free_space, player=player,remaining_turns_x=remaining_turns_x,depth=depth, exp=2, interaction=True, other_player=False, potential="square")))

    # if player==c.COMPUTER:
    sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1], reverse=True)
    # else:
    #   sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1], reverse=False)
    ranked_list = []
    for sp in sorted_list:
      if (sp[1] > -20000000):
        ranked_list.append(sp[0])
      else:
        return ranked_list

    return ranked_list



  def get_free_spaces_ranked(self):
    ''' Return a list of unoccupied spaces. '''
    list_of_spaces = []
    list_of_occupied = []
    list_of_spaces_with_dist = []
    for space in self.board:
      if self.board[space] == c.BLANK:
        list_of_spaces.append(space)
      else:
        list_of_occupied.append(space)
    for free_space in list_of_spaces:
      list_of_spaces_with_dist.append((free_space,self.compute_avg_distance(free_space,list_of_occupied)))

    sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1])
    ranked_list  = [x[0] for x in sorted_list]
    return ranked_list

  def get_children(self, player):
    ''' Return a list of Board objects that are possible options for <player>'''
    free_spaces = self.get_free_spaces()
    children = []
    for space in free_spaces:
      board_copy = self.board.copy()
      board_copy[space] = player
      children.append(board_copy)
      
    return children

  def get_outcome(self):
    """return win score for win, lose for lose and zero for tie. No clever heuristics, just outcome"""
    score = 0

    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0

      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1

      if c.COMPUTER_count == len_path:
        return c.WIN_SCORE

      elif c.HUMAN_count == len_path:
        return c.LOSE_SCORE

    return score


  def immediate_threats(self, player, board = None):
    if board == None:
      board = self.board
    open_win_paths_computer = []
    open_win_paths_human = []
    # if player == c.COMPUTER:
    winning_moves = []
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif board[space] == c.HUMAN:
          c.HUMAN_count += 1
        else:
          free_on_path.append(space)

      # first check win moves
      if player == c.HUMAN:
        if (((c.HUMAN_count == (len_path-1)) & (c.COMPUTER_count==0))):
          winning_moves.append(free_on_path[0])
      if player == c.COMPUTER:
        if (((c.COMPUTER_count == (len_path-1)) & (c.HUMAN_count==0))):
          winning_moves.append(free_on_path[0])

    return winning_moves



  def win_or_forced_move(self, player, board = None):
    if board == None:
      board = self.board
    open_win_paths_computer = []
    open_win_paths_human = []
    # if player == c.COMPUTER:
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif board[space] == c.HUMAN:
          c.HUMAN_count += 1
        else:
          free_on_path.append(space)

      # first check win moves
      if player == c.HUMAN:
        if (((c.HUMAN_count == (len_path-1)) & (c.COMPUTER_count==0))):
          return free_on_path
      if player == c.COMPUTER:
        if (((c.COMPUTER_count == (len_path-1)) & (c.HUMAN_count==0))):
          return free_on_path


    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif board[space] == c.HUMAN:
          c.HUMAN_count += 1
        else:
          free_on_path.append(space)

      # first check win moves
      if player == c.HUMAN:
        if (((c.COMPUTER_count == (len_path-1)) & (c.HUMAN_count==0))):
          return free_on_path
      if player == c.COMPUTER:
        if (((c.HUMAN_count == (len_path-1)) & (c.COMPUTER_count==0))):
          return free_on_path



    return False




  def compute_square_score_paths_old(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square"):
    """ Heurisitc function to be used for the minimax search.
    If it is a winning board for the COMPUTER, returns WIN_SCORE.
    If it is a losing board for the COMPUTER, returns LOSE_SCORE.

    Then, for all the winning paths in which the COMPUTER has i spaces,
    and the HUMAN has 0 spaces, returns 10*3^i.
    For all the winning paths in which the c.HUMAN is i away
    from winning, returns -10*3^i
    """
    score = 0
    # exp =1
    # interaction = False
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    # max_length_path_O = 0
    # if player == c.COMPUTER:
    for path in self.winning_paths:
      if (square in path) | (potential=='full'):
        len_path = len(path)
        c.COMPUTER_count, c.HUMAN_count = 0, 0
        free_on_path = []
        for space in path:
          if self.board[space] == c.COMPUTER:
            c.COMPUTER_count += 1
          elif self.board[space] == c.HUMAN:
            c.HUMAN_count += 1
          elif space!=square:
            free_on_path.append(space)

        if c.COMPUTER_count == len_path:
          # print path
          # Player wins!
          # print turns
          # print self.last_space
          # print c.WIN_SCORE-turns
          if (depth!=0):
            print 'yup'
          return c.WIN_SCORE

        elif c.HUMAN_count == len_path:
          # print path
          # Opponent wins :(
          if (depth!=0):
            print 'yup'
          return c.LOSE_SCORE

        elif c.HUMAN_count == 0:
          # Opponent not on path, so count number of player's tokens on path
          # score += 10*3**(c.COMPUTER_count - 1)
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path))

        elif c.COMPUTER_count == 0:
          # Player not on path, so count number of opponent's tokens on path
          # score -= 10*3**(c.HUMAN_count - 1)
          open_win_paths_human.append((free_on_path,c.HUMAN_count, square in path))
          if (c.HUMAN_count > max_length_path_X):
            max_length_path_X = c.HUMAN_count
        else:
          # Path cannot be won, so it has no effect on score
          pass


    streak_size = len(self.winning_paths[0])
    if (player==c.HUMAN) & (streak_size-max_length_path_X > remaining_turns_x):
    # if (streak_size-max_length_path_X > remaining_turns_x):
      return -20000000
    #
    # if (player==c.COMPUTER) & (streak_size-max_length_path_X > remaining_turns_x):
    # # if (streak_size-max_length_path_X > remaining_turns_x):
    #   return -20000000


    score = 0.0


    # compute the score for the cell based on the potential paths
    if player == c.COMPUTER:
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]

        if streak_size == p1[1]+1:
          return c.WIN_SCORE
        path_length = p1[1]
        if p1[2]:
          path_length+=1
        if streak_size == path_length:
          return c.WIN_SCORE
        score += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
        if interaction:
          for j in range(i+1, len(open_win_paths_computer)):
            p2 = open_win_paths_computer[j]
            path_length2 = p2[1]
            if p2[2]:
             path_length2+=1
            if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_o = True
                # return c.WIN_SCORE
              score += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))

    if player == c.HUMAN:
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        path_length = p1[1]
        if p1[2]:
          path_length += 1
        if path_length == streak_size:
          return  -1*c.LOSE_SCORE
        score += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
        if interaction:
          for j in range(i+1, len(open_win_paths_human)):
            p2 = open_win_paths_human[j]
            path_length2 = p2[1]
            if p2[2]:
             path_length2 += 1
            if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                return -1*c.LOSE_SCORE
              score += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))

    return score


  def compute_square_score_paths_potential(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square"):
    """ Heurisitc function to be used for the minimax search.
    If it is a winning board for the COMPUTER, returns WIN_SCORE.
    If it is a losing board for the COMPUTER, returns LOSE_SCORE.
    """
    # if (square == 18) & (potential == 'full'):
    #   print 'here'
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    # max_length_path_O = 0
    # if player == c.COMPUTER:
    for path in self.winning_paths:
      if (square in path) | (potential == 'full'):
        len_path = len(path)
        c.COMPUTER_count, c.HUMAN_count = 0, 0
        free_on_path = []
        for space in path:
          if self.board[space] == c.COMPUTER:
            c.COMPUTER_count += 1
          elif self.board[space] == c.HUMAN:
            c.HUMAN_count += 1
          elif space!=square:
            free_on_path.append(space)
          if space == square:
            if player == c.HUMAN:
              c.HUMAN_count += 1
            else:
              c.COMPUTER_count += 1



        if c.COMPUTER_count == len_path:
          return c.WIN_SCORE

        elif c.HUMAN_count == len_path:
          return c.LOSE_SCORE

        if (c.HUMAN_count == 0) & (c.COMPUTER_count>0):
          # Opponent not on path, so count number of player's tokens on path
          # score += 10*3**(c.COMPUTER_count - 1)
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path))

        if (c.COMPUTER_count == 0) & (c.HUMAN_count>0):
          # Player not on path, so count number of opponent's tokens on path
          # score -= 10*3**(c.HUMAN_count - 1)
          open_win_paths_human.append((free_on_path,c.HUMAN_count, square in path))
          if (c.HUMAN_count > max_length_path_X):
            max_length_path_X = c.HUMAN_count
        else:
          # Path cannot be won, so it has no effect on score
          pass


    streak_size = len(self.winning_paths[0])
    if (player==c.HUMAN) & (streak_size-max_length_path_X > remaining_turns_x):
    # if (streak_size-max_length_path_X > remaining_turns_x):
      return -20000000
    #
    # if (player==c.COMPUTER) & (streak_size-max_length_path_X > remaining_turns_x):
    # # if (streak_size-max_length_path_X > remaining_turns_x):
    #   return -20000000


    score = 0.0
    score_x = 0.0
    score_o = 0.0
    win_x = False
    win_o = False

    # compute the score for the cell based on the potential paths
    if (player == c.COMPUTER) | other_player:
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]
        path_length = p1[1]
        if streak_size == path_length:
          win_o = True
          return c.WIN_SCORE
        score_o += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
        if interaction:
          for j in range(i+1, len(open_win_paths_computer)):
            p2 = open_win_paths_computer[j]
            path_length2 = p2[1]
            if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_o = True
                return c.WIN_SCORE
              top = 0.0 + path_length*path_length2
              bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
              score_o += math.pow(top/bottom, exp)
    score_x_raw = 0.0
    score_x_interaction = 0.0
    if (player == c.HUMAN) | other_player:
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        path_length = p1[1]
        if path_length == streak_size:
          win_x = True
          return  -1*c.LOSE_SCORE
        score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
        score_x_raw += 1.0/math.pow((streak_size-path_length), exp)
        if interaction:
          for j in range(i+1, len(open_win_paths_human)):
            p2 = open_win_paths_human[j]
            path_length2 = p2[1]
            if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_x = True
                return  -1*c.LOSE_SCORE
              if (((streak_size-1)*(streak_size-1))-(path_length*path_length2)) == 0:
                print 'boo'
              top =0.0 + path_length*path_length2
              bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
              score_x += math.pow(top/bottom, exp)
              score_x_interaction += math.pow(top/bottom, exp)
              interaction_new = math.pow(top/bottom, exp)
              interaction_old = 1.0/(math.pow(((((streak_size-1)*(streak_size-1))-(path_length*path_length2))), exp))
              # print interaction_new
              # print interaction_old

    if other_player & (potential == 'full'):  # simply subtract other player's potential
      if player == c.COMPUTER:
        score = score_o - score_x
      else:
        score = score_x - score_o

    elif other_player & (potential == 'square'): # need to compute which paths the square blocks
      if player == c.COMPUTER:  # check X paths that can be blocked
        for i in range(len(open_win_paths_human)):
          p1 = open_win_paths_human[i]
          path_length = p1[1]
          if path_length == streak_size:
            win_x = True
            return  -1*c.LOSE_SCORE
          score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_human)):
              p2 = open_win_paths_human[j]
              path_length2 = p2[1]
              # if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap; not needed, paths with the same square always overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_x = True
                return  -1*c.LOSE_SCORE
              # score_x += 1.0/(math.pow(((path_length*path_length2)/(((streak_size-1)*(streak_size-1))-(path_length*path_length2))), exp))
              top =0.0 + path_length*path_length2
              bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
              score_x += math.pow(top/bottom, exp)

      if player == c.HUMAN:  # check O paths that can be blocked
        for i in range(len(open_win_paths_computer)):
          p1 = open_win_paths_computer[i]
          path_length = p1[1]
          if path_length == streak_size:
            win_o = True
            return c.WIN_SCORE
          score_o += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_computer)):
              p2 = open_win_paths_computer[j]
              path_length2 = p2[1]
              # if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap; not needed, paths with the same square always overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_o = True
                return c.WIN_SCORE
              # score_o += 1.0/(math.pow(((path_length*path_length2)/(((streak_size-1)*(streak_size-1))-(path_length*path_length2))), exp))
              top =0.0 + path_length*path_length2
              bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
              score_o += math.pow(top/bottom, exp)
      score = score_o + score_x

    elif not(other_player):
      if player == c.COMPUTER:
        score = score_o
      else:
        score = score_x


    return score


  def compute_square_score_paths_potential_block(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square"):
    """ Heurisitc function to be used for the minimax search.
    If it is a winning board for the COMPUTER, returns WIN_SCORE.
    If it is a losing board for the COMPUTER, returns LOSE_SCORE.
    """
    # if (square == 18) & (potential == 'full'):
    #   print 'here'
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    # max_length_path_O = 0
    # if player == c.COMPUTER:
    for path in self.winning_paths:
      if (square in path) | True:
        len_path = len(path)
        c.COMPUTER_count, c.HUMAN_count = 0, 0
        free_on_path = []
        for space in path:
          if self.board[space] == c.COMPUTER:
            c.COMPUTER_count += 1
          elif self.board[space] == c.HUMAN:
            c.HUMAN_count += 1
          elif space!=square:
            free_on_path.append(space)
          if space == square:
            if player == c.HUMAN:
              c.HUMAN_count += 1
            else:
              c.COMPUTER_count += 1



        if c.COMPUTER_count == len_path:
          return c.WIN_SCORE

        elif c.HUMAN_count == len_path:
          return c.LOSE_SCORE

        if (c.HUMAN_count == 0) & (c.COMPUTER_count>0):
          # Opponent not on path, so count number of player's tokens on path
          # score += 10*3**(c.COMPUTER_count - 1)
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path))

        if (c.COMPUTER_count == 0) & (c.HUMAN_count>0):
          # Player not on path, so count number of opponent's tokens on path
          # score -= 10*3**(c.HUMAN_count - 1)
          open_win_paths_human.append((free_on_path,c.HUMAN_count, square in path))
          if (c.HUMAN_count > max_length_path_X):
            max_length_path_X = c.HUMAN_count
        else:
          # Path cannot be won, so it has no effect on score
          pass


    streak_size = len(self.winning_paths[0])
    if (player==c.HUMAN) & (streak_size-max_length_path_X > remaining_turns_x):
    # if (streak_size-max_length_path_X > remaining_turns_x):
      return -20000000
    #
    # if (player==c.COMPUTER) & (streak_size-max_length_path_X > remaining_turns_x):
    # # if (streak_size-max_length_path_X > remaining_turns_x):
    #   return -20000000


    score = 0.0
    score_x = 0.0
    score_o = 0.0
    win_x = False
    win_o = False

    # compute the score for the cell based on the potential paths
    if (player == c.COMPUTER) | other_player:
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]
        path_length = p1[1]
        if streak_size == path_length:
          win_o = True
          return c.WIN_SCORE
        score_o += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
        if interaction:
          for j in range(i+1, len(open_win_paths_computer)):
            p2 = open_win_paths_computer[j]
            path_length2 = p2[1]
            if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_o = True
                return c.WIN_SCORE
              top = 0.0 + path_length*path_length2
              bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
              score_o += math.pow(top/bottom, exp)
    score_x_raw = 0.0
    score_x_interaction = 0.0
    if (player == c.HUMAN) | other_player:
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        path_length = p1[1]
        if path_length == streak_size:
          win_x = True
          return  -1*c.LOSE_SCORE
        score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
        score_x_raw += 1.0/math.pow((streak_size-path_length), exp)
        if interaction:
          for j in range(i+1, len(open_win_paths_human)):
            p2 = open_win_paths_human[j]
            path_length2 = p2[1]
            if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_x = True
                return  -1*c.LOSE_SCORE
              if (((streak_size-1)*(streak_size-1))-(path_length*path_length2)) == 0:
                print 'boo'
              top =0.0 + path_length*path_length2
              bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
              score_x += math.pow(top/bottom, exp)
              score_x_interaction += math.pow(top/bottom, exp)
              # interaction_new = math.pow(top/bottom, exp)
              # interaction_old = 1.0/(math.pow(((((streak_size-1)*(streak_size-1))-(path_length*path_length2))), exp))
              # print interaction_new
              # print interaction_old

    if other_player & (potential == 'full'):  # simply subtract other player's potential
      if player == c.COMPUTER:
        score = score_o - score_x
      else:
        score = score_x - score_o

    elif other_player & (potential == 'square'): # need to compute which paths the square blocks
      if player == c.COMPUTER:  # check X paths that can be blocked
        sorted_list = sorted(open_win_paths_human, key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_list),3)):
          p1 = sorted_list[i]
          path_length = p1[1]
          if path_length == streak_size:
            win_x = True
            return  -1*c.LOSE_SCORE
          score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(sorted_list)):
              p2 = sorted_list[j]
              path_length2 = p2[1]
              # if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap; not needed, paths with the same square always overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_x = True
                return  -1*c.LOSE_SCORE
              # score_x += 1.0/(math.pow(((path_length*path_length2)/(((streak_size-1)*(streak_size-1))-(path_length*path_length2))), exp))
              top =0.0 + path_length*path_length2
              bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
              score_x += math.pow(top/bottom, exp)

      if player == c.HUMAN:  # check O paths that can be blocked
        sorted_list = sorted(open_win_paths_computer, key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_list),3)):
        # for i in range(len(open_win_paths_computer)):
          p1 = sorted_list[i]
          path_length = p1[1]
          if path_length == streak_size:
            win_o = True
            return c.WIN_SCORE
          score_o += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(sorted_list)):
              p2 = sorted_list[j]
              path_length2 = p2[1]
              # if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap; not needed, paths with the same square always overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_o = True
                return c.WIN_SCORE
              # score_o += 1.0/(math.pow(((path_length*path_length2)/(((streak_size-1)*(streak_size-1))-(path_length*path_length2))), exp))
              top =0.0 + path_length*path_length2
              bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
              score_o += math.pow(top/bottom, exp)
      score = score_o + score_x

    elif not(other_player):
      if player == c.COMPUTER:
        score = score_o
      else:
        score = score_x


    return score




  def compute_square_score_paths_clean_old(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square"):
    """ Heurisitc function to be used for the minimax search.
    If it is a winning board for the COMPUTER, returns WIN_SCORE.
    If it is a losing board for the COMPUTER, returns LOSE_SCORE.
    """
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    max_length_path_O = 0

    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1
        elif space!=square:
          free_on_path.append(space)

      if c.COMPUTER_count == len_path:
        return c.WIN_SCORE

      elif c.HUMAN_count == len_path:
        return c.LOSE_SCORE

      if (c.HUMAN_count == 0) & ((c.COMPUTER_count > 0) | ((player == c.COMPUTER) & (square in path))):
        open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path))

      if (c.COMPUTER_count == 0) & ((c.HUMAN_count > 0) | ((player == c.HUMAN) & (square in path))):
        open_win_paths_human.append((free_on_path,c.HUMAN_count, square in path))

      else:
        # Path cannot be won, so it has no effect on score
        pass


    streak_size = len(self.winning_paths[0])

    score = 0.0
    score_x = 0.0
    score_o = 0.0

    # compute the score for the cell based on the potential paths
    if (player == c.COMPUTER) | other_player:
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]
        if p1[2]:
          path_length = p1[1]
          if player == c.COMPUTER:
            path_length += 1
          if path_length > max_length_path_O:
            max_length_path_O = path_length

          if streak_size == path_length:
            return c.WIN_SCORE
          score_o += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_computer)):
              p2 = open_win_paths_computer[j]
              if p2[2]:
                path_length2 = p2[1]
                if player == c.COMPUTER:
                  path_length2 += 1

                new_board = copy.deepcopy(self.board)

                if player == c.HUMAN:
                  new_board[square] = c.COMPUTER
                  next_player = c.HUMAN
                else:
                  new_board[square] = c.HUMAN
                  next_player = c.COMPUTER
                # if not(self.win_or_forced_move(next_player, board = new_board)):
                if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square, or if this is the other player we are blocking
                  if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                    if player == c.COMPUTER:
                      return c.WIN_SCORE
                    else:
                      score_o += 10
                  else:
                    top = 0.0 + path_length*path_length2
                    bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                    score_o += math.pow(top/bottom, exp)

                # if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square, or if this is the other player we are blocking
                #   if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                #     return c.WIN_SCORE
                #   top = 0.0 + path_length*path_length2
                #   bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                #   # score_o += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))
                #   score_o += math.pow(top/bottom, exp)

    if (player == c.HUMAN) | other_player:
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        if p1[2]:
          path_length = p1[1]
          if player == c.HUMAN:
            path_length+=1
          if path_length > max_length_path_X:
            max_length_path_X = path_length
          if path_length == streak_size:
            return  -1*c.LOSE_SCORE
          score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_human)):
              p2 = open_win_paths_human[j]
              if p2[2]:
                path_length2 = p2[1]
                if player == c.HUMAN:
                  path_length2+=1
                new_board = copy.deepcopy(self.board)

                if player == c.HUMAN:
                  new_board[square] = c.COMPUTER
                  next_player = c.HUMAN
                else:
                  new_board[square] = c.HUMAN
                  next_player = c.COMPUTER
                if not(self.win_or_forced_move(next_player, board = new_board)):
                  if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square, or if this is the other player we are blocking
                    if player == c.HUMAN:
                      return -1*c.LOSE_SCORE
                    else:
                      score_x += 10
                  else:
                    top = 0.0 + path_length*path_length2
                    bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                    score_o += math.pow(top/bottom, exp)
                    # if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                    #   return  -1*c.LOSE_SCORE
                    # top = 0.0 + path_length*path_length2
                    # bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                    # score_x += math.pow(top/bottom, exp)
                  # score_x += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))


    # if (player==c.HUMAN) & (streak_size-max_length_path_X > remaining_turns_x):  # can't win from here
    #   return -20000000


    if player == c.COMPUTER:
      score = score_o
    else:
      score = score_x

    if other_player:
      if player == c.COMPUTER:
        score = score + score_x
      else:
        score = score + score_o

    return score



  def compute_square_score_block_clean(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square"):
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    max_length_path_O = 0
    # if square == 4:
    #   print 'here'
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1
        elif space!=square:
          free_on_path.append(space)

      if c.COMPUTER_count == len_path:
        return c.WIN_SCORE

      elif c.HUMAN_count == len_path:
        return c.LOSE_SCORE

      if (c.HUMAN_count == 0) & ((c.COMPUTER_count > 0) | ((player == c.COMPUTER) & (square in path))):
        if (square in path) & (player == c.COMPUTER):
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count+1, square in path, path))
          if (c.COMPUTER_count+1) > max_length_path_O:
            max_length_path_O = c.COMPUTER_count+1
        else:
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count+1, square in path, path))
          if (c.COMPUTER_count) > max_length_path_O:
            max_length_path_O = c.COMPUTER_count

      if (c.COMPUTER_count == 0) & ((c.HUMAN_count > 0) | ((player == c.HUMAN) & (square in path))):
        if (square in path) & (player == c.HUMAN):
          open_win_paths_human.append((free_on_path,c.HUMAN_count+1, square in path,path))
          if (c.HUMAN_count+1) > max_length_path_X:
            max_length_path_X = c.HUMAN_count+1
        else:
          open_win_paths_human.append((free_on_path,c.HUMAN_count+1, square in path, path))
          if (c.HUMAN_count) > max_length_path_X:
            max_length_path_X = c.HUMAN_count
        # open_win_paths_human.append((free_on_path,c.HUMAN_count, square in path))

      else:
        # Path cannot be won, so it has no effect on score
        pass


    streak_size = len(self.winning_paths[0])

    score = 0.0
    score_x = 0.0
    score_o = 0.0

    # compute the score for the cell based on the potential paths
    if (player == c.COMPUTER) | (other_player & (max_length_path_X < streak_size-1)):
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]
        if p1[2]:
          path_length = p1[1]
          # if player == c.COMPUTER:
          #   path_length += 1
          # if path_length > max_length_path_O:
          #   max_length_path_O = path_length

          if (streak_size == path_length) & (player == c.COMPUTER):
            return c.WIN_SCORE
          score_o += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_computer)):
              p2 = open_win_paths_computer[j]
              if p2[2]:
                path_length2 = p2[1]
                # if player == c.COMPUTER:
                #   path_length2 += 1
                new_board = copy.deepcopy(self.board)

                if player == c.HUMAN:
                  new_board[square] = c.COMPUTER
                  next_player = c.HUMAN
                else:
                  new_board[square] = c.HUMAN
                  next_player = c.COMPUTER
                # if not(self.win_or_forced_move(next_player, board = new_board)):
                if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square, or if this is the other player we are blocking
                  if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                    if player == c.COMPUTER:
                      return c.WIN_SCORE
                    else:
                      score_o += 10

                  else:
                    top = 0.0 + path_length*path_length2
                    bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                    # score_o += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))
                    score_o += math.pow(top/bottom, exp)

    if (player == c.HUMAN) | (other_player & (max_length_path_O < streak_size-1)):
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        if p1[2]:
          path_length = p1[1]
          # if player == c.HUMAN:
          #   path_length+=1
          # if path_length > max_length_path_X:
          #   max_length_path_X = path_length
          if (path_length == streak_size) & (player == c.HUMAN):
            return  -1*c.LOSE_SCORE
          score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_human)):
              p2 = open_win_paths_human[j]
              if p2[2]:
                path_length2 = p2[1]
                new_board = copy.deepcopy(self.board)

                if player == c.HUMAN:
                  new_board[square] = c.COMPUTER
                  next_player = c.HUMAN
                else:
                  new_board[square] = c.HUMAN
                  next_player = c.COMPUTER
                # if not(self.win_or_forced_move(next_player, board = new_board)):
                # if player == c.HUMAN:
                #   path_length2+=1
                if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square, or if this is the other player we are blocking
                  if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                    if player == c.HUMAN:
                      return -1*c.LOSE_SCORE
                    else:
                      score_x += 10
                  else:
                    top = 0.0 + path_length*path_length2
                    bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                    score_x += math.pow(top/bottom, exp)
                  # score_x += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))


    # if (player==c.HUMAN) & (streak_size-max_length_path_X > remaining_turns_x):  # can't win from here
    #   return -20000000

    # check immediate threat
    if (player == c.COMPUTER) & (max_length_path_O == streak_size-1):
      # grant points for top 3 blocked X paths
      # sorted_paths = sorted(open_win_paths_human, key=lambda x: x[1], reverse=True)
      # for i in range(min(3,len(sorted_paths))):
      #   p1 = sorted_paths[i]
      #   path_length = p1[1]
      #   # if player == c.HUMAN:
      #   #   path_length+=1
      #   if path_length > max_length_path_X:
      #     max_length_path_X = path_length
      #   if path_length == streak_size:
      #     return  -1*c.LOSE_SCORE
      #   score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
      #   if interaction:
      #     for j in range(i+1, min(3,len(sorted_paths))):
      #       p2 = sorted_paths[j]
      #       path_length2 = p2[1]
      #       # if player == c.HUMAN:
      #       #   path_length2+=1
      #       if self.check_overlap(p1[3],p2[3]):  # the opponent paths interact, get extra score
      #         if (streak_size-1)*(streak_size-1) == path_length*path_length2:
      #           return  -1*c.LOSE_SCORE
      #         top = 0.0 + path_length*path_length2
      #         bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
      #         score_x += math.pow(top/bottom, exp)
      score_x = 20

    if (player == c.HUMAN) & (max_length_path_X == streak_size-1):
      # grant points for top 3 blocked O paths
      # sorted_paths = sorted(open_win_paths_computer, key=lambda x: x[1], reverse=True)
      # for i in range(min(3,len(sorted_paths))):
      #   p1 = sorted_paths[i]
      #   path_length = p1[1]
      #   # if player == c.HUMAN:
      #   #   path_length+=1
      #   if path_length > max_length_path_X:
      #     max_length_path_X = path_length
      #   if path_length == streak_size:
      #     return  -1*c.LOSE_SCORE
      #   score_o += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
      #   if interaction:
      #     for j in range(i+1, min(3,len(sorted_paths))):
      #       p2 = sorted_paths[j]
      #       path_length2 = p2[1]
      #       # if player == c.HUMAN:
      #       #   path_length2+=1
      #       if self.check_overlap(p1[3],p2[3]):  # the opponent paths interact, get extra score
      #         if (streak_size-1)*(streak_size-1) == path_length*path_length2:
      #           return  -1*c.LOSE_SCORE
      #         top = 0.0 + path_length*path_length2
      #         bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
      #         score_o += math.pow(top/bottom, exp)
      score_o = 20
    if player == c.COMPUTER:
      score = score_o
    else:
      score = score_x

    if other_player:
      if player == c.COMPUTER:
        score = score + score_x
      else:
        score = score + score_o

    return score



  def compute_square_score_paths_clean(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square"):
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    max_length_path_O = 0
    # if square == 4:
    #   print 'here'
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1
        elif space!=square:
          free_on_path.append(space)

      if c.COMPUTER_count == len_path:
        return c.WIN_SCORE

      elif c.HUMAN_count == len_path:
        return c.LOSE_SCORE

      if (c.HUMAN_count == 0) & ((c.COMPUTER_count > 0) | ((player == c.COMPUTER) & (square in path))):
        if (square in path) & (player == c.COMPUTER):
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count+1, square in path, path))
          if (c.COMPUTER_count+1) > max_length_path_O:
            max_length_path_O = c.COMPUTER_count+1
        else:
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count+1, square in path, path))
          if (c.COMPUTER_count) > max_length_path_O:
            max_length_path_O = c.COMPUTER_count

      if (c.COMPUTER_count == 0) & ((c.HUMAN_count > 0) | ((player == c.HUMAN) & (square in path))):
        if (square in path) & (player == c.HUMAN):
          open_win_paths_human.append((free_on_path,c.HUMAN_count+1, square in path,path))
          if (c.HUMAN_count+1) > max_length_path_X:
            max_length_path_X = c.HUMAN_count+1
        else:
          open_win_paths_human.append((free_on_path,c.HUMAN_count+1, square in path, path))
          if (c.HUMAN_count) > max_length_path_X:
            max_length_path_X = c.HUMAN_count
        # open_win_paths_human.append((free_on_path,c.HUMAN_count, square in path))

      else:
        # Path cannot be won, so it has no effect on score
        pass


    streak_size = len(self.winning_paths[0])

    score = 0.0
    score_x = 0.0
    score_o = 0.0

    # compute the score for the cell based on the potential paths
    if (player == c.COMPUTER) | (other_player):
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]
        if p1[2]:
          path_length = p1[1]
          # if player == c.COMPUTER:
          #   path_length += 1
          # if path_length > max_length_path_O:
          #   max_length_path_O = path_length

          if (streak_size == path_length) & (player == c.COMPUTER):
            return c.WIN_SCORE
          score_o += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_computer)):
              p2 = open_win_paths_computer[j]
              if p2[2]:
                path_length2 = p2[1]
                # if player == c.COMPUTER:
                #   path_length2 += 1
                new_board = copy.deepcopy(self.board)

                if player == c.HUMAN:
                  new_board[square] = c.COMPUTER
                  next_player = c.HUMAN
                else:
                  new_board[square] = c.HUMAN
                  next_player = c.COMPUTER
                # if not(self.win_or_forced_move(next_player, board = new_board)):
                if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square, or if this is the other player we are blocking
                  if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                    if player == c.COMPUTER:
                      return c.WIN_SCORE
                    else:
                      score_o += 10

                  else:
                    top = 0.0 + path_length*path_length2
                    bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                    # score_o += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))
                    score_o += math.pow(top/bottom, exp)

    if (player == c.HUMAN) | (other_player):
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        if p1[2]:
          path_length = p1[1]
          # if player == c.HUMAN:
          #   path_length+=1
          # if path_length > max_length_path_X:
          #   max_length_path_X = path_length
          if (path_length == streak_size) & (player == c.HUMAN):
            return  -1*c.LOSE_SCORE
          score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_human)):
              p2 = open_win_paths_human[j]
              if p2[2]:
                path_length2 = p2[1]
                new_board = copy.deepcopy(self.board)

                if player == c.HUMAN:
                  new_board[square] = c.COMPUTER
                  next_player = c.HUMAN
                else:
                  new_board[square] = c.HUMAN
                  next_player = c.COMPUTER
                # if not(self.win_or_forced_move(next_player, board = new_board)):
                # if player == c.HUMAN:
                #   path_length2+=1
                if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square, or if this is the other player we are blocking
                  if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                    if player == c.HUMAN:
                      return -1*c.LOSE_SCORE
                    else:
                      score_x += 10
                  else:
                    top = 0.0 + path_length*path_length2
                    bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                    score_x += math.pow(top/bottom, exp)
                  # score_x += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))


    # if (player==c.HUMAN) & (streak_size-max_length_path_X > remaining_turns_x):  # can't win from here
    #   return -20000000

    if player == c.COMPUTER:
      score = score_o
    else:
      score = score_x

    if other_player:
      if player == c.COMPUTER:
        score = score + score_x
      else:
        score = score + score_o

    return score


  def compute_square_score_pathsJan31(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square", reduced_opponent = True):
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    max_length_path_O = 0
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1
        elif space!=square:
          free_on_path.append(space)

      if c.COMPUTER_count == len_path:
        return c.WIN_SCORE

      elif c.HUMAN_count == len_path:
        return c.LOSE_SCORE

      if (c.HUMAN_count == 0) & ((c.COMPUTER_count > 0) | (square in path)):
        if (square in path):
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count+1, square in path, path))
          if (c.COMPUTER_count+1) > max_length_path_O:
            max_length_path_O = c.COMPUTER_count+1
        else:
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path, path))
          if (c.COMPUTER_count) > max_length_path_O:
            max_length_path_O = c.COMPUTER_count

      if (c.COMPUTER_count == 0) & ((c.HUMAN_count > 0) | (square in path)):
        if (square in path):
          open_win_paths_human.append((free_on_path,c.HUMAN_count+1, square in path, path))
          if (c.HUMAN_count+1) > max_length_path_X:
            max_length_path_X = c.HUMAN_count+1
        else:
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path, path))
          if (c.HUMAN_count) > max_length_path_X:
            max_length_path_X = c.HUMAN_count

      else:
        # Path cannot be won, so it has no effect on score
        pass


    streak_size = len(self.winning_paths[0])

    score = 0.0
    score_x = 0.0
    score_o = 0.0

    # compute the score for the cell based on the potential paths
    if (player == c.COMPUTER) | other_player:
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]
        if p1[2]:
          path_length = p1[1]
          if reduced_opponent & (player == c.HUMAN):
            path_length -= 1
          if path_length > 0:  # if it's the opponent it might be 0 again
            if streak_size == path_length:
              if player == c.COMPUTER:
                return c.WIN_SCORE
              else:
                return c.OPPONENT_THREAT
            score_o += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
            if interaction:
              for j in range(i+1, len(open_win_paths_computer)):
                p2 = open_win_paths_computer[j]
                if self.check_overlap(p1[3], p2[3]):
                  path_length2 = p2[1]
                  if reduced_opponent & (player == c.HUMAN) & (p2[2]):
                    path_length2 -= 1
                  if path_length2 > 0:
                    if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square
                      if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                        if player == c.COMPUTER:
                          return c.WIN_SCORE
                        else:
                          return c.OPPONENT_THREAT

                      else:
                        top = 0.0 + path_length*path_length2
                        bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                        score_o += math.pow(top/bottom, exp)

    if (player == c.HUMAN) | other_player:
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        if p1[2]:
          path_length = p1[1]
          if reduced_opponent & (player == c.COMPUTER):
            path_length -= 1
          if path_length > 0:  # if it's the opponent it might be 0 again
            if path_length == streak_size:
              if player == c.COMPUTER:
                return -1*c.LOSE_SCORE
              else:
                return c.OPPONENT_THREAT

            score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
            if interaction:
              for j in range(i+1, len(open_win_paths_human)):
                p2 = open_win_paths_human[j]
                if self.check_overlap(p1[3], p2[3]):
                  path_length2 = p2[1]
                  if reduced_opponent & (player == c.COMPUTER) & (p2[2]):
                    path_length2 -= 1
                  if path_length2 > 0:
                    if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square
                      if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                        if player == c.COMPUTER:
                          return -1*c.LOSE_SCORE
                        else:
                          return c.OPPONENT_THREAT
                      else:
                        top = 0.0 + path_length*path_length2
                        bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                        score_x += math.pow(top/bottom, exp)

    score = score_x + score_o
    return score


  def compute_square_score_blockJan31(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square", reduced_opponent = True):
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    max_length_path_O = 0
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1
        elif space!=square:
          free_on_path.append(space)

      if c.COMPUTER_count == len_path:
        return c.WIN_SCORE

      elif c.HUMAN_count == len_path:
        return c.LOSE_SCORE

      if (c.HUMAN_count == 0) & ((c.COMPUTER_count > 0) | (square in path)):
        if (square in path):
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count+1, square in path, path))
          if (c.COMPUTER_count+1) > max_length_path_O:
            max_length_path_O = c.COMPUTER_count+1
        else:
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path, path))
          if (c.COMPUTER_count) > max_length_path_O:
            max_length_path_O = c.COMPUTER_count

      if (c.COMPUTER_count == 0) & ((c.HUMAN_count > 0) | (square in path)):
        if (square in path):
          open_win_paths_human.append((free_on_path,c.HUMAN_count+1, square in path, path))
          if (c.HUMAN_count+1) > max_length_path_X:
            max_length_path_X = c.HUMAN_count+1
        else:
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path, path))
          if (c.HUMAN_count) > max_length_path_X:
            max_length_path_X = c.HUMAN_count

      else:
        # Path cannot be won, so it has no effect on score
        pass


    streak_size = len(self.winning_paths[0])

    score = 0.0
    score_x = 0.0
    score_o = 0.0

    # compute the score for the cell based on the potential paths
    if (player == c.COMPUTER) | other_player:
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]
        if p1[2]:
          path_length = p1[1]
          if reduced_opponent & (player == c.HUMAN):
            path_length -= 1
          if path_length > 0:  # if it's the opponent it might be 0 again
            if streak_size == path_length:
              if player == c.COMPUTER:
                return c.WIN_SCORE
              else:
                return c.OPPONENT_THREAT
            score_o += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
            if interaction:
              for j in range(i+1, len(open_win_paths_computer)):
                p2 = open_win_paths_computer[j]
                if self.check_overlap(p1[3], p2[3]):
                  path_length2 = p2[1]
                  if reduced_opponent & (player == c.HUMAN) & (p2[2]):
                    path_length2 -= 1
                  if path_length2 > 0:
                    if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square
                      if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                        if player == c.COMPUTER:
                          return c.WIN_SCORE
                        else:
                          return c.OPPONENT_THREAT

                      else:
                        top = 0.0 + path_length*path_length2
                        bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                        score_o += math.pow(top/bottom, exp)

    if (player == c.HUMAN) | other_player:
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        if p1[2]:
          path_length = p1[1]
          if reduced_opponent & (player == c.COMPUTER):
            path_length -= 1
          if path_length > 0:  # if it's the opponent it might be 0 again
            if path_length == streak_size:
              if player == c.COMPUTER:
                return -1*c.LOSE_SCORE
              else:
                return c.OPPONENT_THREAT

            score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
            if interaction:
              for j in range(i+1, len(open_win_paths_human)):
                p2 = open_win_paths_human[j]
                if self.check_overlap(p1[3], p2[3]):
                  path_length2 = p2[1]
                  if reduced_opponent & (player == c.COMPUTER) & (p2[2]):
                    path_length2 -= 1
                  if path_length2 > 0:
                    if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square
                      if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                        if player == c.COMPUTER:
                          return -1*c.LOSE_SCORE
                        else:
                          return c.OPPONENT_THREAT
                      else:
                        top = 0.0 + path_length*path_length2
                        bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                        score_x += math.pow(top/bottom, exp)


    if (max_length_path_O == streak_size-1) & (player == c.COMPUTER):
      score_x = c.PLAYER_THREAT
    elif (max_length_path_X == streak_size-1) & (player == c.HUMAN):
      score_o = c.PLAYER_THREAT

    score = score_x + score_o
    return score


  def compute_square_score_paths_reducedO(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square"):
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    max_length_path_O = 0
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1
        elif space!=square:
          free_on_path.append(space)

      if c.COMPUTER_count == len_path:
        return c.WIN_SCORE

      elif c.HUMAN_count == len_path:
        return c.LOSE_SCORE

      if (c.HUMAN_count == 0) & (c.COMPUTER_count > 0):
        if (square in path) & (player == c.COMPUTER):
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count+1, square in path, path))
          if (c.COMPUTER_count+1) > max_length_path_O:
            max_length_path_O = c.COMPUTER_count+1
        else:
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path, path))
          if (c.COMPUTER_count) > max_length_path_O:
            max_length_path_O = c.COMPUTER_count

      if (c.COMPUTER_count == 0) & ((c.HUMAN_count > 0) | ((player == c.HUMAN) & (square in path))):
        if (square in path) & (player == c.HUMAN):
          open_win_paths_human.append((free_on_path,c.HUMAN_count+1, square in path, path))
          if (c.HUMAN_count+1) > max_length_path_X:
            max_length_path_X = c.HUMAN_count+1
        else:
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path, path))
          if (c.HUMAN_count) > max_length_path_X:
            max_length_path_X = c.HUMAN_count

      else:
        # Path cannot be won, so it has no effect on score
        pass


    streak_size = len(self.winning_paths[0])

    score = 0.0
    score_x = 0.0
    score_o = 0.0

    # compute the score for the cell based on the potential paths
    if player == c.COMPUTER:
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]
        if p1[2]:
          path_length = p1[1]
          if streak_size == path_length:
            return c.WIN_SCORE
          score_o += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_computer)):
              p2 = open_win_paths_computer[j]
              if self.check_overlap(p1[3], p2[3]):
                path_length2 = p2[1]
                new_board = copy.deepcopy(self.board)
                if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square
                  if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                    return c.WIN_SCORE

                  else:
                    top = 0.0 + path_length*path_length2
                    bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                    score_o += math.pow(top/bottom, exp)

    if player == c.HUMAN:
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        if p1[2]:
          path_length = p1[1]
          if path_length == streak_size:
            return  -1*c.LOSE_SCORE
          score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_human)):
              p2 = open_win_paths_human[j]
              if self.check_overlap(p1[3], p2[3]):
                path_length2 = p2[1]
                if (not(self.check_overlap(p1[0],p2[0]))):  # interaction score if the paths can't be blocked by same square
                  if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                    return -1*c.LOSE_SCORE
                  else:
                    top = 0.0 + path_length*path_length2
                    bottom = ((streak_size-1)*(streak_size-1))-(path_length*path_length2)
                    score_x += math.pow(top/bottom, exp)

    return score



  def compute_square_score_paths_potential_block_old(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square"):
    """ Heurisitc function to be used for the minimax search.
    If it is a winning board for the COMPUTER, returns WIN_SCORE.
    If it is a losing board for the COMPUTER, returns LOSE_SCORE.
    """

    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    max_length_path_O = 0
    # max_length_path_O = 0
    # if player == c.COMPUTER:
    for path in self.winning_paths:
      if (square in path) | (potential == 'full'):
        len_path = len(path)
        c.COMPUTER_count, c.HUMAN_count = 0, 0
        free_on_path = []
        for space in path:
          if self.board[space] == c.COMPUTER:
            c.COMPUTER_count += 1
          elif self.board[space] == c.HUMAN:
            c.HUMAN_count += 1
          elif space!=square:
            free_on_path.append(space)
          if space == square:
            if player == c.HUMAN:
              c.HUMAN_count += 1
            else:
              c.COMPUTER_count += 1



        if c.COMPUTER_count == len_path:
          return c.WIN_SCORE

        elif c.HUMAN_count == len_path:
          return c.LOSE_SCORE

        if (c.HUMAN_count == 0):
          # Opponent not on path, so count number of player's tokens on path
          # score += 10*3**(c.COMPUTER_count - 1)
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path))
          if (c.COMPUTER_count > max_length_path_O):
            max_length_path_O = c.COMPUTER_count

        if (c.COMPUTER_count == 0):
          # Player not on path, so count number of opponent's tokens on path
          # score -= 10*3**(c.HUMAN_count - 1)
          open_win_paths_human.append((free_on_path,c.HUMAN_count, square in path))
          if (c.HUMAN_count > max_length_path_X):
            max_length_path_X = c.HUMAN_count
        else:
          # Path cannot be won, so it has no effect on score
          pass


    streak_size = len(self.winning_paths[0])
    # if (player==c.HUMAN) & (streak_size-max_length_path_X > remaining_turns_x):
    # # if (streak_size-max_length_path_X > remaining_turns_x):
    #   return -20000000
    # #
    # if (player==c.COMPUTER) & (streak_size-max_length_path_X > remaining_turns_x):
    # # if (streak_size-max_length_path_X > remaining_turns_x):
    #   return -20000000


    score = 0.0
    score_x = 0.0
    score_o = 0.0
    win_x = False
    win_o = False

    # compute the score for the cell based on the potential paths
    if (player == c.COMPUTER) | other_player:
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]
        path_length = p1[1]
        if streak_size == path_length:
          win_o = True
          return c.WIN_SCORE
        score_o += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
        if interaction:
          for j in range(i+1, len(open_win_paths_computer)):
            p2 = open_win_paths_computer[j]
            path_length2 = p2[1]
            if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_o = True
                return c.WIN_SCORE
              score_o += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))
    score_x_raw = 0.0
    score_x_interaction = 0.0
    if (player == c.HUMAN) | other_player:
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        path_length = p1[1]
        if path_length == streak_size:
          win_x = True
          return  -1*c.LOSE_SCORE
        score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
        score_x_raw += 1.0/math.pow((streak_size-path_length), exp)
        if interaction:
          for j in range(i+1, len(open_win_paths_human)):
            p2 = open_win_paths_human[j]
            path_length2 = p2[1]
            if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_x = True
                return  -1*c.LOSE_SCORE
              score_x += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))
              score_x_interaction += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))

    if other_player & (potential == 'full'):  # simply subtract other player's potential
      if player == c.COMPUTER:
        if max_length_path_O == (streak_size-1):
          score_x = 0
        score = score_o - score_x
      else:
        if max_length_path_X == (streak_size-1):
          score_o = 0
        score = score_x - score_o

    elif other_player & (potential == 'square'): # need to compute which paths the square blocks
      if player == c.COMPUTER:  # check X paths that can be blocked
        for i in range(len(open_win_paths_human)):
          p1 = open_win_paths_human[i]
          path_length = p1[1]
          if path_length == streak_size:
            win_x = True
            return  -1*c.LOSE_SCORE
          score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_human)):
              p2 = open_win_paths_human[j]
              path_length2 = p2[1]
              # if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap; not needed, paths with the same square always overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_x = True
                return  -1*c.LOSE_SCORE
              score_x += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))

      if player == c.HUMAN:  # check O paths that can be blocked
        for i in range(len(open_win_paths_computer)):
          p1 = open_win_paths_computer[i]
          path_length = p1[1]
          if path_length == streak_size:
            win_o = True
            return c.WIN_SCORE
          score_o += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_computer)):
              p2 = open_win_paths_computer[j]
              path_length2 = p2[1]
              # if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap; not needed, paths with the same square always overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_o = True
                return c.WIN_SCORE
              score_o += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))
      score = score_o + score_x

    elif not(other_player):
      if player == c.COMPUTER:
        score = score_o
      else:
        score = score_x


    return score



  def compute_square_score_paths(self, square, turns = 0, player = None, remaining_turns_x = None, depth = 0, other_player = True, interaction = False, exp = 1, potential = "square"):
    """ Heurisitc function to be used for the minimax search.
    If it is a winning board for the COMPUTER, returns WIN_SCORE.
    If it is a losing board for the COMPUTER, returns LOSE_SCORE.
    """

    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    # max_length_path_O = 0
    # if player == c.COMPUTER:
    for path in self.winning_paths:
      if (square in path) | (potential == 'full'):
        len_path = len(path)
        c.COMPUTER_count, c.HUMAN_count = 0, 0
        free_on_path = []
        for space in path:
          if self.board[space] == c.COMPUTER:
            c.COMPUTER_count += 1
          elif self.board[space] == c.HUMAN:
            c.HUMAN_count += 1
          elif space!=square:
            free_on_path.append(space)

        if c.COMPUTER_count == len_path:
          return c.WIN_SCORE

        elif c.HUMAN_count == len_path:
          return c.LOSE_SCORE

        elif (c.HUMAN_count == 0):
          # Opponent not on path, so count number of player's tokens on path
          # score += 10*3**(c.COMPUTER_count - 1)
          open_win_paths_computer.append((free_on_path,c.COMPUTER_count, square in path))

        elif (c.COMPUTER_count == 0):
          # Player not on path, so count number of opponent's tokens on path
          # score -= 10*3**(c.HUMAN_count - 1)
          open_win_paths_human.append((free_on_path,c.HUMAN_count, square in path))
          if (c.HUMAN_count > max_length_path_X):
            max_length_path_X = c.HUMAN_count
        else:
          # Path cannot be won, so it has no effect on score
          pass


    streak_size = len(self.winning_paths[0])
    if (player==c.HUMAN) & (streak_size-max_length_path_X > remaining_turns_x):
    # if (streak_size-max_length_path_X > remaining_turns_x):
      return -20000000
    #
    # if (player==c.COMPUTER) & (streak_size-max_length_path_X > remaining_turns_x):
    # # if (streak_size-max_length_path_X > remaining_turns_x):
    #   return -20000000


    score = 0.0
    score_x = 0.0
    score_o = 0.0
    win_x = False
    win_o = False

    # compute the score for the cell based on the potential paths
    if (player == c.COMPUTER) | other_player:
      for i in range(len(open_win_paths_computer)):
        p1 = open_win_paths_computer[i]
        path_length = p1[1]
        if p1[2] & (player == c.COMPUTER):
          path_length+=1
        if streak_size == path_length:
          win_o = True
          return c.WIN_SCORE
        score_o += 1.0/math.pow((streak_size-(path_length)), exp)  # score for individual path
        if interaction:
          for j in range(i+1, len(open_win_paths_computer)):
            p2 = open_win_paths_computer[j]
            path_length2 = p2[1]
            if p2[2] & (player == c.COMPUTER):
             path_length2+=1
            if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_o = True
                return c.WIN_SCORE
              score_o += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))

    if (player == c.HUMAN) | other_player:
      for i in range(len(open_win_paths_human)):
        p1 = open_win_paths_human[i]
        path_length = p1[1]
        if p1[2] & (player == c.HUMAN):
          path_length += 1
        if path_length == streak_size:
          win_x = True
          return  -1*c.LOSE_SCORE
        score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
        if interaction:
          for j in range(i+1, len(open_win_paths_human)):
            p2 = open_win_paths_human[j]
            path_length2 = p2[1]
            if p2[2] & (player == c.HUMAN):
             path_length2 += 1
            if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_x = True
                return  -1*c.LOSE_SCORE
              score_x += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))

    if other_player & (potential == 'full'):  # simply subtract other player's potential
      if player == c.COMPUTER:
        score = score_o - score_x
      else:
        score = score_x - score_o

    elif other_player & (potential == 'square'): # need to compute which paths the square blocks
      if player == c.COMPUTER:  # check X paths that can be blocked
        for i in range(len(open_win_paths_human)):
          p1 = open_win_paths_human[i]
          path_length = p1[1]
          if path_length == streak_size:
            win_x = True
            return  -1*c.LOSE_SCORE
          score_x += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_human)):
              p2 = open_win_paths_human[j]
              path_length2 = p2[1]
              # if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap; not needed, paths with the same square always overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_x = True
                return  -1*c.LOSE_SCORE
              score_x += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))

      if player == c.HUMAN:  # check O paths that can be blocked
        for i in range(len(open_win_paths_computer)):
          p1 = open_win_paths_computer[i]
          path_length = p1[1]
          if path_length == streak_size:
            win_o = True
            return c.WIN_SCORE
          score_o += 1.0/math.pow((streak_size-path_length), exp)  # score for individual path
          if interaction:
            for j in range(i+1, len(open_win_paths_computer)):
              p2 = open_win_paths_computer[j]
              path_length2 = p2[1]
              # if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap; not needed, paths with the same square always overlap
              if (streak_size-1)*(streak_size-1) == path_length*path_length2:
                win_o = True
                return c.WIN_SCORE
              score_o += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(path_length*path_length2), exp))
      score = score_o + score_x

    return score

  def check_possible_win(self, remaining_turns_O=0):
    open_win_paths_computer = []
    open_win_paths_human = []

    max_length_path_O = 0
    # if player == c.COMPUTER:
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1
        else:
          free_on_path.append(space)

      if c.COMPUTER_count == len_path:
        return False

      elif c.HUMAN_count == len_path:
        # print path
        # Opponent wins :(
        return True




      elif c.HUMAN_count == 0:
        # Opponent not on path, so count number of player's tokens on path
        # score += 10*3**(c.COMPUTER_count - 1)
        open_win_paths_computer.append((free_on_path,c.COMPUTER_count))


      elif c.COMPUTER_count == 0:
        # Player not on path, so count number of opponent's tokens on path
        # score -= 10*3**(c.HUMAN_count - 1)
        open_win_paths_human.append((free_on_path,c.HUMAN_count))
        if (c.HUMAN_count > max_length_path_O):
          max_length_path_O = c.HUMAN_count
      else:
        # Path cannot be won, so it has no effect on score
        pass
    exp=2
    score = 0.0
    streak_size = len(self.winning_paths[0])

    if (streak_size-max_length_path_O > remaining_turns_O):
      return False
    return True

  def obj_win_loss(self, player, remaining_turns_x = 0):
    """ Heurisitc function to be used for the minimax search.
    If it is a winning board for the COMPUTER, returns WIN_SCORE.
    If it is a losing board for the COMPUTER, returns LOSE_SCORE.

    Then, for all the winning paths in which the COMPUTER has i spaces,
    and the HUMAN has 0 spaces, returns 10*3^i.
    For all the winning paths in which the c.HUMAN is i away
    from winning, returns -10*3^i
    """
    score = 0
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    max_length_path_O = 0
    # if player == c.COMPUTER:
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1
        else:
          free_on_path.append(space)

      if c.COMPUTER_count == len_path:
        # print path
        # Player wins!
        # print turns
        # print self.last_space
        # print c.WIN_SCORE-turns
        return c.WIN_SCORE

      elif c.HUMAN_count == len_path:
        # print path
        # Opponent wins :(
        return c.LOSE_SCORE

      return 0


  def obj_interaction(self, player, remaining_turns_x = 0, depth = 0, other_player = True, exp = 1, interaction = False, block = False):
    """ Heurisitc function to be used for the minimax search.
    If it is a winning board for the COMPUTER, returns WIN_SCORE.
    If it is a losing board for the COMPUTER, returns LOSE_SCORE.

    Then, for all the winning paths in which the COMPUTER has i spaces,
    and the HUMAN has 0 spaces, returns 10*3^i.
    For all the winning paths in which the c.HUMAN is i away
    from winning, returns -10*3^i
    """
    score = 0
    # print depth
    open_win_paths_computer = []
    open_win_paths_human = []
    max_length_path_X = 0
    # max_length_path_O = 0
    # if player == c.COMPUTER:
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      free_on_path = []
      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1
        else:
          free_on_path.append(space)

      if c.COMPUTER_count == len_path:
        return c.WIN_SCORE

      elif c.HUMAN_count == len_path:
        # print path
        # Opponent wins :(
        return c.LOSE_SCORE

      if (c.HUMAN_count == 0) & (c.COMPUTER_count > 0):
        # Opponent not on path, so count number of player's tokens on path
        open_win_paths_computer.append((free_on_path,c.COMPUTER_count))

      if (c.COMPUTER_count == 0) & (c.HUMAN_count > 0):
        # Player not on path, so count number of opponent's tokens on path
        open_win_paths_human.append((free_on_path,c.HUMAN_count))
        if (c.HUMAN_count > max_length_path_X):
          max_length_path_X = c.HUMAN_count
      else:
        # Path cannot be won, so it has no effect on score
        pass

    score = 0.0
    streak_size = len(self.winning_paths[0])


    # compute the score for the cell based on the potential paths
    score_O = 0
    score_X = 0
    for i in range(len(open_win_paths_computer)):
      p1 = open_win_paths_computer[i]
      score_O += 1.0/math.pow((streak_size-p1[1]), exp)  # score for individual path
      if interaction:
        for j in range(i+1, len(open_win_paths_computer)):
          p2 = open_win_paths_computer[j]
          if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
            if (streak_size-1)*(streak_size-1) == p1[1]*p2[1]:
              return c.WIN_SCORE
            top = 0.0 + p1[1]*p2[1]
            bottom = ((streak_size-1)*(streak_size-1))-(p1[1]*p2[1])
            score_O += math.pow(top/bottom, exp)


    for i in range(len(open_win_paths_human)):
      p1 = open_win_paths_human[i]
      score_X += 1.0/math.pow((streak_size-p1[1]), exp)  # score for individual path
      if interaction:
        for j in range(i+1, len(open_win_paths_human)):
          p2 = open_win_paths_human[j]
          if not(self.check_overlap(p1[0],p2[0])):  # interaction score if the paths don't overlap
            if (streak_size-1)*(streak_size-1) == p1[1]*p2[1]:
              return c.LOSE_SCORE
            # score_X -= 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(p1[1]*p2[1]), exp))
            top = 0.0 + p1[1]*p2[1]
            bottom = ((streak_size-1)*(streak_size-1))-(p1[1]*p2[1])
            score_X += math.pow(top/bottom, exp)


    if player==c.COMPUTER:
      score = score_O - score_X
      # if other_player:
      #   score = score - score_X

    elif player==c.HUMAN:
      score = -1*score_X + score_O
      # if other_player:
      #   score = score + score_O

    return score

  def check_overlap(self,p1,p2):
    for p in p1:
      if p in p2:
        return True
    return False

  def obj(self, player, turns = 0):
    """ Heurisitc function to be used for the minimax search. 
    If it is a winning board for the COMPUTER, returns WIN_SCORE.
    If it is a losing board for the COMPUTER, returns LOSE_SCORE. 
    
    Then, for all the winning paths in which the COMPUTER has i spaces,
    and the HUMAN has 0 spaces, returns 10*3^i. 
    For all the winning paths in which the c.HUMAN is i away 
    from winning, returns -10*3^i
    """
    score = 0
    # if player == c.COMPUTER:
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0

      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1

      if c.COMPUTER_count == len_path:
        # print path
        # Player wins!
        # print turns
        # print self.last_space
        # print c.WIN_SCORE-turns

        return c.WIN_SCORE
      elif c.HUMAN_count == len_path:
        # print path
        # Opponent wins :(
        return c.LOSE_SCORE
      elif c.HUMAN_count == 0:
        # Opponent not on path, so count number of player's tokens on path
        score += 10*3**(c.COMPUTER_count - 1)
      elif c.COMPUTER_count == 0:
        # Player not on path, so count number of opponent's tokens on path
        score -= 10*3**(c.HUMAN_count - 1)
      else:
        # Path cannot be won, so it has no effect on score
        pass

    return score
    
  def is_terminal(self):
    ''' Returns True if the board is terminal, False if not. '''
    # First, check to see if the board is won
    # objective_score = self.obj()
    # if objective_score == c.WIN_SCORE:
    #   return c.COMPUTER
    # elif objective_score == c.LOSE_SCORE:
    #   return c.HUMAN
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0

      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1

      if c.COMPUTER_count == len_path:
        # print path
        # Player wins!
        return c.COMPUTER
      elif c.HUMAN_count == len_path:
        # print path
        # Opponent wins :(
        return c.HUMAN
    # else:
      # Then, check to see if there are any c.BLANK spaces
    for space in self.board:
      if self.board[space] == c.BLANK:
        return False
    return c.TIE

  def is_terminal_mcts(self):
    ''' Returns True if the board is terminal, False if not. '''
    # First, check to see if the board is won
    # objective_score = self.obj()
    # if objective_score == c.WIN_SCORE:
    #   return c.COMPUTER
    # elif objective_score == c.LOSE_SCORE:
    #   return c.HUMAN
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0

      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1

      if c.COMPUTER_count == len_path:
        # print path
        # Player wins!
        return True
      elif c.HUMAN_count == len_path:
        # print path
        # Opponent wins :(
        return True
    # else:
      # Then, check to see if there are any c.BLANK spaces
    for space in self.board:
      if self.board[space] == c.BLANK:
        return False
    return True
