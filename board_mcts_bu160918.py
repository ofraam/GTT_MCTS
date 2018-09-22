import numpy as np
# import mcts
# import mcts_local
from mcts import *
from tree_policies import *
from default_policies import *
from backups import *
# from backups import monte_carlo
from utils import *
from graph import *
import config as c
import os
from computational_model import *
import copy
import random
import csv

class Bellman(object):
    """
    A dynamical programming update which resembles the Bellman equation
    of value iteration.

    See Feldman and Domshlak (2014) for reference.
    """
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, node):
        """
        :param node: The node to start the backups from
        """
        while node is not None:
            node.n += 1
            if isinstance(node, StateNode):
                node.q = max([x.q for x in node.children.values()])
            elif isinstance(node, ActionNode):
                n = sum([x.n for x in node.children.values()])
                node.q = sum([(self.gamma * x.q + x.reward) * x.n
                              for x in node.children.values()]) / n
            node = node.parent


def monte_carlo(node):
    """
    A monte carlo update as in classical UCT.

    See feldman amd Domshlak (2014) for reference.
    :param node: The node to start the backup from
    """
    r = node.reward
    while node is not None:
        node.n += 1
        node.q = ((node.n - 1)/node.n) * node.q + 1/node.n * r
        node = node.parent

class TicTacToeAction(object):
    def __init__(self, move):
        self.move = move

    def __eq__(self, other):
        return self.move == other.move

    def __hash__(self):
        return self.move


class TicTactToeState(object):
    def __init__(self, board, player, depth):
        self.board = board
        self.actions = self.board.get_free_spaces();
        self.player = player
        self.depth = depth

    def perform(self, action):
        new_board = copy.deepcopy(self.board)
        if action not in new_board.get_free_spaces():
            print 'prob'
        new_board.add_marker(action, player=self.player)

        if self.player == c.HUMAN:
            player = c.COMPUTER
        else:
            player=c.HUMAN
        depth = self.depth+1
        return TicTactToeState(new_board, player, depth)

    # def reward(self, parent, action):
    #     # return -1*self.board.obj_interaction(self.player)
    #     if self.board.is_terminal():
    #         if self.board.obj(self.player) == c.LOSE_SCORE:
    #              return 1
    #          # if self.board.obj(self.player) == c.WIN_SCORE:
    #         else:
    #             return -1
    #     elif self.board.check_possible_win(math.ceil(self.depth/2.0)):
    #          return 0
    #     else:
    #          return -1
    #     return 0

    def reward(self, parent, action):
        # return -1*self.board.obj_interaction(self.player)
        if self.board.is_terminal():
            if self.board.obj(self.player) == c.LOSE_SCORE:
                 # print 'win'
                 return 10000
             # if self.board.obj(self.player) == c.WIN_SCORE:
            else:
                return -10000
        # elif self.board.check_possible_win(math.ceil(self.depth/2.0)):
        #      return -10000
        # else:
        #      return -10000
        # return -1*self.board.obj_interaction(self.player, exp=2, other_player=False)
        return -10000
    # def reward(self, parent, action):
    #     # return -1*self.board.obj_interaction(self.player)
    #     if self.board.is_terminal():
    #         if self.board.obj(self.player) == c.LOSE_SCORE:
    #             return -1*c.LOSE_SCORE
    #         if self.board.obj(self.player) == c.WIN_SCORE:
    #             return -1*c.WIN_SCORE
    #         else:
    #             return -1*self.board.obj_interaction(player=c.HUMAN, exp=2)
    #     # elif self.board.check_possible_win(math.ceil(self.depth/2.0)):
    #     #     return -10000
    #     else:
    #         return -1*self.board.obj_interaction(player=c.HUMAN, exp=2)
    #     # return 0

    def is_terminal(self):
        # if self.depth == 7:
        #     print 7
        # print self.depth
        if self.board.is_terminal():
            return True
        if c.WIN_DEPTH == self.depth:
            # print 'depth'
            return True
        # if self.board.check_possible_win(math.ceil(self.depth/2.0)):
        #     return True
        # # return True
        # if (random.random()<0.05):
        return False

    def __eq__(self, other):
        # return (str(self.board) == str(other.board)) & (str(self.player) == str(other.player)) & (str(self.depth) == str(other.depth))
        if (len(self.actions)!=len(other.actions)):
            return False
        for act in self.actions:
            if act not in other.actions:
                return False
        if self.depth!= other.depth:
            return False
        if self.player!=other.player:
            return False
        for i in range(1,len(self.board.board)+1):
            if self.board.board[i] != other.board.board[i]:
                return False
        return True

    def __hash__(self):
        # print self.board.board
        # print hash(str(self.board.board)+str(self.player)+'_'+str(self.depth))
        return hash(str(self.board.board)+str(self.player)+'_'+str(self.depth))


class MazeAction(object):
    def __init__(self, move):
        self.move = np.asarray(move)

    def __eq__(self, other):
        return all(self.move == other.move)

    def __hash__(self):
        return 10*self.move[0] + self.move[1]

class MazeState(object):
    def __init__(self, pos):
        self.pos = np.asarray(pos)
        self.actions = [MazeAction([1, 0]),
                        MazeAction([0, 1]),
                        MazeAction([-1, 0]),
                        MazeAction([0, -1])]

    def perform(self, action):
        pos = self.pos + action.move
        pos = np.clip(pos, 0, 2)
        return MazeState(pos)

    def reward(self, parent, action):
        if all(self.pos == np.array([2, 2])):
            return 10
        else:
            return -1

    def is_terminal(self):
        # # return True
        # if se
        # if (random.random()<0.5):
        if (random.random()<0.01):
            return True
        return False
        # return True

    def __eq__(self, other):
        return all(self.pos == other.pos)

    def __hash__(self):
        return 10 * self.pos[0] + self.pos[1]

if __name__ == "__main__":

    root =  StateNode(None, MazeState(np.array([2, 2])))
    # best_action, num_nodes = mcts(root, n=50)
    # print best_action
    #
    results = []
    data_matrices = {}
    for filename in os.listdir("predefinedBoards/"):

        # print filename
        if filename.startswith("6"):
            file_path = "examples/board_6_4.txt"
            # continue
            if not(filename.startswith("6_easy_pruned")):
                continue

        else:
            # if filename.startswith("10by10_easy"):
            # if not(filename.startswith("10_easy")):
            #   continue
            file_path = "examples/board_10_5.txt"
            continue
        chosen_moves = {}
        num_runs = 50
        num_correct = 0.0
        print filename

        total_nodes= 0.0
        success = 0.0
        n = 500
        game = start_game(file_path)

        win_depth = fill_board_from_file("predefinedBoards/"+filename,game)

        mcts = MCTS(tree_policy=UCB1(c=1.41),
                    default_policy=random_terminal_roll_out,
                    # default_policy=immediate_reward,
                    backup=monte_carlo)
        # while (success < 0.95):
        #     n += 50
        #     print n
        total_nodes= 0.0
        move_matrix_aggregate = copy.deepcopy(mcts.move_matrix)
        for r in range(0,len(move_matrix_aggregate)):
            for j in range(0,len(move_matrix_aggregate[r])):
                if (move_matrix_aggregate[r][j]=='X'):
                    move_matrix_aggregate[r][j] = -0.00001
                elif (move_matrix_aggregate[r][j]=='O'):
                    move_matrix_aggregate[r][j] = -0.00002

        for i in range(0,num_runs):
            if i % 10 == 0:
                print i
            game = start_game(file_path)

            win_depth = fill_board_from_file("predefinedBoards/"+filename,game)
            c.WIN_DEPTH = win_depth

            root = StateNode(None,TicTactToeState(game.board,game.whos_turn,0))
            # print c.SIM
            # c.SIM = 50
            best_action, num_nodes = mcts(root, n=n)
            move_matrix = mcts.move_matrix

            move_count = 0.0
            # for r in range(0,len(move_matrix)):
            #     for j in range(0,len(move_matrix[r])):
            #         if (move_matrix[r][j]=='X'):
            #             move_matrix[r][j] = -0.00001
            #         elif (move_matrix[r][j]=='O'):
            #             move_matrix[r][j] = -0.00002
            #         else:
            #             move_count += move_matrix[r][j]
            #         #     print move_matrix[i][j]
            #         # else:
            #         #     print  move_matrix[i][j]
            #
            # for r in range(0,len(move_matrix)):
            #     for j in range(0,len(move_matrix[r])):
            #         if (move_matrix[r][j]>0):
            #             move_matrix[r][j] = move_matrix[r][j]/move_count
            #             move_matrix_aggregate[r][j] += move_matrix[r][j]

            # print move_matrix_aggregate

        # for r in range(0,len(move_matrix_aggregate)):
        #     for j in range(0,len(move_matrix_aggregate[r])):
        #         if (move_matrix_aggregate[r][j]>0):
        #             move_matrix_aggregate[r][j] = move_matrix_aggregate[r][j]/num_runs

        # data_matrices[filename[:-5]] = copy.deepcopy(move_matrix_aggregate)
        # print move_matrix_aggregate
        # print 'done board'

    # write_matrices_to_file(data_matrices, 'data_matrices/cogsci/mctsPeoplePerformance.json')

            # print num_nodes
            # print num_nodes
            # print best_action

            if best_action in chosen_moves.keys():
                chosen_moves[best_action] += 1
            else:
                chosen_moves[best_action] = 1
            correct = 0
            if best_action in c.WIN_MOVES:
                num_correct += 1
                correct = 1
            total_nodes += num_nodes
            res = []
            res.append(filename[:-5])
            res.append(best_action)
            res.append(correct)
            res.append(num_nodes)
            # print num_nodes
            # print num_nodes
            c.NUM_NODES = 0
            results.append(res)

        # print total_nodes/num_runs
        # print chosen_moves
        # print num_correct/num_runs
        # success = num_correct/num_runs
        # print success
        # if success > 0.95:
        #     print num_nodes
        # results.append(num_correct/num_runs)
        # print '----------'
        # print n
        # print total_nodes/num_runs
        # print num_correct/num_runs
        # print chosen_moves
        # print '----------'

    dataFile = open('stats/mctsRuns160918_6easy.csv', 'wb')
    dataWriter = csv.writer(dataFile, delimiter=',')
    dataWriter.writerow(['board','chosen_move','correct','nodes'])
    for res in results:
        print res
        dataWriter.writerow(res)
    # print results





    #
    # root = StateNode(None, MazeState([0, 0]))
    # best_action = mcts(root)
    # print best_action.move