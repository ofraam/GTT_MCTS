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
from multiprocessing import freeze_support
from time import time,sleep
import concurrent.futures


class Params():
    def __init__(self, num_runs=6, csv_filename=''):
        self.num_runs = num_runs
        self.csv_filename = csv_filename


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
        self.player = player
        self.depth = depth

        forced_move = board.win_or_forced_move(player)
        if forced_move:
            moves = forced_move
            # print 'forced'
        else:
            moves, nodes_computed, missed_win = self.board.get_free_spaces_ranked_heuristic_model(player=self.player, shutter=False, noise=0, k=3, stochastic_order=True)

        self.actions = moves;
        # self.actions = self.board.get_free_spaces();

        # self.prior_n = 20.0
        # self.prior_q = -1.0*self.board.obj_interaction(self.player)
        self.prior_n = None
        self.prior_q = None

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
        # if self.board.is_terminal():
        # print self.board.obj(c.HUMAN)
        # print self.board.board
        if self.board.obj(c.HUMAN) == c.LOSE_SCORE:
             # print self.board.board
             # print 'win'
             # print self.board.obj_interaction(self.player, exp=2, other_player=False)
             return 1000
             # return 1
        elif self.board.obj(c.HUMAN) == c.WIN_SCORE:
             # print 'lose'
             return -1000
             # return 0
         # if self.board.obj(self.player) == c.WIN_SCORE:
        # else:
        #     # print 'loss'
        #     return -10000
        # elif self.board.check_possible_win(math.ceil(self.depth/2.0)) == False:
        #      return -10000
        # else:
        #      return -10000
        # print  -1*self.board.obj_interaction(self.player, exp=2, other_player=False)*100
        # return -1*self.board.obj_interaction(self.player, exp=2, other_player=False)
        # return -10000
        # return -1000
        if (self.depth > 3) & (self.board.check_possible_win(math.ceil((c.WIN_DEPTH - self.depth + 1)/2.0))==False):
        # # # # if self.board.check_possible_win(math.ceil(self.depth/2.0)):
        # #     print 'cant win'
            return -1000
        return 0
        # print 'nothing'
        # print 'nothing'
        # return -10000
        # return -1*self.board.obj(c.HUMAN)
        # return -1
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
        if self.board.is_terminal_mcts():
            # print 'end'
            return True
        if c.WIN_DEPTH == self.depth:
            # print self.depth
            return True
        if (self.depth > 3) & (self.board.check_possible_win(math.ceil((c.WIN_DEPTH - self.depth + 1)/2.0))==False):
        # # # # if self.board.check_possible_win(math.ceil(self.depth/2.0)):
        # #     print 'cant win'
            return True
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
        self.prior_n = None
        self.prior_q = None
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


def parallel_mcts(max_workers, N=50):
    starttime = time()
    N_multi = int(N / max_workers)
    temp_file_names = ["stats\\mcts\\"+"{0}TEMP_10boards".format(i) for i in range(max_workers)]
    params = [Params(num_runs=N_multi, csv_filename=name, ) for name in temp_file_names]
    param_list = params
    #param_list = [params]*max_workers
    #_helper(params)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Process the list of files, but split the work across the process pool to use all CPUs!
        for i, h in enumerate(executor.map(run_mcts, param_list)):
            print("i={0} finished".format(i))
    # merge_csv(Commons.TEMPDIR, csv_filename)
    # shutil.rmtree(Commons.TEMPDIR)
    print("total time:{0}".format(time() - starttime))


def run_mcts(params):
    results = []
    simulation_data = {}
    simulation_data['num_nodes'] = []
    simulation_data['chosen_moves'] = []
    simulation_data['correct'] = []
    simulation_data['board'] = []
    simulation_data['run'] = []
    header = ['num_nodes','chosen_moves','correct','board', 'run']
    csv_filename = params.csv_filename
    write_header = True
    for filename in os.listdir("predefinedBoards/"):

        randomk =  RandomKStepRollOut(3)
        mcts = MCTS(tree_policy=UCB1(c=1.41),
                    default_policy=random_terminal_roll_out,
                    # default_policy=randomk,
                    backup=monte_carlo)
        print filename
        # print 'k step'
        if filename.startswith("6"):
            file_path = "examples/board_6_4.txt"
            c.DIMENSION = 6
            # continue
            # if not(filename.startswith("6_easy")):
            #    continue

        else:
            # if filename.startswith("10by10_easy"):
            # if not(filename.startswith("10_easy")):
            #   continue
            file_path = "examples/board_10_5.txt"
            c.DIMENSION = 10
            continue
        chosen_moves = {}
        num_runs = params.num_runs
        num_correct = 0.0
        print filename

        total_nodes = 0.0
        success = -0.1

        n = 5
        game = start_game(file_path)

        win_depth = fill_board_from_file("predefinedBoards/"+filename,game)
        if win_depth == 6:
            n = 50
        elif win_depth == 8:
            n = 50
        else:
            n = 50
        n=100000
        prev_n = n
        c.NODE_LIMIT = 100000
        while ((success < c.SOLVED-c.SOLVED*0.05) & (n<=prev_n)):

        # while (success < 0):
            board_results = []
            simulation_data = {}
            simulation_data['num_nodes'] = []
            simulation_data['num_nodes_rollouts'] = []
            simulation_data['chosen_moves'] = []
            simulation_data['correct'] = []
            simulation_data['board'] = []
            simulation_data['run'] = []

            print success
            n = n*2
            print n
            total_nodes= 0.0
            num_correct = 0.0
            for i in range(0, num_runs):
                if i%10 == 0:
                    print i
                c.NUM_NODES = 0
                c.NUM_NODES_ROLLOUTS = 0
                game = start_game(file_path)

                win_depth = fill_board_from_file("predefinedBoards/"+filename,game)
                c.WIN_DEPTH = win_depth
                root = StateNode(None,TicTactToeState(game.board,game.whos_turn,0))

                best_action, num_nodes = mcts(root, n=n)
                # print best_action
                # print num_nodes

                if best_action in chosen_moves.keys():
                    chosen_moves[best_action] += 1
                else:
                    chosen_moves[best_action] = 1
                correct = 0
                if best_action in c.WIN_MOVES:
                    num_correct += 1
                    correct = 1
                total_nodes += num_nodes

                # print total_nodes
                simulation_data['num_nodes'] = num_nodes
                simulation_data['num_nodes_rollouts'] = c.NUM_NODES_ROLLOUTS
                simulation_data['chosen_moves'] = best_action
                simulation_data['correct'] = correct
                simulation_data['board'] = filename[:-5]
                simulation_data['run'] = i
                board_results.append(copy.deepcopy(simulation_data))
                # print 'len dict'
                # print len(c.PATHS_DICT)
            # print total_nodes/num_runs
            # print chosen_moves
            # print num_correct/num_runs
            success = num_correct/num_runs

            results.extend(board_results)
            # dataFilePath = open(csv_filename+'_paths.csv', 'ab')
            # dataWriterPaths = csv.writer(dataFilePath, delimiter=',')
            # if write_header:
            #    dataWriterPaths.writerow(['path', 'board_state','position', 'board', 'type'])
            #    write_header = False
            # for p in c.PATHS_DICT:
                # print p
                # p.append(str(filename[:-5]))
                # print p
                # dataWriterPaths.writerow(p)
            # dataFilePath.close()
            c.PATHS_DICT = []
        print '----------'
        print n
        print total_nodes/num_runs
        print num_correct
        print num_runs
        print num_correct/num_runs
        print chosen_moves
        print '----------'
        # dataFilePath = open('stats\mcts170918_paths_k3_stochastic_1000nodes_test3.csv', 'ab')
        # dataWriterPaths = csv.writer(dataFilePath, delimiter=',')
        # if write_header:
        #     dataWriterPaths.writerow(['path', 'board_state','position', 'board'])
        #     write_header = False
        # for p in c.PATHS_DICT:
        #     # print p
        #     p.append(str(filename[:-5]))
        #     # print p
        #     dataWriterPaths.writerow(p)
        # dataFilePath.close()
        # c.PATHS_DICT = []
        dataFile = open(csv_filename+'_100000_stats230918.csv', 'ab')
        dataWriter = csv.DictWriter(dataFile, fieldnames=simulation_data.keys(), delimiter=',')
        if write_header:
            dataWriter.writeheader()
            write_header = False
        for record in board_results:
            dataWriter.writerow(record)
        dataFile.close()

    print results


if __name__ == "__main__":
    parallel_mcts(10, N=500)
    exit()
    # mcts = MCTS(tree_policy=UCB1(c=1.41),
    #             default_policy=random_terminal_roll_out,
    #             # default_policy=randomk,
    #             backup=monte_carlo)
    # root =  StateNode(None, MazeState(np.array([2, 2])))
    # best_action, num_nodes = mcts(root, n=50)
    # print best_action
    # print best_action
    #
    results = []
    simulation_data = {}
    simulation_data['num_nodes'] = []
    simulation_data['chosen_moves'] = []
    simulation_data['correct'] = []
    simulation_data['board'] = []
    simulation_data['run'] = []
    header = ['num_nodes','chosen_moves','correct','board', 'run']
    write_header = True
    for filename in os.listdir("predefinedBoards/"):

        randomk =  RandomKStepRollOut(3)
        mcts = MCTS(tree_policy=UCB1(c=1.41),
                    default_policy=random_terminal_roll_out,
                    # default_policy=randomk,
                    backup=monte_carlo)
        print filename
        # print 'k step'
        if filename.startswith("6"):
            file_path = "examples/board_6_4.txt"
            c.DIMENSION = 6
            continue
            if not(filename.startswith("6_easy")):
                continue

        else:
            # if filename.startswith("10by10_easy"):
            # if not(filename.startswith("10_easy")):
            #   continue
            file_path = "examples/board_10_5.txt"
            c.DIMENSION = 10
            # continue
        chosen_moves = {}
        num_runs = 500
        num_correct = 0.0
        print filename

        total_nodes = 0.0
        success = -0.1

        n = 5
        game = start_game(file_path)

        win_depth = fill_board_from_file("predefinedBoards/"+filename,game)
        if win_depth == 6:
            n = 50
        elif win_depth == 8:
            n = 50
        else:
            n = 50
        n=100000
        prev_n = n
        c.NODE_LIMIT = 20000
        while ((success < c.SOLVED-c.SOLVED*0.05) & (n<=prev_n)):

        # while (success < 0):
            board_results = []
            simulation_data = {}
            simulation_data['num_nodes'] = []
            simulation_data['num_nodes_rollouts'] = []
            simulation_data['chosen_moves'] = []
            simulation_data['correct'] = []
            simulation_data['board'] = []
            simulation_data['run'] = []

            print success
            n = n*2
            print n
            total_nodes= 0.0
            num_correct = 0.0
            for i in range(0, num_runs):
                if i%10 == 0:
                    print i
                c.NUM_NODES = 0
                c.NUM_NODES_ROLLOUTS = 0
                game = start_game(file_path)

                win_depth = fill_board_from_file("predefinedBoards/"+filename,game)
                c.WIN_DEPTH = win_depth
                root = StateNode(None,TicTactToeState(game.board,game.whos_turn,0))

                best_action, num_nodes = mcts(root, n=n)
                # print best_action
                # print num_nodes

                if best_action in chosen_moves.keys():
                    chosen_moves[best_action] += 1
                else:
                    chosen_moves[best_action] = 1
                correct = 0
                if best_action in c.WIN_MOVES:
                    num_correct += 1
                    correct = 1
                total_nodes += num_nodes

                # print total_nodes
                simulation_data['num_nodes'] = num_nodes
                simulation_data['num_nodes_rollouts'] = c.NUM_NODES_ROLLOUTS
                simulation_data['chosen_moves'] = best_action
                simulation_data['correct'] = correct
                simulation_data['board'] = filename[:-5]
                simulation_data['run'] = i
                board_results.append(copy.deepcopy(simulation_data))
                # print 'len dict'
                # print len(c.PATHS_DICT)
            # print total_nodes/num_runs
            # print chosen_moves
            # print num_correct/num_runs
            success = num_correct/num_runs

            results.extend(board_results)
            dataFilePath = open('stats\mcts210918_paths_k3_stochastic__20000nodes_6easy.csv', 'ab')
            dataWriterPaths = csv.writer(dataFilePath, delimiter=',')
            if write_header:
                dataWriterPaths.writerow(['path', 'board_state','position', 'board', 'type'])
                write_header = False
            for p in c.PATHS_DICT:
                # print p
                p.append(str(filename[:-5]))
                # print p
                dataWriterPaths.writerow(p)
            dataFilePath.close()
            c.PATHS_DICT = []
        print '----------'
        print n
        print total_nodes/num_runs
        print num_correct
        print num_runs
        print num_correct/num_runs
        print chosen_moves
        print '----------'
        # dataFilePath = open('stats\mcts170918_paths_k3_stochastic_1000nodes_test3.csv', 'ab')
        # dataWriterPaths = csv.writer(dataFilePath, delimiter=',')
        # if write_header:
        #     dataWriterPaths.writerow(['path', 'board_state','position', 'board'])
        #     write_header = False
        # for p in c.PATHS_DICT:
        #     # print p
        #     p.append(str(filename[:-5]))
        #     # print p
        #     dataWriterPaths.writerow(p)
        # dataFilePath.close()
        # c.PATHS_DICT = []
        dataFile = open('stats\mcts210918_k3_stochastic__20000nodes_6easy.csv', 'ab')
        dataWriter = csv.DictWriter(dataFile, fieldnames=simulation_data.keys(), delimiter=',')
        if write_header:
            dataWriter.writeheader()
            write_header = False
        for record in board_results:
            dataWriter.writerow(record)
        dataFile.close()

    print results





    #
    # root = StateNode(None, MazeState([0, 0]))
    # best_action = mcts(root)
    # print best_action.move
