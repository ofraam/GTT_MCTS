from __future__ import print_function

import random
import utils
import config as c
from utils import convert_ab_board_to_matrix, convert_position_to_row_col, convert_position_to_int
from board import compute_paths_scores_for_matrix

NODE_COUNTER = 0

class MCTS(object):
    """
    The central MCTS class, which performs the tree search. It gets a
    tree policy, a default policy, and a backup strategy.
    See e.g. Browne et al. (2012) for a survey on monte carlo tree search
    """
    def __init__(self, tree_policy, default_policy, backup):
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.backup = backup
        self.node_counter = 0


    def __call__(self, root, n=100):
        """
        Run the monte carlo tree search.

        :param root: The StateNode
        :param n: The number of roll-outs to be performed
        :return:
        """
        if root.parent is not None:
            raise ValueError("Root's parent must be None.")
        for _ in range(n):
            # print (root.children.values())
            if c.NUM_NODES >= c.NODE_LIMIT:
                return (utils.rand_max(root.children.values(), key=lambda x: x.q).action,c.NUM_NODES)

            (node, count, path) = _get_next_node(root, self.tree_policy)

            # c.NUM_NODES +=1
            node.reward = self.default_policy(node, current_path=path)
            self.node_counter+=count
            self.backup(node)

        # print ()
        return (utils.rand_max(root.children.values(), key=lambda x: x.q).action,c.NUM_NODES)


def _expand(state_node, current_path=[]):
    action = random.choice(state_node.untried_actions)
    # print (action)
    # c.NUM_NODES +=1
    pos = convert_position_to_row_col(action, c.DIMENSION)
    pos_move = [pos[0], pos[1], state_node.state.player]

    # if str(current_path) in c.PATHS_DICT:
    #     c.PATHS_DICT[str(current_path)][1] += 1
    # else:
    #     c.PATHS_DICT[str(current_path)] = [len(current_path), 1, convert_ab_board_to_matrix(state_node.state.board.board), str(pos[0])+'_'+str(pos[1])]
    c.PATHS_DICT.append([str(current_path), convert_ab_board_to_matrix(state_node.state.board.board), str(pos[0])+'_'+str(pos[1]), 'simulation'])
    current_path.append(pos_move)
    return state_node.children[action].sample_state()


def _best_child(state_node, tree_policy, current_path=[]):
    # print ('best')
    best_action_node = utils.rand_max(state_node.children.values(),
                                      key=tree_policy)
    pos = convert_position_to_row_col(best_action_node.action, c.DIMENSION)
    pos_move = [pos[0], pos[1], state_node.state.player]

    # if str(current_path) in c.PATHS_DICT:
    #     c.PATHS_DICT[str(current_path)][1] += 1
    c.PATHS_DICT.append([str(current_path), convert_ab_board_to_matrix(state_node.state.board.board), str(pos[0])+'_'+str(pos[1]), 'simulation'])
    current_path.append(pos_move)
    # else:
    #     c.PATHS_DICT[str(current_path)] = [len(current_path), 1, convert_ab_board_to_matrix(state_node.state.board.board), {str(pos[0]+'_'+pos[1]): 1}]
    return best_action_node.sample_state()


def _get_next_node(state_node, tree_policy):
    current_path = []
    node_counter = 0
    while not state_node.state.is_terminal():
        # print ('while')
        if state_node.untried_actions:
            c.NUM_NODES +=1
            return (_expand(state_node, current_path),node_counter, current_path)
        else:
            c.NUM_NODES +=1
            state_node = _best_child(state_node, tree_policy, current_path)
    # print (node_counter)
    return (state_node,c.NUM_NODES, current_path)
