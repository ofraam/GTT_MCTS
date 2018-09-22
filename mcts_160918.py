from __future__ import print_function

import random
import utils
import config as c
import copy
import math

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
        self.move_matrix = copy.deepcopy(c.BOARD)
        for row in range(len(self.move_matrix)):
            for col in range(len(self.move_matrix)):
                if self.move_matrix[row][col] == 1:
                    self.move_matrix[row][col] = 'X'
                elif self.move_matrix[row][col] == 2:
                    self.move_matrix[row][col] = 'O'
                else:
                    self.move_matrix[row][col] = 0.0

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
            (node,count) = _get_next_node(root, self.tree_policy, move_matrix=self.move_matrix)

            # c.NUM_NODES +=1
            node.reward = self.default_policy(node)
            self.node_counter+=count
            self.backup(node)
        # print ()
        return (utils.rand_max(root.children.values(), key=lambda x: x.q).action,c.NUM_NODES)


def _expand(state_node, move_matrix = None):
    action = random.choice(state_node.untried_actions)
    # print (action)
    c.NUM_NODES +=1
    if move_matrix != None:
        col = ((action - 1) % len(move_matrix))
        row = (float(action)/float(len(move_matrix)))-1
        row = int(math.ceil(row))
        move_matrix[row][col] += 1
    return state_node.children[action].sample_state()


def _best_child(state_node, tree_policy):
    # print ('best')
    best_action_node = utils.rand_max(state_node.children.values(),
                                      key=tree_policy)
    return best_action_node.sample_state()


def _get_next_node(state_node, tree_policy, move_matrix = None):
    node_counter = 0
    while not state_node.state.is_terminal():
        # print ('while')
        c.NUM_NODES +=1
        if state_node.untried_actions:
            return (_expand(state_node,move_matrix),node_counter)
        else:
            state_node = _best_child(state_node, tree_policy)
    # print (node_counter)
    return (state_node,c.NUM_NODES)
