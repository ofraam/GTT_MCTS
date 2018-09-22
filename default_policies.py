import random
import config as c
from utils import convert_ab_board_to_matrix, convert_position_to_row_col, convert_position_to_int


def immediate_reward(state_node):
    """
    Estimate the reward with the immediate return of that state.
    :param state_node:
    :return:
    """
    return state_node.state.reward(state_node.parent.parent.state,
                                   state_node.parent.action)


class RandomKStepRollOut(object):
    """
    Estimate the reward with the sum of returns of a k step rollout
    """
    def __init__(self, k):
        self.k = k

    def __call__(self, state_node):
        self.current_k = 0

        def stop_k_step(state):
            # print 'here'
            self.current_k += 1
            return self.current_k > self.k or state.is_terminal()

        return _roll_out(state_node, stop_k_step)


def random_terminal_roll_out(state_node, current_path=[]):
    """
    Estimate the reward with the sum of a rollout till a terminal state.
    Typical for terminal-only-reward situations such as games with no
    evaluation of the board as reward.

    :param state_node:
    :return:
    """
    def stop_terminal(state):
        return state.is_terminal()

    return _roll_out(state_node, stop_terminal, current_path)


def _roll_out(state_node, stopping_criterion, current_path=[]):
    reward = 0
    state = state_node.state
    parent = state_node.parent.parent.state
    action = state_node.parent.action
    # print '--rollout--'
    # print 'depth =' +str(state.depth)
    # print action
    # current_path = []
    while not stopping_criterion(state):
        c.NUM_NODES_ROLLOUTS += 1
        # reward += state.reward(parent, action)

        # action = random.choice(state_node.state.actions)
        # print state.actions
        action = random.choice(state.actions)
        # print 'player =' +str(state.player)
        # print action
        parent = state
        pos = convert_position_to_row_col(action, c.DIMENSION)
        pos_move = [pos[0], pos[1], state.player]
        current_path.append(pos_move)
        c.PATHS_DICT.append([str(current_path), convert_ab_board_to_matrix(state.board.board), str(pos[0])+'_'+str(pos[1]), 'rollout'])
        state = parent.perform(action)

    reward += state.reward(parent, action)
    # print reward
    return reward

    # reward = 0
    # state = state_node.state
    # if state_node.parent is None:
    #     return state.reward(None, None)
    # parent = state_node.parent.parent.state
    # action = state_node.parent.action
    # counter = 0
    # # print '--rollout--'
    # while not stopping_criterion(state):
    #     reward += state.reward(parent, action)
    #
    #     # action = random.choice(state_node.state.actions)
    #     # print state.actions
    #     action = random.choice(state.actions)
    #     # print action
    #     if action not in state.actions:
    #         print 'problem1'
    #     # parent = state
    #
    #     if action not in parent.actions:
    #         print 'problme2'
    #     state = parent.perform(action)
    #     c.NUM_NODES+=1.0
    #     # counter += 1
    #     # print c.NUM_NODES
    # # print counter
    # # print c.NUM_NODES
    # reward += state.reward(parent, action)
    # return reward
