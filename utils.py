import random
import numpy as np
import copy
import math


def convert_ab_board_to_matrix(ab_board):
    board_matrix = []
    dimension = int(math.sqrt(len(ab_board)))
    i = 1
    for row in range(dimension):
        board_matrix.append([])
        for col in range(dimension):
            board_matrix[row].append(ab_board[i])
            i += 1
    return board_matrix


def convert_position_to_row_col(pos, dimension):
    col = int(((pos - 1) % dimension))
    row = (float(pos)/float(dimension))-1
    row = int(math.ceil(row))
    return row, col



def convert_position_to_int(row, col, dimension):
    return row * dimension + col + 1

def get_open_paths_through_square(row, col, board, player='X'):
    other_player = 'O'
    if player == 'O':
        other_player = 'X'

    max_length_path = 0
    threshold = -1
    # if len(board)==10:
    #     threshold +=1

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    paths = []
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player==1):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((other_player == 2) & path_x_count+1>threshold)):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 2:
                open_paths_data.extend(path)
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.extend(path)

    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player==1):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 2:
                open_paths_data.extend(path)
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.extend(path)

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player==1):
                empty_squares.append([square_row,square_col])

            path.append([square_row,square_col])
            square_row += 1
            path_length += 1



        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 2:
                open_paths_data.extend(path)
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.extend(path)

    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player==1):
                empty_squares.append([square_row, square_col])

            path.append([square_row, square_col])
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 2:
                open_paths_data.extend(path)
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif path_x_count > threshold:
                open_paths_data.extend(path)

    return remove_duplicates(open_paths_data)


def remove_duplicates(square_list):
    unique_squares = []
    for square in square_list:
        if not check_square_in_list(square, unique_squares):
            unique_squares.append(square)
    return unique_squares


def expand_neighborhood(squares, size, prob=1.0):
    checked = []
    new_neighborhood = []
    for square in squares:
        if str(square) not in checked:
            neighbors = get_neighboring_squares(size, square, 1)
            for neighbor in neighbors:
                if not check_square_in_list(neighbor,new_neighborhood):
                    if random.random() <= prob:
                        new_neighborhood.append(neighbor)
                    # else:
                    #     print 'not adding'
            checked.append(str(square))
            # new_neighborhood.extend(get_neighboring_squares(size, square, 1))
    return remove_duplicates(new_neighborhood)


def check_square_in_list(square, squares_list):
    for i in range(len(squares_list)):
        if (square[0] == squares_list[i][0]) & (square[1] == squares_list[i][1]):
            return True

    return False

def get_neighboring_squares(size, square, neighborhood_size):
    neighbors = []
    row = square[0]
    col = square[1]
    for i in range(-1*neighborhood_size,neighborhood_size+1):
        for j in range(-1*neighborhood_size,neighborhood_size+1):
            if (i != 0) | (j != 0):
                r = row + i
                c = col + j
                if (r < size) & (r >= 0) & (c < size) & (c >= 0):
                    neighbors.append([r,c])
    return neighbors
    # if (square[0]+1) < size:
    #     neighbors.append([square[0]+1,square[1]])
    #     if (square[1]-1) > 0:
    #         neighbors.append([square[0]+1,square[1]-1])
    #     if (square[1]+1) < size:
    #         neighbors.append([square[0]+1,square[1]+1])
    # if (square[0]-1) > 0:
    #     neighbors.append([square[0]-1,square[1]])
    #     if (square[1]-1) > 0:
    #         neighbors.append([square[0]-1,square[1]-1])
    #     if (square[1]+1) < size:
    #         neighbors.append([square[0]-1,square[1]+1])
    # if (square[1]+1) < size:
    #     neighbors.append([square[0],square[1]+1])
    # if (square[1]-1) > 0:
    #     neighbors.append([square[0],square[1]-1])


def rand_max(iterable, key=None):
    """
    A max function that tie breaks randomly instead of first-wins as in
    built-in max().
    :param iterable: The container to take the max from
    :param key: A function to compute tha max from. E.g.:
      >>> rand_max([-2, 1], key=lambda x:x**2
      -2
      If key is None the identity is used.
    :return: The entry of the iterable which has the maximum value. Tie
    breaks are random.
    """
    if key is None:
        key = lambda x: x

    max_v = -np.inf
    max_l = []

    for item, value in zip(iterable, [key(i) for i in iterable]):
        # print item
        # print value
        if value == max_v:
            max_l.append(item)
        elif value > max_v:
            max_l = [item]
            max_v = value

    return random.choice(max_l)

if __name__== "__main__":
    row, col = convert_position_to_row_col(23, 10)
    num = convert_position_to_int(row, col, 10)
    print row
    print col
    print num