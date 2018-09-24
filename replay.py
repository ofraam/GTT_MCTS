import csv
import re
import copy
from user_game import *
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from model import *
import model
import pandas as pd
import ast
# from stats_analyses import *

LOGFILE = ['logs/6_hard_full_dec19.csv','logs/6_hard_pruned_dec19.csv','logs/10_hard_full_dec19.csv','logs/10_hard_pruned_dec19.csv', 'logs/6_easy_full_dec19.csv','logs/6_easy_pruned_dec19.csv','logs/10_easy_full_dec19.csv','logs/10_easy_pruned_dec19.csv','logs/10_medium_full_dec19.csv','logs/10_medium_pruned_dec19.csv']

# LOGFILE = ['logs/6_hard_verify_dec19.csv','logs/10_hard_verify_dec19.csv', 'logs/6_easy_verify_dec19.csv','logs/10_easy_verify_dec19.csv','logs/10_medium_verify_dec3.csv']

USERID = '11e212ff'
DIMENSION = 6
START_POSITION = [[[0,2,0,0,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[0,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                  [[0,2,0,1,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[2,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                  [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,0,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[0,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                 [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,1,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[2,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                  [[0,1,0,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,0],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                  [[0,1,2,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,1],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                [[0,0,0,0,1,0,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                  [[0,0,0,0,1,2,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,1],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]],
                 [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,1,1,1,2,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]]
                  ]

# START_POSITION = [[[0,2,0,1,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[0,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
#                  [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,1,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[0,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
#                   [[0,1,0,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,1],[1,0,2,2,0,0],[0,0,0,0,0,0]],
#                   [[0,0,0,0,1,0,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,1],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
#                  [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,1,1,1,0,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]]
#                   ]

IGNORE_LIST = [[[0,3],[3,0]],
                  None,
                  [[2,4],[6,0]],
                None,
               [[0,2],[3,5]],
                None,
                [[0,5],[4,9]],
               None,
            [[3,2],[3,5]],
               None]

def replay():
    with open(LOGFILE, 'rb') as csvfile:
        log_reader = csv.DictReader(csvfile)
        for row in log_reader:
            if row['userid'] == USERID:
                if row['key'] in ('click','undo','reset'):
                    draw_board(row)


def seperate_log(log_file):
    with open(log_file, 'rb') as csvfile:
        log_reader = csv.DictReader(csvfile)
        curr_log = ''
        curr_log_records = []
        for row in log_reader:
            log = row['boardSize']+'_'+row['boardType']+'_'+row['condition']
            if log == curr_log:
                curr_log_records.append(row)
            elif len(curr_log_records)>0:
                dataFile = open('logs/'+curr_log+'_dec19.csv', 'wb')
                print curr_log_records[0]
                dataWriter = csv.DictWriter(dataFile, fieldnames=curr_log_records[0].keys(), delimiter=',')
                dataWriter.writeheader()
                for record in curr_log_records:
                    dataWriter.writerow(record)
                curr_log_records = []
                curr_log = log
            else:
                curr_log = log

def heat_map_game(normalized = False):
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
        # move_matrix = []
        # for row in range(DIMENSION):
        #     row_positions = []
        #     for col in range(DIMENSION):
        #         row_positions.append(0)
        #     move_matrix.append(copy.deepcopy(row_positions))
        # to_ignore = IGNORE_LIST[g];
        move_count = 0.0
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                # if row['userid'] == USERID:
                if row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O') & (player==1)):
                        move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1.0
                        move_count+=1.0
                    # else:
                    #     print 'bad click'
                    #
                    # # print move_matrix
                    #     move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
        print LOGFILE[g]
        for row in move_matrix:
            print row

        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[i])):
                if (move_matrix[r][j]=='X'):
                    move_matrix[r][j] = -1
                elif (move_matrix[r][j]=='O'):
                    move_matrix[r][j] = -2
                #     print move_matrix[i][j]
                # else:
                #     print  move_matrix[i][j]

        if (normalized):
            for r in range(0,len(move_matrix)):
                for j in range(0,len(move_matrix[i])):
                    # if (move_matrix[r][j]>0):
                    move_matrix[r][j] = move_matrix[r][j]/move_count
                    # else:
        #

        print move_matrix
        a = np.array(move_matrix)
        a = np.flip(a,0)
        print a
        heatmap = plt.pcolor(a)

        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if((a[y,x]==-1) | (a[y,x]==-1.0/move_count)):
                    plt.text(x + 0.5, y + 0.5, 'X',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif((a[y,x]==-2) | (a[y,x]==-2.0/move_count)):
                    plt.text(x + 0.5, y + 0.5, 'O',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif(a[y,x]!=0):
                    plt.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                     )

        fig = plt.colorbar(heatmap)
        fig_file_name = LOGFILE[g]
        fig_file_name=fig_file_name[:-4]
        fig_file_name = fig_file_name + '.png'
        plt.savefig(fig_file_name)
        plt.clf()

        # plt.imshow(a, cmap='hot', interpolation='nearest')

        # plt.show()





def heat_map_solution(normalized = False):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
        # move_matrix = []
        # for row in range(DIMENSION):
        #     row_positions = []
        #     for col in range(DIMENSION):
        #         row_positions.append(0)
        #     move_matrix.append(copy.deepcopy(row_positions))
        # to_ignore = IGNORE_LIST[g];
        move_count = 0.0
        rows = ['1','2','3','4','5','6','7','8','9','10']
        cols = ['a','b','c','d','e','f','g','h','i','j']
        user_count = 0
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                # if row['userid'] == USERID:
                if row['key']=='best_move':
                    # print row
                    move = str(row['value']).lower()
                    user_count+=1
                    # print move
                    if(len(move)==2):
                        if (move[0] in cols):
                            colPos = cols.index(move[0])
                            if(move[1] in rows):
                                rowPos = len(move_matrix)-rows.index(move[1])-1


                                if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                                    move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1.0
                                    move_count+=1.0
                        elif (move[0] in rows):
                            rowPos = rows.index(move[0])
                            if(move[1] in cols):
                                colPos = len(move_matrix)-cols.index(move[1])-1


                                if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                                    move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1.0
                                    move_count+=1.0
                    # else:
                    #     print 'bad click'
                    #
                    # # print move_matrix
                    #     move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
        print LOGFILE[g]
        print user_count
        for row in move_matrix:
            print row

        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[i])):
                if (move_matrix[r][j]=='X'):
                    move_matrix[r][j] = -0.00001
                elif (move_matrix[r][j]=='O'):
                    move_matrix[r][j] = -0.00002
                #     print move_matrix[i][j]
                # else:
                #     print  move_matrix[i][j]

        if (normalized):
            for r in range(0,len(move_matrix)):
                for j in range(0,len(move_matrix[r])):
                    if (move_matrix[r][j]>0):
                        move_matrix[r][j] = move_matrix[r][j]/move_count
                    # else:

        fig_file_name = LOGFILE[g]
        fig_file_name=fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = move_matrix

        print move_matrix

    write_matrices_to_file(data_matrices,'data_matrices/participant_solutions.json')
        # a = np.array(move_matrix)
        # a = np.flip(a,0)
        # print a
        # heatmap = plt.pcolor(a)
        #
        # for y in range(a.shape[0]):
        #     for x in range(a.shape[1]):
        #         if((a[y,x]==-1) | (a[y,x]==-1.0/move_count)):
        #             plt.text(x + 0.5, y + 0.5, 'X',
        #                  horizontalalignment='center',
        #                  verticalalignment='center',
        #                  color='white'
        #             )
        #         elif((a[y,x]==-2) | (a[y,x]==-2.0/move_count)):
        #             plt.text(x + 0.5, y + 0.5, 'O',
        #                  horizontalalignment='center',
        #                  verticalalignment='center',
        #                  color='white'
        #             )
        #         elif(a[y,x]!=0):
        #             plt.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
        #                      horizontalalignment='center',
        #                      verticalalignment='center',
        #                      color='white'
        #              )
        #
        # fig = plt.colorbar(heatmap)
        # fig_file_name = LOGFILE[g]
        # fig_file_name=fig_file_name[:-4]
        # fig_file_name = fig_file_name + 'solutionHeatmap.png'
        # plt.savefig(fig_file_name)
        # plt.clf()

        # plt.imshow(a, cmap='hot', interpolation='nearest')

        # plt.show()

def compare_paths(p1,p2):
    if len(p1)!=len(p2):
        return False
    for i in range(len(p1)):
        if p1[i]!=p2[i]:
            return False

    return True

def add_path_count(paths_counts, new_path):
    for path in paths_counts:
        if(compare_paths(path[0],new_path)):
            path[1] = path[1]+1.0
            return
    paths_counts.append([new_path,1.0])

def add_path_count_subpaths(paths_counts, new_path):
    for i in range(0,len(new_path)):
        add_path_count(paths_counts,new_path[0:i+1])


def user_stats(subpaths=False):
    user_data_headers = ['boardSize','boardType','condition','userid','curr_user_nodes','curr_user_num_paths','curr_user_sum_depth','curr_user_undo','curr_user_restart', 'confidence', 'correctness']
    users_data = []
    user_counter = 0
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        path_counter = 0.0
        taken_cells = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells



        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            paths = []
            paths_counts = []
            curr_path = []
            curr_user = ''
            curr_user_num_paths = 0
            curr_user_undo = 0
            curr_user_restart = 0
            curr_user_nodes = 0
            curr_user_sum_depth = 0
            curr_user_data = {}

            # user_data_headers = ['boardSize','boardType','condition','userid','curr_user_nodes','curr_user_num_paths','curr_user_sum_depth','curr_user_undo','curr_user_restart']


            for row in log_reader:
                # if row['userid'] == USERID:
                if curr_user=='':
                    curr_user = row['userid']



                if row['userid']!=curr_user:
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        add_path_count(paths_counts,copy.deepcopy(curr_path))
                        path_counter+=1
                        curr_user_num_paths+=1
                        curr_user_sum_depth = curr_user_sum_depth + len(curr_path)


                    cond = LOGFILE[g][5:-10]
                    condition_details = cond.split('_')
                    curr_user_data['boardSize'] = condition_details[0]
                    curr_user_data['boardType'] = condition_details[1]
                    curr_user_data['condition'] = condition_details[2]
                    curr_user_data['userid'] = curr_user


                    # print condition_details

                    curr_user_data['curr_user_num_paths'] = curr_user_num_paths
                    curr_user_data['curr_user_undo'] = curr_user_undo
                    curr_user_data['curr_user_restart'] = curr_user_restart
                    curr_user_data['curr_user_nodes'] = curr_user_nodes
                    curr_user_data['curr_user_sum_depth'] = curr_user_sum_depth
                    users_data.append(copy.deepcopy(curr_user_data))

                    curr_user_num_paths = 0
                    curr_user_undo = 0
                    curr_user_restart = 0
                    curr_user_nodes = 0
                    curr_user_sum_depth = 0
                    curr_user_data = {}

                    curr_path = []
                    curr_user = row['userid']
                    user_counter+=1

                elif row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        # move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                        curr_path.append([rowPos,colPos,player])
                        curr_user_nodes+=1

                elif row['key']=='reset':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1

                        curr_user_num_paths+=1
                        # curr_user_nodes+=1
                        curr_user_sum_depth = curr_user_sum_depth + len(curr_path)
                        curr_user_restart+=1
                        curr_path = []
                elif row['key']=='undo':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1

                        curr_user_num_paths+=1
                        # curr_user_nodes+=1
                        curr_user_sum_depth = curr_user_sum_depth + len(curr_path)
                        curr_user_undo+=1
                        curr_path = curr_path[:-1]

                elif row['key'] == 'confidence':
                    curr_user_data['confidence'] = row['value']

                elif row['key'] == 'solvedCorrect':
                    curr_user_data['correctness'] = row['value']

        print LOGFILE[g][5:]

    dataFile = open('userStats/participantsStats_jan20.csv', 'wb')

    dataWriter = csv.DictWriter(dataFile, fieldnames=user_data_headers, delimiter=',')
    dataWriter.writeheader()
    for record in users_data:
        # print record
        dataWriter.writerow(record)
        # for path in paths:
        #     print path
        #     print '-----------'

        # pk = []
        # for p in paths_counts:
        #     # print p
        #     pk.append(p[1]/path_counter)
        #
        # ent = stats.entropy(pk)
        # print ent
    print user_counter



def entropy_paths(subpaths = False):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        path_counter = 0.0
        taken_cells = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells

        # to_ignore = IGNORE_LIST[g];
        # to_ignore = None
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            paths = []
            paths_counts = []
            curr_path = []
            curr_user = ''
            move_count = 0.0
            for row in log_reader:
                # if row['userid'] == USERID:
                if curr_user=='':
                    curr_user = row['userid']

                if row['userid']!=curr_user:
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        add_path_count(paths_counts,copy.deepcopy(curr_path))
                        path_counter+=1
                    curr_path = []
                    curr_user = row['userid']

                elif row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        # move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                        curr_path.append([rowPos,colPos,player])
                        if len(curr_path)==1 & player == 1:
                            move_matrix[rowPos][colPos] += 1
                            move_count +=1
                        # move_counter+=1.0
                            # else:
                            #     print 'ignore'
                elif row['key']=='reset':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1
                        curr_path = []
                elif row['key']=='undo':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1
                        curr_path = curr_path[:-1]

        # print LOGFILE[g]
        # for path in paths:
        #     print path
        #     print '-----------'

        pk = []
        for p in paths_counts:
            # print p
            pk.append(p[1]/path_counter)

        ent = stats.entropy(pk)

        condition = LOGFILE[g][5:-10].replace("_",",")

        measure = 'path entropy (no subpaths)'
        if (subpaths):
            measure = 'path entropy (subpaths)'

        # print condition+',' + measure + ',' + str(ent)

        condition = condition + "," + str(subpaths)
        # if (subpaths):
        #     condition = condition + "," + "(subpaths)"
        # else:
        #     condition = condition + "_" + "with pruned cell"
        # for i in range(len(entropy_values)):
        print condition + ',' + str(ent)

        ###heatmap first moves
        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[i])):
                if (move_matrix[r][j]=='X'):
                    move_matrix[r][j] = -0.00001
                elif (move_matrix[r][j]=='O'):
                    move_matrix[r][j] = -0.00002

        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[i])):
                if (move_matrix[r][j]!=-0.00001) & (move_matrix[r][j]!=-0.00002):
                    move_matrix[r][j] = move_matrix[r][j]/move_count
        a = np.array(move_matrix)
        a = np.flip(a,0)
        print a
        heatmap = plt.pcolor(a)

        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if((a[y,x]==-1) | (a[y,x]==-0.00001)):
                    plt.text(x + 0.5, y + 0.5, 'X',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif((a[y,x]==-2) | (a[y,x]==-0.00002)):
                    plt.text(x + 0.5, y + 0.5, 'O',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif(a[y,x]!=0):
                    plt.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                     )

        fig = plt.colorbar(heatmap)
        fig_file_name = LOGFILE[g]
        fig_file_name=fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = move_matrix
        fig_file_name = fig_file_name + 'first_moves.png'
        # plt.savefig(fig_file_name)
        plt.clf()
        # print condition+',entropy, aggregated,' + str(subpaths)+ ',' + str(ent)
    return data_matrices
        # print ent


def entropy_paths_average(subpaths = False):
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        path_counter = 0.0
        taken_cells = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells

        to_ignore = IGNORE_LIST[g];
        # to_ignore = None
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            paths = []
            paths_counts = []
            entropy_values = []
            curr_path = []
            curr_user = ''
            for row in log_reader:
                # if row['userid'] == USERID:
                if curr_user=='':
                    curr_user = row['userid']

                if row['userid']!=curr_user:
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1
                    pk = []
                    for p in paths_counts:
                        # print p
                        pk.append(p[1]/path_counter)

                    ent = stats.entropy(pk)
                    if (path_counter>0):
                        entropy_values.append(ent)
                    curr_path = []
                    curr_user = row['userid']
                    paths_counts = []
                    path_counter = 0
                    paths = []

                elif row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        # move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                        curr_path.append([rowPos,colPos,player])
                        # move_counter+=1.0
                            # else:
                            #     print 'ignore'
                elif row['key']=='reset':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1
                        curr_path = []
                elif row['key']=='undo':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1
                        curr_path = curr_path[:-1]

        # print LOGFILE[g]
        avg_ent = sum(entropy_values)/len(entropy_values)

        condition = LOGFILE[g][5:-10].replace("_",",")
        measure = 'avg path entropy (no subpaths)'
        if (subpaths):
            measure = 'avg path entropy (subpaths)'

        # print condition+',' + measure + ',' + str(avg_ent)
        # print avg_ent

        condition = condition + "," + str(subpaths)
        for i in range(len(entropy_values)):
            print condition + ',' + str(entropy_values[i])


def check_participant_answer(userid):
    with open('logs/tttResults_2603.csv', 'rb') as csvfile:
        log_reader = csv.DictReader(csvfile)
        for row in log_reader:
            if row['userid']==userid:
                if row['solvedCorrect'] == 'TRUE':
                    if row['validatedCorrect'] == 'TRUE':
                        return 'validatedCorrect'
                    else:
                        return 'solvedCorrect'
        return 'wrong'


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax2(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_prob_matrix(score_matrix):
    move_prob = copy.deepcopy(score_matrix)
    all_scores = []
    sum_scores = 0.0
    counter = 0.0
    scores = np.matrix(score_matrix)

    for r in range(len(score_matrix)):
        for c in range(len(score_matrix[r])):
            if (score_matrix[r][c] != -0.00001) & (score_matrix[r][c] != -0.00002):
                all_scores.append(score_matrix[r][c])
                # move_prob[r][c] = np.exp((((score_matrix[r][c]-scores.min())*(100))/(scores.max()-scores.min())))
                # move_prob[r][c] = np.exp((((score_matrix[r][c]-scores.min())*(100))/(scores.max()-scores.min())))
                # move_prob[r][c] = (((score_matrix[r][c]-scores.min())*(100))/(scores.max()-scores.min()))
                sum_scores += move_prob[r][c]
                counter += 1.0


    for r in range(len(score_matrix)):
        for c in range(len(score_matrix[r])):
            if (move_prob[r][c] != -0.00001) & (move_prob[r][c] != -0.00002):
                if sum_scores!=0:
                    move_prob[r][c] = move_prob[r][c]/sum_scores

                else:
                    move_prob[r][c] = 1.0/counter


    return move_prob


def get_scores(score_matrix, row, col, prob_matrix):
    top_score = -10000
    second_score = -10000
    chosen_score = score_matrix[row][col]
    move_prob = copy.deepcopy(score_matrix)
    all_scores = []
    sum_scores = 0.0
    counter = 0.0
    scores = np.matrix(score_matrix)

    for r in range(len(score_matrix)):
        for c in range(len(score_matrix[r])):
            if (score_matrix[r][c] != -0.00001) & (score_matrix[r][c] != -0.00002):
                all_scores.append(score_matrix[r][c])
                move_prob[r][c] = (((score_matrix[r][c]-scores.min())*(100))/(scores.max()-scores.min()))
                sum_scores += move_prob[r][c]
                counter += 1.0

    for r in range(len(score_matrix)):
        for c in range(len(score_matrix[r])):
            if (move_prob[r][c] != -0.00001) & (move_prob[r][c] != -0.00002):
                if sum_scores!=0:
                    move_prob[r][c] = move_prob[r][c]/sum_scores

                else:
                    move_prob[r][c] = 1.0/counter

    sorted_scores = sorted(all_scores, reverse=True)
    if len(sorted_scores) == 0:
        return (0, 0, 0, 0, 0, 0)
    i = 0
    if len(sorted_scores)>1:
        while(chosen_score<sorted_scores[i]):
            i+=1
            if i == len(sorted_scores):
                break

        return (chosen_score, i, sorted_scores[0], sorted_scores[1], len(sorted_scores), move_prob[row][col])

    else:
        return (chosen_score, i, sorted_scores[0], sorted_scores[0], len(sorted_scores), move_prob[row][col])


def transition_probs_o_blind(output_file, normalized=False):
    moves_data_matrics = {}
    data_first_moves = {}

    first_moves_data_matrices = {}
    results_table = []

    scores_blocking = []
    scores_interaction = []
    scores_linear = []
    scores_density = []

    board_states = []
    boards = []
    player_list = []
    for g in range(len(LOGFILE)):
        # print g
        initial_board = copy.deepcopy(START_POSITION[g])

        path_counter = 0.0
        path_counter_subpaths = 0.0
        taken_cells = 0.0


        move_matrix = copy.deepcopy(initial_board)
        first_move_matrix = copy.deepcopy(initial_board)

        curr_move_matrix = copy.deepcopy(move_matrix)
        curr_first_move_matrix = copy.deepcopy(move_matrix)

        with open(LOGFILE[g], 'rb') as csvfile:
            print LOGFILE[g]
            cond = LOGFILE[g][5:-10].replace("_",",")
            cond = cond.split(',')
            board_size = cond[0]
            board_type = cond[1]
            condition = cond[2]
            board_name = LOGFILE[g]
            board_name=board_name[:-4]
            board_name = board_name[5:-6]

            scores_block = compute_paths_scores_for_matrix(initial_board,player='X', normalized=normalized,o_weight=0.0, exp=2, block=True)
            scores_int = compute_paths_scores_for_matrix(initial_board,player='X', normalized=normalized,o_weight=0.0, exp=2, block=False)
            scores_lin = compute_paths_scores_for_matrix(initial_board,player='X', normalized=normalized,o_weight=0.0, exp=1, block=False)
            scores_dens = compute_scores_density_new(initial_board,player='X', normalized=normalized, neighborhood_size=2)

            scores_blocking.append(str(scores_block))
            scores_interaction.append(str(scores_int))
            scores_density.append(str(scores_dens))
            scores_linear.append(str(scores_lin))

            board_states.append(str(curr_move_matrix))
            player_list.append(1)
            boards.append(board_name)

            log_reader = csv.DictReader(csvfile)
            move_number = 1
            move_stack = []
            curr_path = []
            curr_user = ''

            path_number = 0
            i = 0
            counter = 0
            for row in log_reader:
                # print i
                # i += 1
                curr_data = {}
                if curr_user == '':
                    curr_user = row['userid']
                    participant_answer = check_participant_answer(curr_user)

                if row['userid'] != curr_user:
                    participant_answer = check_participant_answer(curr_user)

                    # reset all values for next user

                    curr_path = []
                    move_stack = []
                    curr_user = row['userid']

                    path_number = 0

                    curr_move_matrix = copy.deepcopy(initial_board)

                elif row['key'] == 'clickPos':
                    counter+=1
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    # if ((curr_user == '17b4c1ea') & (rowPos==2) & (colPos==3)):
                    #     print 'here'
                    move_stack.append((rowPos, colPos))
                    player = int(row['value'][4])
                    first_move = False
                    player_type = 'X'
                    if player == 2:
                        player_type = 'O'

                    if len(curr_path) == 0:
                        path_number += 1
                    # scores computation
                    # curr_data['board_state'] = copy.deepcopy(curr_move_matrix)


                    if str(curr_move_matrix) not in board_states:
                        o_weight = 0.0
                        if player == 2:
                            o_weight = 1.0

                    # if str(curr_move_matrix) == '[[0, 2, 1, 1, 1, 2], [0, 2, 1, 2, 0, 0], [1, 1, 1, 2, 0, 0], [2, 1, 2, 2, 0, 0], [1, 1, 1, 0, 0, 0], [0, 2, 0, 2, 2, 0]]':
                    #     print 'here'

                        scores_block = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=o_weight, exp=2, block=True)
                        scores_int = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=o_weight, exp=2, block=False)
                        scores_lin = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=o_weight, exp=1,block=False)
                        scores_dens = compute_scores_density_new(curr_move_matrix,player=player_type, normalized=normalized, neighborhood_size=2)

                        scores_blocking.append(str(scores_block))
                        scores_interaction.append(str(scores_int))
                        scores_density.append(str(scores_dens))
                        scores_linear.append(str(scores_lin))

                        board_states.append(str(curr_move_matrix))
                        player_list.append(player)
                        boards.append(board_name)


                    if (curr_move_matrix[rowPos][colPos]!=1) & (curr_move_matrix[rowPos][colPos]!=2):
                        curr_move_matrix[rowPos][colPos] = player

                    if (str(curr_move_matrix) == '[[0, 2, 0, 0, 1, 0], [0, 2, 1, 2, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0], [0, 2, 1, 0, 2, 0]]'):
                        print 'here'
                    # if str(curr_move_matrix) == '[[0, 2, 1, 1, 1, 2], [0, 2, 1, 2, 0, 0], [1, 1, 1, 2, 0, 0], [2, 1, 2, 2, 0, 0], [1, 1, 1, 0, 0, 0], [0, 2, 0, 2, 2, 0]]':
                    #     print 'here'
                    if player_type == 'O':
                        player_type = 'X'
                        player = 1
                    else:
                        player_type = 'O'
                        player = 2
                    # if str(curr_move_matrix) not in board_states:
                    if str(curr_move_matrix) not in board_states:
                        o_weight = 0.0
                        if player == 2:
                            o_weight = 1.0
                        scores_block = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=o_weight, exp=2, block=True)
                        scores_int = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=o_weight, exp=2, block=False)
                        scores_lin = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=o_weight, exp=1, block=False)
                        scores_dens = compute_scores_density_new(curr_move_matrix,player=player_type, normalized=normalized, neighborhood_size=2)

                        scores_blocking.append(str(scores_block))
                        scores_interaction.append(str(scores_int))
                        scores_density.append(str(scores_dens))
                        scores_linear.append(str(scores_lin))

                        board_states.append(str(curr_move_matrix))
                        player_list.append(player)
                        boards.append(board_name)

                elif row['key'] == 'reset':
                    curr_move_matrix = copy.deepcopy(initial_board)
                    curr_path = []
                    move_stack = []

                elif row['key'] == 'undo':
                    if len(move_stack) > 0:
                        undo_move = move_stack.pop()
                        curr_move_matrix[undo_move[0]][undo_move[1]] = 0
                        curr_path = curr_path[:-1]

                    elif prev_player == 1:
                        prev_player = 2
                        # delta_x_score = ""
                        prev_x_score = None
                    else:
                        prev_player = 1

                elif row['key'] == 'start':
                    prev_time = row['time']
                    initial_time = int(prev_row['time'])

                prev_row = copy.deepcopy(row)


        # dataFile = open(board_name+'_'+output_file, 'wb')
    print boards
    print len(boards)
    print len(player_list)
    print len(board_states)
    print len(scores_blocking)
    print len(scores_interaction)
    print len(scores_density)

    trans_dict = {'board_name':boards, 'player':player_list,'board_state':board_states,'probs_blocking':scores_blocking,'probs_interaction':scores_interaction,'probs_density':scores_density,'probs_linear': scores_linear}

    transitions =pd.DataFrame(trans_dict)
    print str(len(transitions['board_state'].unique()))

    # plt.title("tt")
    if normalized:
        output_file += '_noramlized'
    else:
        output_file += '_raw'
    transitions.to_csv(output_file +'.csv')
        # fieldnames = ['board_state','player','probs_blocking','probs_interaction','probs_density']
        # dataWriter = csv.DictWriter(dataFile, fieldnames=fieldnames, delimiter=',')
        # dataWriter.writeheader()
        # for record in results_table:
        #     dataWriter.writerow(record)


def compute_transition_probs_heuristic(heuristic, board, player, normalized=False, prev_x_move=None):
    if heuristic == 'density':
        return compute_scores_density_new(board, player=player, normalized=normalized, neighborhood_size=2)
    elif heuristic == 'linear':
        return compute_paths_scores_for_matrix(board, player=player, normalized=normalized, o_weight=0.5, exp=1, block=False, interaction=False)
    elif heuristic == 'non-linear':
        return compute_paths_scores_for_matrix(board, player=player, normalized=normalized, o_weight=0.5, exp=2, block=False, interaction=False)
    elif heuristic == 'interaction':
        return compute_paths_scores_for_matrix(board, player=player, normalized=normalized, o_weight=0.5, exp=2, block=False, interaction=True)
    elif heuristic == 'blocking':
        return compute_paths_scores_for_matrix(board, player=player, normalized=normalized, o_weight=0.5, exp=2, block=True, interaction=True)
    elif heuristic == 'interaction_blind':
        o_weight = 0.0
        if player == 'O':
            o_weight = 1.0
        return compute_paths_scores_for_matrix(board, player=player, normalized=normalized, o_weight=o_weight, exp=2, block=False, interaction=True)
    elif heuristic == 'blocking_blind':
        o_weight = 0.0
        if player == 'O':
            o_weight = 1.0
        return compute_paths_scores_for_matrix(board, player=player, normalized=normalized, o_weight=o_weight, exp=2, block=True, interaction=True)
    elif heuristic.startswith('blocking_shutter'):  # if use shutter, define heuristic as blocking_shutter_k, where k is shutter size
        shutter_size = int(heuristic[len(heuristic)-1])
        return compute_paths_scores_for_matrix(board, player=player, normalized=normalized, o_weight=0.5, exp=2, block=True, interaction=True, shutter=True, shutter_size=shutter_size, prev_x_move=prev_x_move)
    elif heuristic.startswith('interaction_shutter'):  # if use shutter, define heuristic as blocking_shutter_k, where k is shutter size
        shutter_size = int(heuristic[len(heuristic)-1])
        return compute_paths_scores_for_matrix(board, player=player, normalized=normalized, o_weight=0.5, exp=2, block=False, interaction=True, shutter=True, shutter_size=shutter_size, prev_x_move=prev_x_move)


def get_move_rank(scores, move_score):
    all_scores = []
    for r in range(len(scores)):
        for c in range(len(scores[r])):
            if (scores[r][c] != -0.00001) & (scores[r][c] != -0.00002):
                all_scores.append(scores[r][c])
    sorted_scores = sorted(all_scores, reverse=True)
    # print sum(sorted_scores)
    for i in range(len(sorted_scores)):
        if move_score == sorted_scores[i]:
            return float(i)

    return -1


def fit_heuristics_conf(heuristics_list, output_file, win_scores=[100], blocking_vals=[10]):

    cogsci_participants = pd.read_csv("stats/cogsci_participants.csv")
    cogsci_part_list = cogsci_participants['userid'].unique().tolist()
    heuristic_rank_list = []
    for heuristic in heuristics_list:
        heuristic_rank_list.append(heuristic+'_rank')
    write_header = True
    for win_score in win_scores:
        # blocking_vals = [0.05, 0.1, 0.2, 0.4, 0.8]
        for blocking_val in blocking_vals:
            model.WIN_SCORE = win_score
            # model.INFINITY_O = blocking_val*win_score
            model.INFINITY_O = blocking_val
            results_table = []
            for g in range(len(LOGFILE)):
                # print g
                initial_board = copy.deepcopy(START_POSITION[g])

                move_matrix = copy.deepcopy(initial_board)

                curr_move_matrix = copy.deepcopy(move_matrix)

                with open(LOGFILE[g], 'rb') as csvfile:
                    print LOGFILE[g]
                    cond = LOGFILE[g][5:-10].replace("_",",")
                    cond = cond.split(',')
                    board_size = cond[0]
                    board_type = cond[1]
                    condition = cond[2]
                    board_name = LOGFILE[g]
                    board_name = board_name[:-4]
                    board_name = board_name[5:-6]

                    log_reader = csv.DictReader(csvfile)
                    curr_path = []
                    curr_user = ''
                    prev_x_move = None
                    user_move_count = 0.0
                    heuristics_log_likelihoods = {}
                    heuristics_move_ranks = {}
                    for heuristic in heuristics_list:
                        heuristics_log_likelihoods[heuristic] = []
                        heuristics_move_ranks[heuristic] = []
                    move_stack = []

                    i = 0
                    for row in log_reader:
                        # print i
                        # i += 1
                        curr_data = {}
                        if row['userid'] not in cogsci_part_list:
                            continue
                        if curr_user == '':
                            curr_user = row['userid']

                        if row['userid'] != curr_user:

                            # save prev user's data
                            if user_move_count > 0:
                                curr_data['userid'] = curr_user
                                curr_data['board_name'] = board_name
                                curr_data['board_size'] = board_size
                                curr_data['board_type'] = board_type
                                curr_data['condition'] = condition
                                curr_data['win_score'] = win_score
                                curr_data['blocking_score'] = blocking_val
                                curr_data['significant'] = False
                                max_log_likelihood = -10000000
                                second_log_likelihood = -10000000
                                max_heuristic = None
                                second_heuristic = None
                                for heuristic in heuristics_list:
                                    avg_log_likelihood = sum(heuristics_log_likelihoods[heuristic])/len(heuristics_log_likelihoods[heuristic])
                                    avg_rank = sum(heuristics_move_ranks[heuristic])/len(heuristics_move_ranks[heuristic])
                                    curr_data[heuristic] = avg_log_likelihood
                                    curr_data[heuristic+'_rank'] = avg_rank
                                    if avg_log_likelihood > max_log_likelihood:
                                        second_heuristic = max_heuristic
                                        max_heuristic = heuristic
                                        max_log_likelihood = avg_log_likelihood
                                    elif avg_log_likelihood > second_log_likelihood:
                                        second_heuristic = heuristic
                                        second_log_likelihood = avg_log_likelihood

                                # is stats sig diff
                                p_val = bootstrap_t_pvalue(np.asarray(heuristics_log_likelihoods[max_heuristic]), np.asarray(heuristics_log_likelihoods[second_heuristic]))
                                curr_data['significant'] = p_val
                                # if p_val < 0.05:
                                #     curr_data['significant'] = True
                                #     print 'sig'
                                # else:
                                #     print 'not'
                                curr_data['fitted_heuristic'] = max_heuristic

                                results_table.append(curr_data)
                            # reset all values for next user
                            curr_path = []
                            prev_x_move = None
                            heuristics_log_likelihoods = {}
                            heuristics_move_ranks = {}
                            for heuristic in heuristics_list:
                                heuristics_log_likelihoods[heuristic] = []
                                heuristics_move_ranks[heuristic] = []
                            curr_user = row['userid']

                            move_stack = []
                            user_move_count = 0.0
                            curr_move_matrix = copy.deepcopy(initial_board)

                        elif row['key'] == 'clickPos':
                            user_move_count += 1
                            rowPos = int(row['value'][0])
                            colPos = int(row['value'][2])

                            player = int(row['value'][4])

                            player_type = 'X'
                            if player == 2:
                                player_type = 'O'
                                # continue
                            # if player == 1:
                            for heuristic in heuristics_list:
                                # print heuristic
                                probs = compute_transition_probs_heuristic(heuristic, curr_move_matrix, player_type, normalized=True, prev_x_move=prev_x_move)
                                prob_move = probs[rowPos][colPos]
                                heuristics_move_ranks[heuristic].append(get_move_rank(probs, prob_move))
                                if prob_move <= 0.0:
                                    prob_move = 0.0001
                                heuristics_log_likelihoods[heuristic].append(math.log(prob_move))


                            if player == 1:
                                prev_x_move = [rowPos, colPos]

                            if (curr_move_matrix[rowPos][colPos] != 1) & (curr_move_matrix[rowPos][colPos] != 2):
                                curr_move_matrix[rowPos][colPos] = player
                                curr_path.append([rowPos, colPos, player])
                                move_stack.append((rowPos, colPos))

                                if len(curr_path) == 1:
                                    first_move = True
                            else:
                                print 'weird'

                        elif row['key'] == 'reset':
                            curr_move_matrix = copy.deepcopy(initial_board)
                            curr_path = []
                            prev_x_move = None
                            move_stack = []

                        elif row['key'] == 'undo':
                            if len(move_stack) > 0:
                                undo_move = move_stack.pop()
                                curr_move_matrix[undo_move[0]][undo_move[1]] = 0
                                curr_path = curr_path[:-1]
                                prev_x_move = None
                                if len(curr_path) > 0:
                                    if len(curr_path) % 2 == 0:  # last move is O move, get one before
                                        prev_x_move = curr_path[len(curr_path)-2]
                                    else:  # last move is X move, get it
                                        prev_x_move = curr_path[len(curr_path)-1]

                            else:
                                print 'problem undo'
                    # save last user
                    if user_move_count > 0:
                        curr_data['userid'] = curr_user
                        curr_data['board_name'] = board_name
                        curr_data['board_size'] = board_size
                        curr_data['board_type'] = board_type
                        curr_data['condition'] = condition
                        curr_data['win_score'] = win_score
                        curr_data['blocking_score'] = blocking_val
                        curr_data['significant'] = False
                        max_log_likelihood = -10000000
                        second_log_likelihood = -10000000
                        max_heuristic = None
                        second_heuristic = None
                        for heuristic in heuristics_list:
                            avg_log_likelihood = sum(heuristics_log_likelihoods[heuristic])/len(heuristics_log_likelihoods[heuristic])
                            avg_rank = sum(heuristics_move_ranks[heuristic])/len(heuristics_move_ranks[heuristic])
                            curr_data[heuristic] = avg_log_likelihood
                            curr_data[heuristic+'_rank'] = avg_rank
                            if avg_log_likelihood > max_log_likelihood:
                                second_heuristic = max_heuristic
                                max_heuristic = heuristic
                                max_log_likelihood = avg_log_likelihood
                            elif avg_log_likelihood > second_log_likelihood:
                                second_heuristic = heuristic
                                second_log_likelihood = avg_log_likelihood

                            # is stats sig diff
                        p_val = bootstrap_t_pvalue(np.asarray(heuristics_log_likelihoods[max_heuristic]), np.asarray(heuristics_log_likelihoods[second_heuristic]))
                        curr_data['significant'] = p_val
                        # if p_val < 0.05:
                        #     curr_data['significant'] = True
                        #     print 'sig'
                        # else:
                        #     print 'not'
                        curr_data['fitted_heuristic'] = max_heuristic

                        results_table.append(curr_data)

            dataFile = open(output_file, 'ab')
            fieldnames = ['userid','board_name','board_size','board_type','condition','fitted_heuristic', 'blocking_score', 'win_score','significant']
            fieldnames.extend(heuristics_list)
            fieldnames.extend(heuristic_rank_list)
            dataWriter = csv.DictWriter(dataFile, fieldnames=fieldnames, delimiter=',')
            if write_header:
                dataWriter.writeheader()
                write_header = False
            for record in results_table:
                dataWriter.writerow(record)
            dataFile.close()


def fit_heuristics_by_move(heuristics_list, output_file, win_scores=[100], blocking_vals=[10]):

    cogsci_participants = pd.read_csv("stats/cogsci_participants.csv")
    cogsci_part_list = cogsci_participants['userid'].unique().tolist()
    heuristic_rank_list = []
    for heuristic in heuristics_list:
        heuristic_rank_list.append(heuristic+'_rank')
    write_header = True
    for win_score in win_scores:
        # blocking_vals = [0.05, 0.1, 0.2, 0.4, 0.8]
        for blocking_val in blocking_vals:
            model.WIN_SCORE = win_score
            # model.INFINITY_O = blocking_val*win_score
            model.INFINITY_O = blocking_val
            results_table = []
            for g in range(len(LOGFILE)):
                # print g
                initial_board = copy.deepcopy(START_POSITION[g])

                move_matrix = copy.deepcopy(initial_board)

                curr_move_matrix = copy.deepcopy(move_matrix)

                with open(LOGFILE[g], 'rb') as csvfile:
                    print LOGFILE[g]
                    cond = LOGFILE[g][5:-10].replace("_",",")
                    cond = cond.split(',')
                    board_size = cond[0]
                    board_type = cond[1]
                    condition = cond[2]
                    board_name = LOGFILE[g]
                    board_name = board_name[:-4]
                    board_name = board_name[5:-6]

                    log_reader = csv.DictReader(csvfile)
                    curr_path = []
                    num_moves_user = 0
                    path_number = 0
                    prob_path = {}

                    curr_user = ''
                    prev_x_move = None
                    user_move_count = 0.0
                    heuristics_log_likelihoods = {}
                    heuristics_move_ranks = {}
                    for heuristic in heuristics_list:
                        heuristics_log_likelihoods[heuristic] = []
                        heuristics_move_ranks[heuristic] = []
                        prob_path[heuristic] = []
                    move_stack = []

                    i = 0
                    for row in log_reader:
                        # print i
                        # i += 1
                        curr_data = {}
                        if row['userid'] not in cogsci_part_list:
                            continue
                        # if row['userid'] != '17e120dd':
                        #     continue
                        if curr_user == '':
                            curr_user = row['userid']

                        if row['userid'] != curr_user:

                            # # save prev user's data
                            # if user_move_count > 0:
                            #     curr_data['userid'] = curr_user
                            #     curr_data['board_name'] = board_name
                            #     curr_data['board_size'] = board_size
                            #     curr_data['board_type'] = board_type
                            #     curr_data['condition'] = condition
                            #     curr_data['win_score'] = win_score
                            #     curr_data['blocking_score'] = blocking_val
                            #     curr_data['significant'] = False
                            #     max_log_likelihood = -10000000
                            #     second_log_likelihood = -10000000
                            #     max_heuristic = None
                            #     second_heuristic = None
                            #     for heuristic in heuristics_list:
                            #         avg_log_likelihood = sum(heuristics_log_likelihoods[heuristic])/len(heuristics_log_likelihoods[heuristic])
                            #         avg_rank = sum(heuristics_move_ranks[heuristic])/len(heuristics_move_ranks[heuristic])
                            #         curr_data[heuristic] = avg_log_likelihood
                            #         curr_data[heuristic+'_rank'] = avg_rank
                            #         if avg_log_likelihood > max_log_likelihood:
                            #             second_heuristic = max_heuristic
                            #             max_heuristic = heuristic
                            #             max_log_likelihood = avg_log_likelihood
                            #         elif avg_log_likelihood > second_log_likelihood:
                            #             second_heuristic = heuristic
                            #             second_log_likelihood = avg_log_likelihood
                            #
                            #     # is stats sig diff
                            #     p_val = bootstrap_t_pvalue(np.asarray(heuristics_log_likelihoods[max_heuristic]), np.asarray(heuristics_log_likelihoods[second_heuristic]))
                            #     curr_data['significant'] = p_val
                            #     # if p_val < 0.05:
                            #     #     curr_data['significant'] = True
                            #     #     print 'sig'
                            #     # else:
                            #     #     print 'not'
                            #     curr_data['fitted_heuristic'] = max_heuristic
                            #
                            #     results_table.append(curr_data)
                            # reset all values for next user
                            curr_path = []
                            prob_path = {}

                            num_moves_user = 0
                            path_number = 0
                            prev_x_move = None
                            heuristics_log_likelihoods = {}
                            heuristics_move_ranks = {}
                            for heuristic in heuristics_list:
                                heuristics_log_likelihoods[heuristic] = []
                                heuristics_move_ranks[heuristic] = []
                                prob_path[heuristic] = []
                            curr_user = row['userid']

                            move_stack = []
                            user_move_count = 0.0
                            curr_move_matrix = copy.deepcopy(initial_board)

                        elif row['key'] == 'clickPos':

                            user_move_count += 1
                            rowPos = int(row['value'][0])
                            colPos = int(row['value'][2])

                            player = int(row['value'][4])

                            player_type = 'X'
                            if player == 2:
                                player_type = 'O'
                                # continue
                            # if player == 1:

                            if player == 1:
                                prev_x_move = [rowPos, colPos]

                            num_moves_user += 1

                            for heuristic in heuristics_list:
                                # curr_data = {}
                                # print heuristic
                                probs = compute_transition_probs_heuristic(heuristic, curr_move_matrix, player_type, normalized=True, prev_x_move=prev_x_move)
                                prob_move = probs[rowPos][colPos]

                                max_prob = np.max(probs)
                                move_prob_ratio = prob_move/max_prob

                                move_rank = get_move_rank(probs, prob_move)

                                if prob_move <= 0.0:
                                    prob_move = 0.0001



                                curr_data['userid'] = curr_user
                                curr_data['board_name'] = board_name
                                curr_data['board_size'] = board_size
                                curr_data['board_type'] = board_type
                                curr_data['condition'] = condition
                                curr_data['player'] = player
                                curr_data['win_score'] = win_score
                                curr_data['blocking_score'] = blocking_val
                                curr_data['heuristic'] = heuristic
                                curr_data['prob_move'] = prob_move

                                curr_data['prob_move_to_best_ratio'] = move_prob_ratio
                                curr_data['log_move'] = math.log(prob_move)
                                curr_data['rank_move'] = move_rank
                                curr_data['move_number'] = num_moves_user
                                # curr_data['move_number_in_path'] = len(curr_path)
                                curr_data['path_number'] = path_number
                                path_copy = copy.deepcopy(curr_path)
                                path_copy.append([rowPos, colPos, player])
                                curr_data['move_number_in_path'] = len(curr_path)+1
                                curr_data['path'] = copy.deepcopy(path_copy)
                                if len(prob_path[heuristic]) == 0:
                                    prob_path[heuristic].append(prob_move)
                                else:
                                    prob_path[heuristic].append(prob_path[heuristic][len(prob_path[heuristic])-1] * prob_move)
                                curr_data['prob_path'] = prob_path[heuristic][len(prob_path[heuristic])-1]

                                results_table.append(copy.deepcopy(curr_data))


                            if (curr_move_matrix[rowPos][colPos] != 1) & (curr_move_matrix[rowPos][colPos] != 2):
                                curr_move_matrix[rowPos][colPos] = player
                                curr_path.append([rowPos, colPos, player])
                                move_stack.append((rowPos, colPos))

                                if len(curr_path) == 1:
                                    first_move = True
                                    path_number += 1

                            else:
                                print 'weird'




                        elif row['key'] == 'reset':
                            curr_move_matrix = copy.deepcopy(initial_board)
                            curr_path = []
                            prev_x_move = None
                            move_stack = []
                            prob_path = {}
                            for h in heuristics_list:
                                prob_path[h] = []


                        elif row['key'] == 'undo':
                            if len(move_stack) > 0:
                                undo_move = move_stack.pop()
                                curr_move_matrix[undo_move[0]][undo_move[1]] = 0
                                curr_path = curr_path[:-1]
                                for h in heuristics_list:
                                    prob_path[h] = prob_path[h][:-1]
                                prev_x_move = None
                                if len(curr_path) > 0:
                                    if len(curr_path) % 2 == 0:  # last move is O move, get one before
                                        prev_x_move = curr_path[len(curr_path)-2]
                                    else:  # last move is X move, get it
                                        prev_x_move = curr_path[len(curr_path)-1]

                            else:
                                print 'problem undo'

            dataFile = open(output_file, 'ab')
            fieldnames = ['userid','board_name','board_size','board_type','condition', 'player', 'blocking_score', 'win_score','heuristic','prob_move','prob_move_to_best_ratio', 'log_move','rank_move','move_number','move_number_in_path','path_number','prob_path','path']
            # fieldnames.extend(heuristics_list)
            # fieldnames.extend(heuristic_rank_list)
            dataWriter = csv.DictWriter(dataFile, fieldnames=fieldnames, delimiter=',')
            if write_header:
                dataWriter.writeheader()
                write_header = False
            for record in results_table:
                dataWriter.writerow(record)
            dataFile.close()



def fit_heuristics(heuristics_list, output_file, win_scores=[100], blocking_vals=[10]):

    write_header = True
    for win_score in win_scores:
        # blocking_vals = [0.05, 0.1, 0.2, 0.4, 0.8]
        for blocking_val in blocking_vals:
            model.WIN_SCORE = win_score
            # model.INFINITY_O = blocking_val*win_score
            model.INFINITY_O = blocking_val
            results_table = []
            for g in range(len(LOGFILE)):
                # print g
                initial_board = copy.deepcopy(START_POSITION[g])

                move_matrix = copy.deepcopy(initial_board)

                curr_move_matrix = copy.deepcopy(move_matrix)

                with open(LOGFILE[g], 'rb') as csvfile:
                    print LOGFILE[g]
                    cond = LOGFILE[g][5:-10].replace("_",",")
                    cond = cond.split(',')
                    board_size = cond[0]
                    board_type = cond[1]
                    condition = cond[2]
                    board_name = LOGFILE[g]
                    board_name = board_name[:-4]
                    board_name = board_name[5:-6]

                    log_reader = csv.DictReader(csvfile)
                    curr_path = []
                    curr_user = ''
                    prev_x_move = None
                    user_move_count = 0.0
                    heuristics_log_likelihoods = {}
                    for heuristic in heuristics_list:
                        heuristics_log_likelihoods[heuristic] = 0.0

                    move_stack = []

                    i = 0
                    for row in log_reader:
                        # print i
                        # i += 1
                        curr_data = {}
                        if curr_user == '':
                            curr_user = row['userid']

                        if row['userid'] != curr_user:

                            # save prev user's data
                            if user_move_count > 0:
                                curr_data['userid'] = curr_user
                                curr_data['board_name'] = board_name
                                curr_data['board_size'] = board_size
                                curr_data['board_type'] = board_type
                                curr_data['condition'] = condition
                                curr_data['win_score'] = win_score
                                curr_data['blocking_score'] = blocking_val
                                max_log_likelihood = -10000000
                                max_heuristic = None
                                for heuristic in heuristics_list:
                                    avg_log_likelihood = heuristics_log_likelihoods[heuristic] / user_move_count
                                    curr_data[heuristic] = avg_log_likelihood
                                    if avg_log_likelihood > max_log_likelihood:
                                        max_heuristic = heuristic
                                        max_log_likelihood = avg_log_likelihood
                                curr_data['fitted_heuristic'] = max_heuristic
                                results_table.append(curr_data)
                            # reset all values for next user
                            curr_path = []
                            prev_x_move = None
                            heuristics_log_likelihoods = {}
                            for heuristic in heuristics_list:
                                heuristics_log_likelihoods[heuristic] = 0.0
                            curr_user = row['userid']

                            move_stack = []
                            user_move_count = 0.0
                            curr_move_matrix = copy.deepcopy(initial_board)

                        elif row['key'] == 'clickPos':
                            user_move_count += 1
                            rowPos = int(row['value'][0])
                            colPos = int(row['value'][2])

                            player = int(row['value'][4])

                            player_type = 'X'
                            if player == 2:
                                player_type = 'O'

                            for heuristic in heuristics_list:
                                # print heuristic
                                probs = compute_transition_probs_heuristic(heuristic, curr_move_matrix, player_type, normalized=True, prev_x_move=prev_x_move)
                                prob_move = probs[rowPos][colPos]
                                if prob_move <= 0.0:
                                    prob_move = 0.0001
                                heuristics_log_likelihoods[heuristic] += math.log(prob_move)

                            if player == 1:
                                prev_x_move = [rowPos, colPos]

                            if (curr_move_matrix[rowPos][colPos] != 1) & (curr_move_matrix[rowPos][colPos] != 2):
                                curr_move_matrix[rowPos][colPos] = player
                                curr_path.append([rowPos, colPos, player])
                                move_stack.append((rowPos, colPos))

                                if len(curr_path) == 1:
                                    first_move = True
                            else:
                                print 'weird'

                        elif row['key'] == 'reset':
                            curr_move_matrix = copy.deepcopy(initial_board)
                            curr_path = []
                            prev_x_move = None
                            move_stack = []

                        elif row['key'] == 'undo':
                            if len(move_stack) > 0:
                                undo_move = move_stack.pop()
                                curr_move_matrix[undo_move[0]][undo_move[1]] = 0
                                curr_path = curr_path[:-1]
                                prev_x_move = None
                                if len(curr_path) > 0:
                                    if len(curr_path) % 2 == 0:  # last move is O move, get one before
                                        prev_x_move = curr_path[len(curr_path)-2]
                                    else:  # last move is X move, get it
                                        prev_x_move = curr_path[len(curr_path)-1]

                            else:
                                print 'problem undo'
                    # save last user
                    if user_move_count > 0:
                        curr_data['userid'] = curr_user
                        curr_data['board_name'] = board_name
                        curr_data['board_size'] = board_size
                        curr_data['board_type'] = board_type
                        curr_data['condition'] = condition
                        curr_data['win_score'] = win_score
                        curr_data['blocking_score'] = blocking_val
                        max_log_likelihood = -10000000
                        max_heuristic = None
                        for heuristic in heuristics_list:
                            avg_log_likelihood = heuristics_log_likelihoods[heuristic] / user_move_count
                            curr_data[heuristic] = avg_log_likelihood
                            if avg_log_likelihood > max_log_likelihood:
                                max_heuristic = heuristic
                                max_log_likelihood = avg_log_likelihood
                        curr_data['fitted_heuristic'] = max_heuristic
                        results_table.append(curr_data)

            dataFile = open(output_file, 'ab')
            fieldnames = ['userid','board_name','board_size','board_type','condition','fitted_heuristic', 'blocking_score', 'win_score']
            fieldnames.extend(heuristics_list)
            dataWriter = csv.DictWriter(dataFile, fieldnames=fieldnames, delimiter=',')
            if write_header:
                dataWriter.writeheader()
                write_header = False
            for record in results_table:
                dataWriter.writerow(record)
            dataFile.close()

def transition_probs(output_file, normalized=False):
    moves_data_matrics = {}
    data_first_moves = {}

    first_moves_data_matrices = {}
    results_table = []
    scores_blocking = []
    scores_interaction = []
    scores_linear = []
    scores_density = []

    board_states = []
    boards = []
    player_list = []
    for g in range(len(LOGFILE)):
        # print g
        initial_board = copy.deepcopy(START_POSITION[g])

        path_counter = 0.0
        path_counter_subpaths = 0.0
        taken_cells = 0.0


        move_matrix = copy.deepcopy(initial_board)
        first_move_matrix = copy.deepcopy(initial_board)

        curr_move_matrix = copy.deepcopy(move_matrix)
        curr_first_move_matrix = copy.deepcopy(move_matrix)



        with open(LOGFILE[g], 'rb') as csvfile:
            print LOGFILE[g]
            cond = LOGFILE[g][5:-10].replace("_",",")
            cond = cond.split(',')
            board_size = cond[0]
            board_type = cond[1]
            condition = cond[2]
            board_name = LOGFILE[g]
            board_name=board_name[:-4]
            board_name = board_name[5:-6]

            scores_block = compute_paths_scores_for_matrix(initial_board,player='X', normalized=normalized,o_weight=0.5, exp=2, block=True)
            scores_int = compute_paths_scores_for_matrix(initial_board,player='X', normalized=normalized,o_weight=0.5, exp=2, block=False)
            scores_lin = compute_paths_scores_for_matrix(initial_board,player='X', normalized=normalized,o_weight=0.5, exp=1,block=False, interaction=False)
            scores_dens = compute_scores_density_new(initial_board,player='X', normalized=normalized, neighborhood_size=2)
            scores_block_density = scores_block
            scores_o_blind = scores_block

            scores_int_density = scores_int
            scores_lin_dens = scores_lin
            # probs_block = get_prob_matrix(scores_block)
            # probs_int = get_prob_matrix(scores_int)
            # probs_dens = get_prob_matrix(scores_dens)
            probs_block = scores_block
            probs_int = scores_int
            probs_dens = scores_dens

            scores_blocking.append(str(probs_block))
            scores_interaction.append(str(probs_int))
            scores_density.append(str(probs_dens))
            scores_linear.append(str(scores_lin))

            board_states.append(str(curr_move_matrix))
            player_list.append(1)
            boards.append(board_name)

            log_reader = csv.DictReader(csvfile)
            move_number = 1
            move_stack = []
            curr_path = []
            curr_user = ''



            path_number = 0
            i = 0
            counter = 0
            for row in log_reader:
                # print i
                # i += 1
                curr_data = {}
                if curr_user == '':
                    curr_user = row['userid']
                    participant_answer = check_participant_answer(curr_user)

                if row['userid'] != curr_user:
                    participant_answer = check_participant_answer(curr_user)

                    # reset all values for next user

                    curr_path = []
                    move_stack = []
                    curr_user = row['userid']

                    path_number = 0

                    curr_move_matrix = copy.deepcopy(initial_board)

                elif row['key'] == 'clickPos':
                    counter+=1
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    # if ((curr_user == '17b4c1ea') & (rowPos==2) & (colPos==3)):
                    #     print 'here'
                    move_stack.append((rowPos, colPos))
                    player = int(row['value'][4])
                    first_move = False
                    player_type = 'X'
                    if player == 2:
                        player_type = 'O'

                    if len(curr_path) == 0:
                        path_number += 1
                    # scores computation
                    # curr_data['board_state'] = copy.deepcopy(curr_move_matrix)

                    if (str(curr_move_matrix) == '[[1, 2, 0, 0, 1, 0], [0, 2, 1, 2, 0, 0], [0, 1, 2, 1, 0, 0], [0, 1, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0], [0, 2, 1, 2, 2, 0]]'):
                        print 'here'

                    if str(curr_move_matrix) not in board_states:
                        scores_block = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=0.5, exp=2, block=True)
                        scores_int = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=0.5, exp=2, block=False)
                        scores_lin = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=0.5, exp=1,block=False, interaction=False)
                        scores_dens = compute_scores_density_new(curr_move_matrix,player=player_type, normalized=normalized, neighborhood_size=2)

                        scores_block_density = scores_block
                        scores_int_density = scores_int
                        scores_lin_dens = scores_lin
                        if player == 2:
                            scores_block_density = scores_dens
                            scores_int_density = scores_dens
                            scores_lin_dens = scores_dens
                        # else:
                        #     scores_o_blind = scores_block
                        # probs_block = get_prob_matrix(scores_block)
                        # probs_int = get_prob_matrix(scores_int)
                        # probs_dens = get_prob_matrix(scores_dens)
                        probs_block = scores_block
                        probs_int = scores_int
                        probs_dens = scores_dens
                        # print np.sum(probs_block)
                        # print np.sum(probs_int)
                        # print np.sum(probs_dens)
                        # if np.sum(probs_block) < 0.99:
                        #     print 'wow'
                        # if np.sum(probs_int) < 0.99:
                        #     print 'wow'
                        # if np.sum(probs_dens) < 0.99:
                        #     print 'wow'
                        scores_blocking.append(str(probs_block))
                        scores_interaction.append(str(probs_int))
                        scores_linear.append(str(scores_lin))

                        board_states.append(str(curr_move_matrix))
                        player_list.append(player)
                        boards.append(board_name)


                    if (curr_move_matrix[rowPos][colPos]!=1) & (curr_move_matrix[rowPos][colPos]!=2):
                        curr_move_matrix[rowPos][colPos] = player

                    if (str(curr_move_matrix) == '[[1, 2, 0, 0, 1, 0], [0, 2, 1, 2, 0, 0], [0, 1, 2, 1, 0, 0], [0, 1, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0], [0, 2, 1, 2, 2, 0]]'):
                        print 'here'
                    if player_type == 'O':
                        player_type = 'X'
                        player = 1
                    else:
                        player_type = 'O'
                        player = 2
                    # if str(curr_move_matrix) not in board_states:
                    if str(curr_move_matrix) not in board_states:
                        scores_block = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=0.5, exp=2, block=True)
                        scores_int = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=0.5, exp=2, block=False)
                        scores_lin = compute_paths_scores_for_matrix(curr_move_matrix,player=player_type, normalized=normalized,o_weight=0.5, exp=1,block=False, interaction=False)
                        scores_dens = compute_scores_density_new(curr_move_matrix,player=player_type, normalized=normalized, neighborhood_size=2)

                        scores_block_density = scores_block
                        scores_int_density = scores_int
                        scores_lin_dens = scores_lin
                        if player == 2:
                            scores_block_density = scores_dens
                            scores_int_density = scores_dens
                            scores_lin_dens = scores_dens
                        # else:
                        #     scores_o_blind = scores_block
                        # probs_block = get_prob_matrix(scores_block)
                        # probs_int = get_prob_matrix(scores_int)
                        # probs_dens = get_prob_matrix(scores_dens)
                        probs_block = scores_block
                        probs_int = scores_int
                        probs_dens = scores_dens
                        # print np.sum(probs_block)
                        # print np.sum(probs_int)
                        # print np.sum(probs_dens)
                        # if np.sum(probs_block) < 0.99:
                        #     print 'wow'
                        # if np.sum(probs_int) < 0.99:
                        #     print 'wow'
                        # if np.sum(probs_dens) < 0.99:
                        #     print 'wow'
                        scores_blocking.append(str(probs_block))
                        scores_interaction.append(str(probs_int))
                        scores_density.append(str(probs_dens))
                        scores_linear.append(str(scores_lin))

                        board_states.append(str(curr_move_matrix))
                        player_list.append(player)
                        boards.append(board_name)

                elif row['key'] == 'reset':
                    curr_move_matrix = copy.deepcopy(initial_board)
                    curr_path = []
                    move_stack = []

                elif row['key'] == 'undo':
                    if len(move_stack) > 0:
                        undo_move = move_stack.pop()
                        curr_move_matrix[undo_move[0]][undo_move[1]] = 0
                        curr_path = curr_path[:-1]

                    elif prev_player == 1:
                        prev_player = 2
                        # delta_x_score = ""
                        prev_x_score = None
                    else:
                        prev_player = 1


                elif row['key'] == 'start':
                    prev_time = row['time']
                    initial_time = int(prev_row['time'])

                prev_row = copy.deepcopy(row)


        # dataFile = open(board_name+'_'+output_file, 'wb')
    print boards
    print len(boards)
    print len(player_list)
    print len(board_states)
    print len(scores_blocking)
    print len(scores_interaction)
    print len(scores_density)



    trans_dict = {'board_name':boards, 'player':player_list,'board_state':board_states,'probs_blocking':scores_blocking,'probs_interaction':scores_interaction,'probs_density':scores_density, 'probs_linear': scores_linear}

    transitions =pd.DataFrame(trans_dict)
    print str(len(transitions['board_state'].unique()))

    # plt.title("tt")
    if normalized:
        output_file += 'normalized'
    else:
        output_file += 'raw'
    transitions.to_csv(output_file +'.csv')
        # fieldnames = ['board_state','player','probs_blocking','probs_interaction','probs_density']
        # dataWriter = csv.DictWriter(dataFile, fieldnames=fieldnames, delimiter=',')
        # dataWriter.writeheader()
        # for record in results_table:
        #     dataWriter.writerow(record)




def moves_stats(output_file):
    moves_data_matrics = {}
    data_first_moves = {}

    first_moves_data_matrices = {}
    results_table = []
    for g in range(len(LOGFILE)):
        # print g
        initial_board = copy.deepcopy(START_POSITION[g])

        path_counter = 0.0
        path_counter_subpaths = 0.0
        taken_cells = 0.0

        # for i in range(len(initial_board)):
        #     for j in range(len(initial_board[i])):
        #         if ((initial_board[i][j]!=1) & (initial_board[i][j]!=2)):
        #             initial_board[i][j] = int(initial_board[i][j])
        #         elif (initial_board[i][j]==1):
        #             initial_board[i][j]='X'
        #             taken_cells+=1
        #         elif (initial_board[i][j]==2):
        #             initial_board[i][j]='O'
        #             taken_cells+=1

        move_matrix = copy.deepcopy(initial_board)
        first_move_matrix = copy.deepcopy(initial_board)

        curr_move_matrix = copy.deepcopy(move_matrix)
        curr_first_move_matrix = copy.deepcopy(move_matrix)



        with open(LOGFILE[g], 'rb') as csvfile:
            print LOGFILE[g]
            cond = LOGFILE[g][5:-10].replace("_",",")
            cond = cond.split(',')
            board_size = cond[0]
            board_type = cond[1]
            condition = cond[2]
            board_name = LOGFILE[g]
            board_name=board_name[:-4]
            board_name = board_name[5:-6]

            log_reader = csv.DictReader(csvfile)
            move_number = 1
            move_stack = []
            curr_path = []
            curr_user = ''
            prev_time = None
            prev_click_time = None
            prev_action = None
            initial_time = None
            prev_x_score = 0
            prev_o_score = None
            delta_x_score = 0
            delta_o_score = None
            prev_player = None
            path_prob = 1.0
            path_number = 0
            i = 0
            for row in log_reader:
                # print i
                # i += 1
                curr_data = {}
                if curr_user == '':
                    curr_user = row['userid']
                    participant_answer = check_participant_answer(curr_user)

                if row['userid'] != curr_user:
                    # participant_answer = check_participant_answer(curr_user)

                    # reset all values for next user
                    prev_time = None
                    prev_action = None
                    prev_click_time = None
                    prev_row = None
                    first_move_for_player = True
                    move_number = 1
                    curr_path = []
                    move_stack = []
                    curr_user = row['userid']
                    participant_answer = check_participant_answer(curr_user)
                    initial_time = None
                    prev_x_score = 0
                    prev_o_score = None
                    delta_x_score = 0
                    delta_o_score = None
                    prev_player = None
                    path_prob = 1.0
                    path_number = 0

                    curr_move_matrix = copy.deepcopy(initial_board)

                elif row['key'] == 'clickPos':
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    move_stack.append((rowPos, colPos))
                    player = int(row['value'][4])
                    first_move = False
                    player_type = 'X'
                    if player == 2:
                        player_type = 'O'

                    if ((len(curr_path) == 0) | (prev_action == 'undo')):
                        path_number += 1
                    # scores computation

                    scores = compute_scores_layers_for_matrix(curr_move_matrix,player=player_type, normalized=False,o_weight=0.5, exp=2, neighborhood_size=2, block=True)
                    probs = compute_scores_layers_for_matrix(curr_move_matrix,player=player_type, normalized=True,o_weight=0.5, exp=2, neighborhood_size=2, block=True, lamb=None)
                    scores_data = get_scores(scores, rowPos, colPos, probs)
                    move_prob = scores_data[5]
                    # print board_name
                    if move_prob == 0 :
                        move_prob += 0.0001
                    # if abs(move_prob) > 1:
                    #     print 'what'

                    path_prob = move_prob*path_prob

                    delta_score = ""
                    if player == 1:
                        if prev_x_score != None:
                            delta_x_score = scores_data[0] - prev_x_score
                            delta_score = delta_x_score
                        prev_x_score = scores_data[0]

                    elif player == 2:
                        if prev_x_score != None:
                            delta_x_score = (-1*scores_data[0]) - prev_x_score
                            delta_score = delta_x_score
                            prev_x_score = -1*scores_data[0]
                        # if prev_o_score != None:
                        #     delta_o_score = scores_data[0] - prev_o_score
                        #     delta_score = delta_o_score
                        # prev_o_score = scores_data[0]
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    curr_data['board_state'] = copy.deepcopy(curr_move_matrix)
                    if (curr_move_matrix[rowPos][colPos]!=1) & (curr_move_matrix[rowPos][colPos]!=2):
                        curr_move_matrix[rowPos][colPos] = player
                        curr_path.append([rowPos, colPos, player])

                        if len(curr_path) == 1:
                            first_move = True
                    else:
                        print 'weird'
                    time_between = 0
                    if prev_time == None:
                        if prev_row != None:
                            # print 'why'
                            prev_time = prev_row['time']
                            initial_time = int(prev_row['time'])
                    if prev_time == None:
                        print row
                    time_between_action = (int(row['time']) - int(prev_time))/1000.0

                    time_between_click = 0
                    if prev_click_time != None:
                        time_between_click = (int(row['time']) - int(prev_click_time))/1000.0



                    curr_data['userid'] = curr_user

                    curr_data['board_size'] = board_size
                    curr_data['board_type'] = board_type
                    curr_data['condition'] = condition
                    curr_data['board_name'] = board_name
                    curr_data['action'] = 'click'

                    curr_data['score_move'] = scores_data[0]

                    curr_data['move_ranking'] = scores_data[1]
                    curr_data['top_possible_score'] = scores_data[2]
                    curr_data['second_possible_score'] = scores_data[3]
                    curr_data['num_possible_moves'] = scores_data[4]
                    curr_data['delta_score'] = delta_score

                    curr_data['solved'] = participant_answer
                    curr_data['player'] = player
                    curr_data['position'] = str(rowPos) + '_' + str(colPos)
                    curr_data['time'] = row['time']

                    curr_data['time_rel'] = int(row['time']) - initial_time
                    curr_data['time_prev_action'] = prev_time
                    curr_data['time_from_action'] = time_between_action
                    curr_data['time_from_click'] = time_between_click
                    curr_data['prev_action'] = prev_action
                    curr_data['first_move_in_path'] = first_move
                    curr_data['move_number'] = move_number
                    curr_data['move_number_in_path'] = len(curr_path)
                    curr_data['path'] = curr_path[:-1]
                    curr_data['path_after'] = curr_path
                    curr_data['move_prob'] = move_prob
                    curr_data['path_prob'] = path_prob
                    curr_data['path_number'] = path_number

                    player_type = 'X'
                    if player == 1:
                        player_type = 'O'

                    scores = compute_scores_layers_for_matrix(curr_move_matrix,player=player_type, normalized=False,o_weight=0.5, exp=2, neighborhood_size=2, block=True)
                    probs = compute_scores_layers_for_matrix(curr_move_matrix,player=player_type, normalized=True,o_weight=0.5, exp=2, neighborhood_size=2, block=True, lamb=None)
                    scores_data = get_scores(scores, 0, 0, probs)
                    score_x = scores_data[2]
                    if player == 1:
                        score_x = -1*scores_data[0]

                    curr_data['potential_score'] = score_x
                    results_table.append(copy.deepcopy(curr_data))

                    move_number += 1
                    prev_action = 'click'
                    prev_time = row['time']
                    prev_click_time = row['time']
                    prev_player = player

                elif row['key'] == 'reset':
                    player_type = 'X'
                    if player == 1:
                        player_type = 'O'

                    scores = compute_scores_layers_for_matrix(curr_move_matrix,player=player_type, normalized=False,o_weight=0.5, exp=2, neighborhood_size=2, block=True)
                    probs = compute_scores_layers_for_matrix(curr_move_matrix,player=player_type, normalized=True,o_weight=0.5, exp=2, neighborhood_size=2, block=True, lamb=None)
                    scores_data = get_scores(scores, 0, 0, probs)
                    score_x = scores_data[2]
                    # if abs(move_prob) > 1:
                    #     print 'what'

                    path_prob = move_prob*path_prob

                    delta_score = ""
                    if player == 1:
                        score_x = -1*scores_data[0]
                    if prev_x_score != None:
                        delta_x_score = score_x - prev_x_score
                        delta_score = delta_x_score


                    time_between_action = (int(row['time']) - int(prev_time))/1000.0
                    time_between_click = ""
                    if prev_click_time != None:
                        time_between_click = (int(row['time']) - int(prev_click_time))/1000.0

                    curr_data['userid'] = curr_user

                    curr_data['board_size'] = board_size
                    curr_data['board_type'] = board_type
                    curr_data['condition'] = condition
                    curr_data['board_name'] = board_name
                    curr_data['action'] = 'reset'

                    curr_data['score_move'] = prev_x_score
                    curr_data['potential_score'] = score_x
                    curr_data['move_ranking'] = ""
                    curr_data['top_possible_score'] = ""
                    curr_data['second_possible_score'] = ""
                    curr_data['num_possible_moves'] = ""
                    curr_data['delta_score'] = delta_x_score

                    curr_data['solved'] = participant_answer
                    curr_data['player'] = prev_player
                    curr_data['position'] = ""
                    curr_data['time'] = row['time']

                    curr_data['time_rel'] = int(row['time']) - initial_time
                    curr_data['time_prev_action'] = prev_time
                    curr_data['time_from_action'] = time_between_action
                    curr_data['time_from_click'] = time_between_click
                    curr_data['prev_action'] = prev_action
                    curr_data['first_move_in_path'] = ""
                    curr_data['move_number'] = ""
                    curr_data['move_number_in_path'] = len(curr_path)
                    curr_data['board_state'] = copy.deepcopy(curr_move_matrix)
                    curr_data['path'] = curr_path
                    curr_data['path_after'] = curr_path
                    curr_data['move_prob'] = move_prob
                    curr_data['path_prob'] = path_prob
                    curr_data['path_number'] = path_number


                    results_table.append(copy.deepcopy(curr_data))

                    curr_move_matrix = copy.deepcopy(initial_board)
                    prev_action = 'reset'
                    prev_time = row['time']
                    curr_path = []
                    delta_x_score = ""
                    delta_o_score = ""
                    prev_x_score = 0
                    prev_o_score = None
                    prev_player = None
                    path_prob = 1.0
                    move_stack = []

                elif row['key'] == 'undo':
                    if len(move_stack) > 0:
                        undo_move = move_stack.pop()

                        time_between_action = (int(row['time']) - int(prev_time))/1000.0
                        time_between_click = (int(row['time']) - int(prev_click_time))/1000.0

                        curr_data['userid'] = curr_user

                        curr_data['board_size'] = board_size
                        curr_data['board_type'] = board_type
                        curr_data['condition'] = condition
                        curr_data['board_name'] = board_name
                        curr_data['action'] = 'undo'

                        curr_data['score_move'] = prev_x_score
                        curr_data['move_ranking'] = ""
                        curr_data['top_possible_score'] = ""
                        curr_data['second_possible_score'] = ""
                        curr_data['num_possible_moves'] = ""
                        curr_data['delta_score'] = delta_x_score

                        curr_data['solved'] = participant_answer
                        curr_data['player'] = prev_player
                        curr_data['position'] = str(undo_move[0]) +'_' +str(undo_move[1])
                        curr_data['time'] = row['time']

                        curr_data['time_rel'] = int(row['time']) - initial_time
                        curr_data['time_prev_action'] = prev_time
                        curr_data['time_from_action'] = time_between_action
                        curr_data['time_from_click'] = time_between_click
                        curr_data['prev_action'] = prev_action
                        curr_data['first_move_in_path'] = ""
                        curr_data['move_number'] = ""
                        curr_data['move_number_in_path'] = len(curr_path)
                        curr_data['board_state'] = copy.deepcopy(curr_move_matrix)
                        curr_data['path'] = curr_path
                        curr_data['path_after'] = curr_path
                        curr_data['move_prob'] = move_prob
                        curr_data['path_prob'] = path_prob
                        curr_data['path_number'] = path_number

                        player_type = 'X'
                        if player == 1:
                            player_type = 'O'

                        scores = compute_scores_layers_for_matrix(curr_move_matrix,player=player_type, normalized=False,o_weight=0.5, exp=2, neighborhood_size=2, block=True)
                        probs = compute_scores_layers_for_matrix(curr_move_matrix,player=player_type, normalized=True,o_weight=0.5, exp=2, neighborhood_size=2, block=True, lamb=None)
                        scores_data = get_scores(scores, 0, 0, probs)
                        score_x = scores_data[2]
                        if player == 1:
                            score_x = -1*scores_data[0]

                        curr_data['potential_score'] = score_x

                        # curr_data['potential_score'] = ""
                        results_table.append(copy.deepcopy(curr_data))

                        curr_move_matrix[undo_move[0]][undo_move[1]] = 0

                        curr_path = curr_path[:-1]

                        path_prob = path_prob/move_prob
                    else:
                        print 'problem undo'

                    prev_action = 'undo'
                    prev_time = row['time']

                    if len(curr_path) == 0:
                        prev_player = None
                        delta_x_score = ""
                        delta_o_score = ""
                        prev_x_score = 0
                        prev_o_score = None
                    elif prev_player == 1:
                        prev_player = 2
                        # delta_x_score = ""
                        prev_x_score = None
                    else:
                        prev_player = 1
                        # delta_o_score = ""
                        prev_o_score = None
                # else:
                #     prev_time = row['time']

                elif row['key'] == 'start':
                    prev_time = row['time']
                    initial_time = int(prev_row['time'])


                prev_row = copy.deepcopy(row)
    dataFile = open(output_file, 'wb')
    fieldnames = ['userid','board_name','board_size','board_type','condition','solved', 'action','time','time_rel',
                  'player','score_move','move_ranking','top_possible_score', 'second_possible_score', 'num_possible_moves','delta_score','potential_score','path','move_prob', 'path_prob', 'path_number','board_state',
                  'position','path_after', 'time_from_action','time_from_click','prev_action','move_number','move_number_in_path','first_move_in_path','time_prev_action','solution']
    dataWriter = csv.DictWriter(dataFile, fieldnames=fieldnames, delimiter=',')
    dataWriter.writeheader()
    for record in results_table:
        dataWriter.writerow(record)


def explore_exploit(output_file):
    moves_data_matrics = {}
    data_first_moves = {}

    first_moves_data_matrices = {}
    results_table = []
    user_counter = 0
    reset_count = 0
    reseted = False
    for g in range(len(LOGFILE)):
        # print g
        initial_board = copy.deepcopy(START_POSITION[g])

        path_counter = 0.0
        path_counter_subpaths = 0.0
        taken_cells = 0.0

        # for i in range(len(initial_board)):
        #     for j in range(len(initial_board[i])):
        #         if ((initial_board[i][j]!=1) & (initial_board[i][j]!=2)):
        #             initial_board[i][j] = int(initial_board[i][j])
        #         elif (initial_board[i][j]==1):
        #             initial_board[i][j]='X'
        #             taken_cells+=1
        #         elif (initial_board[i][j]==2):
        #             initial_board[i][j]='O'
        #             taken_cells+=1

        move_matrix = copy.deepcopy(initial_board)
        first_move_matrix = copy.deepcopy(initial_board)

        curr_move_matrix = copy.deepcopy(move_matrix)
        curr_first_move_matrix = copy.deepcopy(move_matrix)



        with open(LOGFILE[g], 'rb') as csvfile:
            print LOGFILE[g]
            condition = LOGFILE[g][5:-10].replace("_",",")
            board_name = LOGFILE[g]
            board_name=board_name[:-4]
            log_reader = csv.DictReader(csvfile)
            move_number = 1
            move_stack = []
            curr_path = []
            curr_user = ''
            prev_time = None
            prev_action = None
            initial_time = None
            explore_time_start = None
            exploit_time_start = None
            explore_time_end = None
            exploit_time_end = None
            time_before_reset = None
            time_before_undo = None

            for row in log_reader:
                curr_data = {}
                if curr_user == '':
                    curr_user = row['userid']
                    participant_answer = check_participant_answer(curr_user)

                if row['userid'] != curr_user:
                    user_counter += 1
                    if (exploit_time_start != None) & (explore_time_start != None):
                        exploit_time = exploit_time_end - exploit_time_start
                        explore_time = explore_time_end - explore_time_start
                        curr_data['userid'] = curr_user
                        curr_data['condition'] = condition
                        curr_data['board_name'] = board_name
                        curr_data['solved'] = participant_answer
                        curr_data['explore_time'] = explore_time
                        curr_data['exploit_time'] = exploit_time
                        results_table.append(copy.deepcopy(curr_data))
                        curr_data = {}

                    # reset all values for next user
                    prev_time = None
                    prev_action = None
                    prev_row = None
                    first_move_for_player = True
                    move_number = 1
                    curr_path = []
                    move_stack = []
                    curr_user = row['userid']
                    participant_answer = check_participant_answer(curr_user)
                    initial_time = None
                    curr_move_matrix = copy.deepcopy(initial_board)
                    explore_time_start = None
                    exploit_time_start = None
                    explore_time_end = None
                    exploit_time_end = None
                    time_before_reset = None
                    time_before_undo = None
                    if reseted:
                        reset_count += 1
                    reseted = False



                elif row['key'] == 'clickPos':
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    move_stack.append((rowPos, colPos))
                    player = int(row['value'][4])
                    first_move = False
                    player_type = 'X'
                    if player == 2:
                        player_type = 'O'


                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if (curr_move_matrix[rowPos][colPos]!=1) & (curr_move_matrix[rowPos][colPos]!=2):
                        curr_move_matrix[rowPos][colPos] = player
                        curr_path.append([rowPos, colPos, player])

                        if len(curr_path) == 1:
                            first_move = True
                    else:
                        print 'weird'
                    time_between = 0
                    if prev_time == None:
                        if prev_row != None:
                            # print 'why'
                            prev_time = prev_row['time']
                            initial_time = int(prev_row['time'])
                    if prev_time == None:
                        print row
                    time_between = int(row['time']) - int(prev_time)
                    time_between = time_between/1000.0

                    if explore_time_start != None:
                        explore_time_end = int(row['time'])
                    if exploit_time_start == None:
                        exploit_time_start = int(row['time'])

                    exploit_time_end = int(row['time'])




                    curr_data['userid'] = curr_user
                    curr_data['condition'] = condition
                    curr_data['board_name'] = board_name

                    curr_data['solved'] = participant_answer
                    curr_data['player'] = player
                    curr_data['position'] = str(rowPos) + '_' + str(colPos)
                    curr_data['time'] = row['time']

                    curr_data['time_rel'] = int(row['time']) - initial_time
                    curr_data['time_prev_move'] = prev_time
                    curr_data['time_between'] = time_between
                    curr_data['prev_action'] = prev_action
                    curr_data['first_move_in_path'] = first_move
                    curr_data['move_number'] = move_number
                    curr_data['move_number_in_path'] = len(curr_path)
                    # results_table.append(copy.deepcopy(curr_data))
                    move_number += 1
                    prev_action = 'click'
                    prev_time = row['time']

                elif row['key'] == 'reset':
                    reseted = True
                    curr_move_matrix = copy.deepcopy(initial_board)
                    time_before_reset = int(row['time']) - int(prev_time)
                    # curr_data['time_before'] = time_before_reset
                    # results_table.append(copy.deepcopy(curr_data))
                    prev_action = 'reset'
                    prev_time = row['time']
                    curr_path = []

                    if (exploit_time_start != None) & (explore_time_start != None):
                        exploit_time = exploit_time_end - exploit_time_start
                        explore_time = explore_time_end - explore_time_start
                        curr_data['userid'] = curr_user
                        curr_data['condition'] = condition
                        curr_data['board_name'] = board_name
                        curr_data['solved'] = participant_answer
                        curr_data['explore_time'] = explore_time
                        curr_data['exploit_time'] = exploit_time
                        results_table.append(copy.deepcopy(curr_data))
                        curr_data = {}



                    explore_time_start = int(row['time'])
                    exploit_time_start = None
                    exploit_time_end = None
                    explore_time_end = None



                elif row['key'] == 'undo':
                    if len(move_stack) > 0:
                        undo_move = move_stack.pop()
                        curr_move_matrix[undo_move[0]][undo_move[1]] = 0
                        curr_path = curr_path[:-1]
                    else:
                        print 'problem undo'

                    prev_action = 'undo'
                    time_before_undo = int(row['time']) - int(prev_time)
                    # curr_data['time_before'] = time_before_undo
                    # results_table.append(copy.deepcopy(curr_data))
                    prev_time = row['time']


                    if (exploit_time_start != None) & (explore_time_start != None) & (len(curr_path)==0):
                        exploit_time = exploit_time_end - exploit_time_start
                        explore_time = explore_time_end - explore_time_start
                        curr_data['userid'] = curr_user
                        curr_data['condition'] = condition
                        curr_data['board_name'] = board_name
                        curr_data['solved'] = participant_answer
                        curr_data['explore_time'] = explore_time
                        curr_data['exploit_time'] = exploit_time
                        results_table.append(copy.deepcopy(curr_data))
                        curr_data = {}

                    explore_time_start = int(row['time'])
                    exploit_time_start = None
                    exploit_time_end = None
                    explore_time_end = None
                # else:
                #     prev_time = row['time']
                elif row['key'] == 'start':
                    prev_time = row['time']
                    initial_time = int(prev_row['time'])
                prev_row = copy.deepcopy(row)
    dataFile = open(output_file, 'wb')
    dataWriter = csv.DictWriter(dataFile, fieldnames=results_table[0].keys(), delimiter=',')
    dataWriter.writeheader()
    for record in results_table:
        dataWriter.writerow(record)
    print user_counter
    print reset_count



def paths_stats(participants = 'all'):
    moves_data_matrics = {}
    data_first_moves = {}

    first_moves_data_matrices = {}
    for g in range(len(LOGFILE)):
        # print g
        initial_board = copy.deepcopy(START_POSITION[g])

        path_counter = 0.0
        path_counter_subpaths = 0.0
        taken_cells = 0.0

        for i in range(len(initial_board)):
            for j in range(len(initial_board[i])):
                if ((initial_board[i][j]!=1) & (initial_board[i][j]!=2)):
                    initial_board[i][j] = int(initial_board[i][j])
                elif (initial_board[i][j]==1):
                    initial_board[i][j]='X'
                    taken_cells+=1
                elif (initial_board[i][j]==2):
                    initial_board[i][j]='O'
                    taken_cells+=1

        move_matrix = copy.deepcopy(initial_board)
        first_move_matrix = copy.deepcopy(initial_board)

        curr_move_matrix = copy.deepcopy(move_matrix)
        curr_first_move_matrix = copy.deepcopy(move_matrix)


        with open(LOGFILE[g], 'rb') as csvfile:
            first_moves_values = []
            data_first_moves_board = {}
            log_reader = csv.DictReader(csvfile)
            paths = []
            paths_counts = []
            paths_counts_subpaths = []
            entropy_values = []
            entropy_values_subpaths = []
            entropy_values_clicks = []
            entropy_values_first_moves = []
            avg_times_after_click = []
            avg_times_after_undo = []
            avg_times_after_reset = []
            curr_path = []
            curr_user = ''
            move_counter = 0.0
            total_moves = 0.0
            first_move_counter = 0.0
            num_participants = 0.0
            prev_action = None
            prev_time = None
            prev_row = None
            curr_times_after_click = []
            curr_times_after_undo = []
            curr_times_after_reset = []


            for row in log_reader:
                if curr_user == '':
                    curr_user = row['userid']

                if row['userid']!=curr_user:
                    participant_answer = check_participant_answer(curr_user)
                    if ((participants == 'all') | ((participants == 'validatedCorrect') & (participant_answer=='validatedCorrect')) | ((participants == 'solvedCorrect') & ((participant_answer=='solvedCorrect') | (participant_answer=='validatedCorrect'))) | ((participants == 'wrong') & (participant_answer=='wrong'))):
                        if len(curr_path)>0:
                            paths.append(copy.deepcopy(curr_path))

                            add_path_count_subpaths(paths_counts_subpaths,copy.deepcopy(curr_path))
                            path_counter_subpaths+=len(curr_path)
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter += 1

                        pk = []
                        for p in paths_counts:
                            # print p
                            pk.append(p[1]/path_counter)

                        ent = stats.entropy(pk)
                        if (path_counter>0):
                            entropy_values.append(ent)

                        pk = []
                        for p in paths_counts_subpaths:
                            # print p
                            pk.append(p[1]/path_counter_subpaths)

                        ent = stats.entropy(pk)
                        if (path_counter_subpaths>0):
                            entropy_values_subpaths.append(ent)

                        if move_counter > 0:
                            pk = []
                            # normalize move matrices
                            for i in range(len(curr_move_matrix)):
                                for j in range(len(curr_move_matrix[i])):
                                    if ((curr_move_matrix[i][j]!='X') & (curr_move_matrix[i][j]!='O')):
                                        pk.append(curr_move_matrix[i][j]/move_counter)
                                        curr_move_matrix[i][j] = curr_move_matrix[i][j]/move_counter
                                        move_matrix[i][j] += curr_move_matrix[i][j]
                            ent_moves = stats.entropy(pk)

                            pk_uniform = []
                            for i in range(len(pk)):
                                pk_uniform.append(1.0/len(pk))

                            ent_moves_norm_max = ent/stats.entropy(pk_uniform)
                            condition = LOGFILE[g][5:-10].replace("_",",")
                            # print condition + ',' + curr_user + ',' + str(ent_moves_norm_max) + ',' + participant_answer
                            entropy_values_clicks.append(ent_moves_norm_max)

                            # normalize move matrices
                            pk = []
                            for i in range(len(curr_first_move_matrix)):
                                for j in range(len(curr_first_move_matrix[i])):
                                    if ((curr_first_move_matrix[i][j]!='X') & (curr_first_move_matrix[i][j]!='O')):
                                        pk.append(curr_first_move_matrix[i][j]/first_move_counter)
                                        curr_first_move_matrix[i][j] = curr_first_move_matrix[i][j]/first_move_counter
                                        first_move_matrix[i][j] += curr_first_move_matrix[i][j]
                            ent_first_moves = stats.entropy(pk)
                            pk_uniform = []
                            for i in range(len(pk)):
                                pk_uniform.append(1.0/len(pk))

                            ent_first_moves_norm_max = ent_first_moves/stats.entropy(pk_uniform)
                            entropy_values_first_moves.append(ent_first_moves_norm_max)

                            # times
                            if len(curr_times_after_click) > 0:
                                avg_times_after_click.append(sum(curr_times_after_click)/len(curr_times_after_click))
                            if len(curr_times_after_reset) > 0:
                                avg_times_after_reset.append(sum(curr_times_after_reset)/len(curr_times_after_reset))
                            if len(curr_times_after_undo) > 0:
                               avg_times_after_undo.append(sum(curr_times_after_undo)/len(curr_times_after_undo))

                            data_first_moves_board[curr_user] = copy.deepcopy(first_moves_values)
                            total_moves += move_counter
                            num_participants += 1.0

                    # reset all values for next user
                    curr_path = []
                    curr_user = row['userid']
                    paths_counts = []
                    path_counter = 0
                    path_counter_subpaths = 0
                    paths_counts_subpaths = []
                    paths = []
                    move_counter = 0.0
                    first_move_counter = 0
                    curr_first_move_matrix = copy.deepcopy(initial_board)
                    curr_move_matrix = copy.deepcopy(initial_board)
                    curr_times_after_click = []
                    curr_times_after_undo = []
                    curr_times_after_reset = []
                    prev_time = None
                    prev_action = None
                    first_moves_values = []



                if row['key'] == 'clickPos':
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((curr_move_matrix[rowPos][colPos]!='X') & (curr_move_matrix[rowPos][colPos]!='O')):
                        curr_move_matrix[rowPos][colPos] += 1
                        move_counter+=1.0
                        curr_path.append([rowPos,colPos,player])
                        if len(curr_path) == 1:
                            curr_first_move_matrix[rowPos][colPos] +=1
                            first_moves_values.append((rowPos,colPos))
                            first_move_counter += 1.0
                    time_between = 0
                    if prev_time == None:
                        if prev_row != None:
                            # print 'why'
                            prev_time = prev_row['time']

                    if int(row['time']) < int(prev_time):
                        print int(row['time'])
                        print int(prev_time)
                        print 'problem'
                    time_between = int(row['time']) - int(prev_time)
                    if (time_between < 120000):
                        time_between = time_between/1000.0
                        # print time_between
                        # if (prev_action is None) | (prev_action == 'reset'):
                        if len(curr_path) == 1:
                            curr_times_after_reset.append(time_between)
                        elif prev_action == 'undo':
                            curr_times_after_undo.append(time_between)
                        elif prev_action == 'click':
                            curr_times_after_click.append(time_between)

                    prev_action = 'click'
                    prev_time = row['time']

                elif row['key'] == 'reset':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        add_path_count_subpaths(paths_counts_subpaths,copy.deepcopy(curr_path))
                        path_counter_subpaths+=len(curr_path)
                        add_path_count(paths_counts,copy.deepcopy(curr_path))
                        path_counter += 1
                        curr_path = []
                    prev_action = 'reset'
                    prev_time = row['time']

                elif row['key'] == 'undo':
                    if len(curr_path) > 0:
                        paths.append(copy.deepcopy(curr_path))
                        add_path_count_subpaths(paths_counts_subpaths,copy.deepcopy(curr_path))
                        path_counter_subpaths += len(curr_path)
                        add_path_count(paths_counts,copy.deepcopy(curr_path))
                        path_counter += 1
                        curr_path = curr_path[:-1]
                    prev_action = 'undo'
                    prev_time = row['time']
                prev_row = copy.deepcopy(row)
        # print LOGFILE[g]
        avg_ent = sum(entropy_values)/len(entropy_values)
        avg_ent_subpaths = sum(entropy_values_subpaths)/len(entropy_values_subpaths)
        avg_ent_moves = sum(entropy_values_clicks)/len(entropy_values_clicks)
        # print 'entropy_values_clicks'
        # print entropy_values_clicks
        avg_ent_first_moves = sum(entropy_values_first_moves)/len(entropy_values_first_moves)
        # avg_times_after_click_agg = sum(avg_times_after_click)/len(avg_times_after_click)
        avg_times_after_click_agg = np.median(avg_times_after_click)
        std_times_after_click_agg = np.std(avg_times_after_click)
        # avg_times_after_undo_agg = sum(avg_times_after_undo)/len(avg_times_after_undo)
        avg_times_after_undo_agg = np.median(avg_times_after_undo)
        std_times_after_undo_agg = np.std(avg_times_after_undo)
        # avg_times_after_reset_agg = sum(avg_times_after_reset)/len(avg_times_after_reset)
        avg_times_after_reset_agg = np.median(avg_times_after_reset)
        std_times_after_reset_agg = np.std(avg_times_after_reset)
        #
        # avg_times_after_click_agg = np.median(curr_times_after_click)
        # std_times_after_click_agg = np.std(curr_times_after_click)
        # # avg_times_after_undo_agg = sum(curr_times_after_undo)/len(curr_times_after_undo)
        # avg_times_after_undo_agg = np.median(curr_times_after_undo)
        # std_times_after_undo_agg = np.std(curr_times_after_undo)
        # # avg_times_after_reset_agg = sum(curr_times_after_reset)/len(curr_times_after_reset)
        # avg_times_after_reset_agg = np.median(curr_times_after_reset)
        # std_times_after_reset_agg = np.std(curr_times_after_reset)


        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!='X') & (move_matrix[i][j]!='O')):
                    move_matrix[i][j] = move_matrix[i][j]/num_participants
                elif move_matrix[i][j] == 'X':
                    move_matrix[i][j]  = -0.00001
                elif move_matrix[i][j] == 'O':
                    move_matrix[i][j]  = -0.00002

        for i in range(len(first_move_matrix)):
            for j in range(len(first_move_matrix[i])):
                if ((first_move_matrix[i][j]!='X') & (first_move_matrix[i][j]!='O')):
                    first_move_matrix[i][j] = first_move_matrix[i][j]/num_participants
                elif first_move_matrix[i][j] == 'X':
                    first_move_matrix[i][j]  = -0.00001
                elif first_move_matrix[i][j] == 'O':
                    first_move_matrix[i][j]  = -0.00002


        # data_first_moves[condition] =
        condition = LOGFILE[g][5:-10].replace("_",",")
        # print condition
        # print first_moves_values

        # print condition + "," + str(avg_ent_moves) + "," +str(avg_ent_first_moves) + "," + str(avg_ent) + ","+ str(avg_ent_subpaths) + ","\
        #       + str(avg_times_after_click_agg) + ","+ str(avg_times_after_undo_agg) + "," + str(avg_times_after_reset_agg) + "," \
        # + str(std_times_after_click_agg) + ","+ str(std_times_after_undo_agg) + "," + str(std_times_after_reset_agg) + "," + str(total_moves/num_participants)\
        #       + "," + str(num_participants) + "," + participants

        board_name = LOGFILE[g]
        board_name=board_name[:-4]
        moves_data_matrics[board_name[5:-6]] = move_matrix
        first_moves_data_matrices[board_name[5:-6]] = first_move_matrix
        data_first_moves[board_name[5:-6]] = copy.deepcopy(data_first_moves_board)
    # write_matrices_to_file(moves_data_matrics, 'data_matrices/avg_people_clicks_' +participants +  '.json')
    # write_matrices_to_file(first_moves_data_matrices, 'data_matrices/avg_people_first_moves_' +participants +  '.json')
    write_matrices_to_file(data_first_moves, 'data_matrices/people_first_moves_values_byParticipant_' +participants +  '.json')
    return (move_matrix, first_move_matrix, first_moves_values)


# def normalize_matrix(am)


def first_moves_average():
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        path_counter = 0.0
        taken_cells = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells

        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            paths = []
            paths_counts = []
            entropy_values = []
            curr_path = []
            curr_user = ''
            for row in log_reader:
                if curr_user=='':
                    curr_user = row['userid']

                if row['userid']!=curr_user:
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                    pk = []
                    for p in paths_counts:
                        # print p
                        pk.append(p[1]/path_counter)

                    ent = stats.entropy(pk)
                    if (path_counter>0):
                        entropy_values.append(ent)
                    curr_path = []
                    curr_user = row['userid']
                    paths_counts = []
                    path_counter = 0
                    paths = []

                elif row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])

                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        # move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                        curr_path.append([rowPos,colPos,player])
                        # move_counter+=1.0
                            # else:
                            #     print 'ignore'
                elif row['key']=='reset':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))

                        curr_path = []
                elif row['key']=='undo':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))

                        curr_path = curr_path[:-1]

        # print LOGFILE[g]
        avg_ent = sum(entropy_values)/len(entropy_values)

        condition = LOGFILE[g][5:-10].replace("_",",")
        measure = 'avg path entropy (no subpaths)'

        for i in range(len(entropy_values)):
            print condition + ',' + str(entropy_values[i])



def entropy_board(ignore = False):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        move_counter = 0.0
        taken_cells = 0.0;
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells
        # free_cells = free_cells - 2 #ignoring
        # print free_cells
        # move_matrix = []
        # for row in range(DIMENSION):
        #     row_positions = []
        #     for col in range(DIMENSION):
        #         row_positions.append(0)
        #     move_matrix.append(copy.deepcopy(row_positions))
        # to_ignore = IGNORE_LIST[g];
        to_ignore = None
        if ignore == True:
            to_ignore = IGNORE_LIST[g];
        if to_ignore!=None:
            free_cells = free_cells - len(to_ignore) # ignoring

        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                # if row['userid'] == USERID:
                if row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        if (to_ignore!=None):
                            if ((rowPos!=to_ignore[0][0] | colPos!=to_ignore[0][1]) & (rowPos!=to_ignore[1][0] | colPos!=to_ignore[1][1])):
                                if(player==1):
                                    move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                                    move_counter+=1.0
                            # else:
                            #     print 'ignore'
                        elif (player==1):
                            move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                            move_counter+=1.0
        # print LOGFILE[g]
        # for row in move_matrix:
        #     print row

        #compute entropy
        pk = []
        cell_counter = 0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!='X') & (move_matrix[i][j]!='O')):
                    pk.append(move_matrix[i][j]/move_counter)
        # print pk
        ent = stats.entropy(pk)
        pk_uniform = []
        for i in range(len(pk)):
            pk_uniform.append(1.0/len(pk))

        # print pk_uniform

        ent_norm = ent/free_cells
        ent_norm_max = ent/stats.entropy(pk_uniform)
        # print ent
        # print ent_norm
        condition = LOGFILE[g][5:-10].replace("_",",")
        # if (ignore):
        #     print condition+',entropy (without pruned cells),' + str(ent)
        #     print condition+',entropy normalized free cells (without pruned cells),'+ str(ent_norm)
        #     print condition+',entropy normalized max (without pruned cells),' + str(ent_norm_max)
        # else:
        #     print condition+',entropy (with pruned cells),' + str(ent)
        #     print condition+',entropy normalized free cells (with pruned cells),'+ str(ent_norm)
        #     print condition+',entropy normalized max (with pruned cells),' + str(ent_norm_max)
        # # print condition+',std entropy normalized (participant),' + str(std_entropy_norm)

        if (ignore):
            condition = condition + "_" + "without pruned cell"
        else:
            condition = condition + "_" + "with pruned cell"
        # for i in range(len(entropy_values)):
        print condition + ',' + str(ent) + ',' + str(ent_norm_max) + "," + str(ent_norm)


        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[r])):
                if (move_matrix[r][j]=='X'):
                    move_matrix[r][j] = -0.00001
                elif (move_matrix[r][j]=='O'):
                    move_matrix[r][j] = -0.00002

        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[r])):
                if (move_matrix[r][j]!=-0.00001) & (move_matrix[r][j]!=-0.00002):
                    move_matrix[r][j] = move_matrix[r][j]/move_counter

        fig_file_name = LOGFILE[g]
        fig_file_name=fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = (move_matrix, ent_norm_max)
        print ent_norm_max
    return data_matrices


def entropy_board_average(ignore = False):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        move_counter = 0.0
        taken_cells = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells

        to_ignore = None
        if ignore == True:
            to_ignore = IGNORE_LIST[g];
        if to_ignore!=None:
            free_cells = free_cells - len(to_ignore)

        curr_user = ''
        entropy_values = []
        entropy_values_norm = []
        curr_move_matrix = copy.deepcopy(move_matrix)
        prob_matrix = copy.deepcopy(move_matrix)
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                if curr_user == '':
                    curr_user = row['userid']

                if row['userid']!=curr_user:
                    #compute entropy
                    if move_counter>0:
                        pk = []
                        for i in range(len(move_matrix)):
                            for j in range(len(move_matrix[i])):
                                # prob_matrix[i][j] = np.around(move_matrix[i][j],2)
                                if ((move_matrix[i][j]!='X') & (move_matrix[i][j]!='O')):
                                    pk.append(curr_move_matrix[i][j]/move_counter)
                                    prob_matrix[i][j] = curr_move_matrix[i][j]/move_counter
                                    # print prob_matrix[i][j]
                                else:
                                    prob_matrix[i][j] = 0
                        # print pk
                        ent = stats.entropy(pk)
                        entropy_values.append(ent)
                        ent_norm = ent/free_cells
                        entropy_values_norm.append(ent_norm)

                        curr_move_matrix = copy.deepcopy(move_matrix)
                        move_counter = 0.0
                    curr_user = row['userid']

                else:
                    if row['key']=='clickPos':
                        # print row
                        rowPos = int(row['value'][0])
                        colPos = int(row['value'][2])
                        player = int(row['value'][4])
                        # print rowPos
                        # print colPos
                        # print move_matrix[rowPos][colPos]
                        if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                            if (to_ignore!=None):
                                if ((rowPos!=to_ignore[0][0] | colPos!=to_ignore[0][1]) & (rowPos!=to_ignore[1][0] | colPos!=to_ignore[1][1])):
                                    if(player==1):
                                        curr_move_matrix[rowPos][colPos] = curr_move_matrix[rowPos][colPos]+1
                                        move_counter+=1.0
                                # else:
                                #     print 'ignore'
                            elif (player==1):
                                curr_move_matrix[rowPos][colPos] = curr_move_matrix[rowPos][colPos]+1
                                move_counter+=1.0
        # print LOGFILE[g][5:-10]
        # for row in move_matrix:
        #     print row
        avg_entropy = sum(entropy_values)/len(entropy_values)
        avg_entropy_norm = sum(entropy_values_norm)/len(entropy_values_norm)

        uniform_moves = []
        for i in range(int(free_cells)):
            uniform_moves.append(1.0/free_cells)



        max_entropy = stats.entropy(uniform_moves)


        # print max_entropy

        std_entropy = np.std(entropy_values);
        std_entropy_norm = np.std(entropy_values_norm);

        entropy_values_norm_max = []
        # print entropy_values
        for i in range(len(entropy_values)):
            entropy_values_norm_max.append(entropy_values[i]/max_entropy)

        avg_entropy_norm_max = sum(entropy_values_norm_max)/len(entropy_values_norm_max)
        std_entropy_norm_max = np.std(entropy_values_norm_max);
        # std_entropy_norm_max = ' '
        # avg_entropy_norm_max = ''

        # print np.around(0.555,2)

        condition = LOGFILE[g][5:-10].replace("_",",")
        # if (ignore):
        #     print condition+',avg entropy (without pruned cells),' + str(avg_entropy)
        #     print condition+',std entropy (without pruned cells),' + str(std_entropy)
        #     print condition+',avg entropy normalized free cells (without pruned cells),' + str(avg_entropy_norm)
        #     print condition+',std entropy normalized free cells (without pruned cells),' + str(std_entropy_norm)
        #     print condition+',avg entropy normalized max (without pruned cells),' + str(avg_entropy_norm_max)
        #     print condition+',std entropy normalized max (without pruned cells),' + str(std_entropy_norm_max)
        #
        # else:
        #     print condition+',avg entropy (with pruned cells),' + str(avg_entropy)
        #     print condition+',std entropy (with pruned cells),' + str(std_entropy)
        #     print condition+',avg entropy normalized free cells (with pruned cells),' + str(avg_entropy_norm)
        #     print condition+',std entropy normalized free cells (with pruned cells),' + str(std_entropy_norm)
        #     print condition+',avg entropy normalized max (with pruned cells),' + str(avg_entropy_norm_max)
        #     print condition+',std entropy normalized max (with pruned cells),' + str(std_entropy_norm_max)

        if (ignore):
            condition = condition + "_" + "without pruned cell"
        else:
            condition = condition + "_" + "with pruned cell"
        # for i in range(len(entropy_values)):
        #     print condition + ',' + str(entropy_values[i]) + ',' + str(entropy_values_norm_max[i]) + "," + str(entropy_values_norm[i])

        fig_file_name = LOGFILE[g]
        fig_file_name=fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = (move_matrix, avg_entropy_norm_max)
        print avg_entropy_norm_max
    return data_matrices


        # print avg_entropy_norm

        # prob_mat = np.matrix(prob_matrix)
        # prob_mat = prob_mat.round(2)
        #
        # prob_matrix = np.around(prob_matrix,2)
        # print prob_matrix
        # print len(entropy_values)
        # print entropy_values
        # entropy_values = np.sort(entropy_values)
        # print entropy_values


def get_games():
    with open(LOGFILE, 'rb') as csvfile:
        log_reader = csv.DictReader(csvfile)
        all_games = []
        curr_game = []
        curr_user = ''
        for row in log_reader:
            userid = row['userid']
            if userid == curr_user:
                curr_game.append(row)
            else:
                all_games.append(gameInstance(copy.deepcopy(curr_game)))
                curr_game = []
                curr_game.append(row)
                curr_user = userid

        return all_games


def run_analysis():
    with open(LOGFILE, 'rb') as csvfile:
        log_reader = csv.DictReader(csvfile)
        all_games = []
        curr_game = []
        curr_user = ''
        for row in log_reader:
            userid = row['userid']
            if userid == curr_user:
                curr_game.append(row)
            else:
                all_games.append(gameInstance(copy.deepcopy(curr_game)))
                curr_game = []
                curr_game.append(row)
                curr_user = userid


    print all_games


    filtered_games = []
    for game in all_games:
        if (len(game.actions) > 5) & (game.solution != "") & (game.time > 10000):
            filtered_games.append(game)


    print filtered_games

    get_solutions(filtered_games)
    construct_heat_map(filtered_games)


def get_solutions(games):
    solutions = {}
    for game in games:
        if game.solution not in solutions.keys():
            solutions[game.solution] = 1
        else:
            solutions[game.solution] +=1
    print solutions

def draw_board(move):
    print move['key']
    board = move['value']
    transform_board(board)
    board_to_matrix(board)
    # print board


def transform_board(board):
    char = board[0]
    board2 = board[1:len(board)-1]
    b = board2.split(']')
    # b = board2.replace(char,"")
    # c = b.replace(']','\n')
    for row in b:
        # print row
        row_new = row[1:]
        row_new_final = row_new.replace('[',"")
        print '-----'
        print row_new_final
        print '-----'
        row_ttt = row_new_final.replace('0','_')
        row_ttt = row_ttt.replace('1','X')
        row_ttt = row_ttt.replace('2', 'O')
        row_ttt = row_ttt.replace(',', ' ')
        print row_ttt
        # marks = row_new_final.split(',')
        # for mark in marks:
        #     if mark == '0':
        #         print '_'
        #     elif mark == '1':
        #         print 'X'
        #     elif mark == '2':
        #         print 'O'
    # test = "alpha.Customer[cus_Y4o9qMEZAugtnW] ..."
    # a = re.search(r"\[([A-Za-z0-9_]+)\]",board2)
    # p = re.compile(r"\[(. *?)\]")
    #
    # # a = p.search(board2)
    # print a.group()

def board_to_matrix(board):
    positions = []
    board2 = board[1:len(board)-1]
    b = board2.split(']')
    for row in b:
        position_row = []
        # print row
        row_new = row[1:]
        row_new_final = row_new.replace('[',"")
        if (len(row))>2:
            marks = row_new_final.split(',')
            for mark in marks:
                position_row.append(int(mark))
            positions.append(copy.deepcopy(position_row))
    print positions


def board_to_matrix_list(board):
    positions = []
    board2 = board[1:len(board)-1]
    b = board2.split(']')
    for row in b:
        position_row = []
        # print row
        row_new = row[1:]
        row_new_final = row_new.replace('[',"")
        if (len(row))>2:
            marks = row_new_final.split(',')
            for mark in marks:
                position_row.append(mark)
            positions.append(copy.deepcopy(position_row))
    return positions


def construct_heat_map(games, move = 1):
    move_matrix = []
    for row in range(DIMENSION):
        row_positions = []
        for col in range(DIMENSION):
            row_positions.append(0)
        move_matrix.append(copy.deepcopy(row_positions))


    for game in games:
        move = game.get_action_by_index(1)
        if move is not None:
            move_matrix[move[0]][move[1]] += 1

    print move_matrix

    initial_position = game.board_positions[0]

    for r in range(DIMENSION):
        for c in range(DIMENSION):
            if initial_position[r][c] == 2:
                initial_position[r][c] = 'O'
            elif initial_position[r][c] == 1:
                initial_position[r][c] = 'X'
            else:
                if initial_position[r][c] == 0:
                    initial_position[r][c] = ' '


    for row in initial_position:
        print row

    for row in move_matrix:
        print row



def write_matrices_to_file(data_matrices, filename):
  with open(filename, 'w') as fp:
      json.dump(data_matrices, fp)


def choose_move(state, player_type, states_dict):
    player = 'X'
    if player_type == 2:
        player = 'O'
    if str(state) in states_dict.keys():
        probs_block = states_dict[str(state)]
    else:
        if player_type == 1:
            scores_block = compute_scores_layers_for_matrix(state, player=player, normalized=False, o_weight=0.5, exp=2, neighborhood_size=2, block=True)
        else:
            scores_block = compute_scores_layers_for_matrix(state, player='X', normalized=False, o_weight=0.5, exp=2, neighborhood_size=2, block=True)

            # scores_block = compute_scores_layers_for_matrix(state, player='X', normalized=True, o_weight=0.5, exp=2, neighborhood_size=2, block=True, only_density=True)
        probs_block = get_prob_matrix(scores_block)
        states_dict[str(state)] = copy.deepcopy(probs_block)
    rand = random.random()
    cumm_prob = 0.0
    for row in range(len(state)):
        for col in range(len(state[row])):
            if (probs_block[row][col] != -0.00001) & (probs_block[row][col] != -0.00002):
                cumm_prob += probs_block[row][col]
                if rand < cumm_prob:
                    move = [row, col]
                    return move


def simulate_game(num_simulations):
    paths_data = []
    probs = []
    boards = []
    lengths = []
    for g in range(len(LOGFILE)):
    # for g in range(1):
        board_name = LOGFILE[g]
        board_name=board_name[:-4]
        board_name = board_name[5:-6]
        print board_name
        counters = {}
        # print g
        initial_board = copy.deepcopy(START_POSITION[g])
        states_dict = {}
        paths = {}
        for j in range(num_simulations):
            curr_path = []
            max_path_length = MOVES_TO_WIN[g]
            curr_state = copy.deepcopy(initial_board)
            player = 1
            for i in range(max_path_length):
                move = choose_move(curr_state, player, states_dict)

                curr_path.append(copy.deepcopy(move+[player]))
                if str(curr_path) in paths.keys():
                    paths[str(curr_path)] += 1.0
                else:
                    paths[str(curr_path)] = 1.0
                if len(curr_path) in counters:
                    counters[len(curr_path)] += 1
                else:
                    counters[len(curr_path)] = 1
                curr_state[move[0]][move[1]] = player
                if player == 1:
                    player = 2
                elif player == 2:
                    player = 1
        for path, count in paths.iteritems():
            # print counters
            paths_data.append(path)
            p = np.array(ast.literal_eval(path))
            prob_path = count/float(counters[len(p)])
            # print len(p)
            boards.append(board_name)
            probs.append(prob_path)
            lengths.append(len(p))
    data = pd.DataFrame({'board_name':boards, 'path': paths_data, 'probability':probs, 'path_length':lengths})
    data.to_csv('stats/paths_simulations_blocking_blocking_softmax_5.csv')

if __name__ == "__main__":
    # simulate_game(500)
    # heat_map_solution(normalized=True)
    # paths_stats(participants='all')
    # paths_stats(participants='solvedCorrect')
    # paths_stats(participants='wrong')
    # paths_stats(participants='wrong')
    # moves_stats('stats/cogsci_moves_230718_solution.csv')
    # win_scores = [25,50,100,200,400,800,1600,3200,6400,12800]
    # blocking_vals = [5,10,20,40,80,160,320]
    # win_scores = [25, 250, 500, 750, 1000,1250, 1500, 1750, 2000, 2250, 2500]
    # blocking_vals = [2,20,40,60,80,100,120,140,160,180,200]
    win_scores = [1250, 1500, 1750, 2000, 2250, 2500]
    blocking_vals = [120,140,160,180,200]
    # win_scores = [200,400,800]
    # win_scores = [1600,3200,6400,12800]
    # blocking_vals = [5,10,20,40,80,160]
    # ,'linear','non-linear','interaction','blocking','interaction_blind','blocking_blind'
    # fit_heuristics(['density','linear','non-linear','interaction','blocking','interaction_blind','blocking_blind'], 'stats/heuristics_sensitivity100foldHigh.csv',win_scores=win_scores,blocking_vals=blocking_vals)
    # fit_heuristics_conf(['density','blocking','blocking_blind'], 'stats/test_conf_heuristics_soph_pval.csv',win_scores=[100],blocking_vals=[10])
    # fit_heuristics_conf(['density','blocking'], 'stats/test_conf_heuristics_pval_densityBlocking.csv',win_scores=[100],blocking_vals=[10])
    # fit_heuristics_conf(['density','blocking'], 'stats/test_conf_heuristics_blockDensity_DensWin.csv',win_scores=[100],blocking_vals=[10])
    fit_heuristics_by_move(['density','linear','non-linear','interaction','blocking'], 'stats/heuristics_byMove_player_cogsci_withPaths.csv',win_scores=[100],blocking_vals=[10])
    # fit_heuristics(['blocking'], 'stats/test.csv',win_scores=win_scores, blocking_vals=blocking_vals)
    # fit_heuristics(['density','linear','non-linear','interaction','blocking','interaction_blind','blocking_blind'], 'stats/missing_parts_heuristics_25.csv',win_scores=win_scores,blocking_vals=blocking_vals)
    # fit_heuristics(['linear','non-linear','interaction','interaction_shutter_0','blocking_shutter_0','blocking','density'], 'stats/heuristic_fit_shutter_test_noBline.csv',win_scores=[100],blocking_vals=[10])
    # fit_heuristics(['linear','non-linear','interaction','interaction_blind','blocking_blind','blocking','density'], 'stats/heuristic_fit_reg.csv',win_scores=[100],blocking_vals=[10])

    # fit_heuristics(['density','blocking'], 'stats/test_heuristic_fit_1000.csv',win_scores=[10,50,100,200,400,800,1600,3200,6400,12800],blocking_vals=[5,10,20,40,80,160,320])
    # check_participant_answer('63e5efe1')
    # transition_probs('stats/state_scores_heuristics_180718_linearNoInteraction_', normalized=True)
    # transition_probs_o_blind('stats/state_scores_heuristics_o_blind_180718', normalized=True)
    # explore_exploit('stats/exploreExploitTimesPathLength0416.csv')
    # seperate_log('logs/fullLogCogSci.csv')
    # # entropy_board()
    # # entropy_board(ignore=True)
    # # entropy_board_average()
    # # entropy_board_average(ignore=True)
    # entropy_paths(subpaths=True)
    # entropy_paths_average(subpaths=True)
    # entropy_paths(subpaths=False)
    # entropy_paths_average(subpaths=False)

    # user_stats()

    # print stats.entropy([0.25,0.25,0.25,0.25])
    # heat_map_game(normalized=True)
    # heat_map_solution(normalized=True)
    # run_analysis()
    # replay()
