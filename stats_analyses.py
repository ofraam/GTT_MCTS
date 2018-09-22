import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import seaborn as sns
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as stats
import copy
import csv
from sklearn.covariance import EllipticEnvelope
from  sklearn.neighbors import LocalOutlierFactor
from altair import Chart
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve
from scipy import interp
from lmfit import minimize as lmmin
from lmfit import Parameters
from scipy.interpolate import griddata


from sklearn import metrics
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
# from patsy import
from replay import *


# import Tkinter
import ast


# these are the files with user data foeach of the board
LOGFILE = ['6_hard_full','6_hard_pruned','10_hard_full','10_hard_pruned', '6_easy_full','6_easy_pruned','10_easy_full','10_easy_pruned', '10_medium_full','10_medium_pruned']
# these are the boards starting positions (1 = X, 2 = O)
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


def rank_biserial_effect_size(x,y):
    mann_whitney_res = stats.mannwhitneyu(x, y)
    u = mann_whitney_res[0]
    effect_size = 1.0-((2.0*u)/(len(x)*len(y)))
    return effect_size


def upper_bound_ci_correct(df):
    ci95_hi = df.correct.mean() + df.correct.sem() * 1.96
    ci95_hi_heuristic = df.numberOfHeuristicComp.mean() + df.numberOfHeuristicComp.sem() * 1.96
    return ci95_hi, ci95_hi_heuristic



def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def spearmanr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    # r, p = stats.pearsonr(x,y)
    # r_z = np.arctanh(r)
    # se = 1/np.sqrt(x.size-3)
    # z = stats.norm.ppf(1-alpha/2)
    # lo_z, hi_z = r_z-z*se, r_z+z*se
    # lo, hi = np.tanh((lo_z, hi_z))
    r, p = stats.spearmanr(x,y)
    stderr = 1.0 / math.sqrt(x.size - 3)
    z = stats.norm.ppf(1-alpha/2)
    print z
    delta = z * stderr
    lower = math.tanh(math.atanh(r) - delta)
    upper = math.tanh(math.atanh(r) + delta)
    # print "lower %.6f upper %.6f" % (lower, upper)
    return r, p, lower, upper

def medianCI(data, ci, p):
	'''
	data: pandas datafame/series or numpy array
	ci: confidence level
	p: percentile' percent, for median it is 0.5
	output: a list with two elements, [lowerBound, upperBound]
	'''
	if type(data) is pd.Series or type(data) is pd.DataFrame:
		#transfer data into np.array
		data = data.values

	#flat to one dimension array
	data = data.reshape(-1)
	data = np.sort(data)
	N = data.shape[0]

	lowCount, upCount = stats.binom.interval(ci, N, p, loc=0)
	#given this: https://onlinecourses.science.psu.edu/stat414/node/316
	#lowCount and upCount both refers to  W's value, W follows binomial Dis.
	#lowCount need to change to lowCount-1, upCount no need to change in python indexing
	lowCount -= 1
	# print lowCount, upCount
	return data[int(lowCount)], data[int(upCount)]


def boot_matrix(z, B):
    """Bootstrap sample

    Returns all bootstrap samples in a matrix"""

    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]

def bootstrap_mean(x, B=10000, alpha=0.05, plot=False):
    """Bootstrap standard error and (1-alpha)*100% c.i. for the population mean

    Returns bootstrapped standard error and different types of confidence intervals"""

    # Deterministic things
    n = len(x)  # sample size
    orig = x.mean()  # sample mean
    se_mean = x.std()/np.sqrt(n) # standard error of the mean
    qt = stats.t.ppf(q=1 - alpha/2, df=n - 1) # Student quantile

    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    sampling_distribution = xboot.mean(axis=1)

   # Standard error and sample quantiles
    se_mean_boot = sampling_distribution.std()
    quantile_boot = np.percentile(sampling_distribution, q=(100*alpha/2, 100*(1-alpha/2)))

    # RESULTS
    print("Estimated mean:", orig)
    print("Classic standard error:", se_mean)
    print("Classic student c.i.:", orig + np.array([-qt, qt])*se_mean)
    print("\nBootstrap results:")
    print("Standard error:", se_mean_boot)
    print("t-type c.i.:", orig + np.array([-qt, qt])*se_mean_boot)
    print("Percentile c.i.:", quantile_boot)
    print("Basic c.i.:", 2*orig - quantile_boot[::-1])

    if plot:
        plt.hist(sampling_distribution, bins="fd")



def bootstrap_t_pvalue(x, y, equal_var=False, B=10000, plot=False):
    """Bootstrap p values for two-sample t test

    Returns boostrap p value, test statistics and parametric p value"""

    # Original t test statistic
    orig = stats.ttest_ind(x, y, equal_var=equal_var)

    # Generate boostrap distribution of t statistic
    xboot = boot_matrix(x - x.mean(), B=B) # important centering step to get sampling distribution under the null
    yboot = boot_matrix(y - y.mean(), B=B)
    sampling_distribution = stats.ttest_ind(xboot, yboot, axis=1, equal_var=equal_var)[0]

    # Calculate proportion of bootstrap samples with at least as strong evidence against null
    p = np.mean(sampling_distribution >= orig[0])

    # RESULTS
    print("p value for null hypothesis of equal population means:")
    print("Parametric:", orig[1])
    print("Bootstrap:", 2*min(p, 1-p))

    # Plot bootstrap distribution
    if plot:
        plt.figure()
        plt.hist(sampling_distribution, bins="fd")

    return 2*min(p, 1-p)


def residual(p, x, data):
    vmax = p['vmax'].value
    # vmax = 1.0
    km = p['km'].value
    model = vmax * x / (km + x)
    # model =  x / (km + x)
    return (data - model)

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_y(patch.get_y() + diff * .5)


def get_user_stats(dynamics, exploreExploitData):
    dataset = dynamics
    all_data = []
    curr_data = {}
    for userid in dataset['userid'].unique():
        user_data = dataset.loc[dataset['userid'] == userid]
        exploreExploitData_user = exploreExploitData.loc[exploreExploitData['userid'] == userid]
        curr_data['userid'] = userid

        curr_data['num_resets'] = user_data[user_data['action'] == 'reset'].shape[0]
        curr_data['num_restarts'] = user_data[(user_data['prev_action'] != '') & (user_data['move_number_in_path'] == 1)].shape[0]
        curr_data['mean_score'] = user_data['score_heuristic_x'].mean()
        curr_data['median_score'] = user_data['score_heuristic_x'].median()
        curr_data['num_moves_win_score'] = user_data[(user_data['score_heuristic_x'] == 100) & (user_data['player'] == 1)].shape[0]
        first_moves = user_data.loc[user_data['move_number_in_path'] == 1]
        curr_data['num_unique_first_moves'] = len(first_moves['position'].unique())
        curr_data['solved'] = user_data.iloc[0]['solved']
        curr_data['board'] = user_data.iloc[0]['board_name']
        curr_data['number_of_moves'] = user_data['move_number'].max()
        curr_data['solution_time'] = user_data['time_rel_sec'].max()
        curr_data['explore_time'] = None
        curr_data['exploit_time'] = None
        curr_data['avg_first_move_score'] = first_moves['score_heuristic_x'].mean()
        curr_data['median_first_move_score'] = first_moves['score_heuristic_x'].median()
        if exploreExploitData_user.shape[0]>0:
            curr_data['explore_time'] = exploreExploitData_user.iloc[0]['explore_time']
            curr_data['exploit_time'] = exploreExploitData_user.iloc[0]['exploit_time']
        all_data.append(copy.deepcopy(curr_data))
    dataFile = open('stats\user_stats1604.csv', 'wb')
    dataWriter = csv.DictWriter(dataFile, fieldnames=curr_data.keys(), delimiter=',')
    dataWriter.writeheader()
    for record in all_data:
        dataWriter.writerow(record)

def compare_start_move(dynamics):
    first_move = {'6hard':'0_3', '6easy':'3_5', '10hard':'2_4', '10easy':'4_9', '10medium':'3_2'}
    second_move = {'6hard':'3_0', '6easy':'0_2', '10hard':'6_0', '10easy':'0_5', '10medium':'3_5'}
    # hard6pruned = dynamics.loc[(dynamics['board_name']=='6_hard_pruned') & (dynamics['move_number_in_path']==1)]
    # hard6full = dynamics.loc[dynamics['board_name']=='6_hard_full' ]

    # hard6full['prev_pos'] = hard6full['position'].shift()
    # hard6full.ix[0,'prev_pos'] =
    dynamics['first_pruned'] = False
    for index, row in dynamics.iterrows():
        if (row['move_number_in_path'] == 3) & (row['condition'] == 'full'):
            board = row['sizeType']
            # print board
            prev_row = dynamics.iloc[[index-1]]
            prev_prev_row = dynamics.iloc[[index-2]]
            # print prev_row['position'].values[0]
            # print prev_prev_row['position'].values[0]
            if (prev_row['position'].values[0] == second_move[board]) & (prev_prev_row['position'].values[0] == first_move[board]):
                # print 'boo'
                dynamics.loc[index,'first_pruned'] = True

    dynamics.to_csv('stats/dynamicsFirstMoves.csv')

        # print row

def probs_clicks(dynamics):
    data_matrics = {}
    full = dynamics.loc[(dynamics['condition'] == 'full') & (dynamics['first_pruned'] == True) & (dynamics['action'] == 'click')]
    pruned = dynamics.loc[(dynamics['condition'] == 'pruned') & (dynamics['move_number_in_path'] == 1) & (dynamics['action'] == 'click')]
    boards_full = full.board_name.unique()
    for i in range(len(LOGFILE)):
        board = LOGFILE[i]
        board_matrix = copy.deepcopy(START_POSITION[i])

        board_data = full
        if board.endswith('pruned'):
            board_data = pruned
        else:
            board_matrix = copy.deepcopy(START_POSITION[i+1])


        first_moves = board_data.loc[board_data['board_name'] == board]
        total_first_moves = first_moves.shape[0]

        print total_first_moves
        for row in range(len(board_matrix)):
            for col in range(len(board_matrix[row])):
                if board_matrix[row][col] == 1:
                    board_matrix[row][col] = -0.00001
                elif board_matrix[row][col] == 2:
                    board_matrix[row][col] = -0.00002
                elif total_first_moves > 0:
                    num = first_moves[(first_moves['row'] == row) & (first_moves['col'] == col)].shape[0]
                    board_matrix[row][col] = float(num)/float(total_first_moves)

        data_matrics[board] = copy.deepcopy(board_matrix)

    write_matrices_to_file(data_matrics, 'data_matrices/cogsci/first_pruned280818.json')


def merge_undos_to_reset(dynamics):
    first_undo_index = -1
    curr_undo_list = []
    undo_index_list = []
    curr_user = ""
    for index, row in dynamics.iterrows():
        if curr_user != row['userid']: # reset undos if new user
            curr_user = row['userid']
            first_undo_index = -1
            curr_undo_list = []

        if row['action'] == 'undo':
            if first_undo_index == -1:
                first_undo_index = index
                curr_undo_list.append(index)
            else:
                curr_undo_list.append(index)
            if row['move_number_in_path'] == 1:  # undos lead to reset, replace
                dynamics.loc[index, 'action'] = 'reset'
                curr_undo_list.pop()
                undo_index_list.extend(copy.deepcopy(curr_undo_list))
                first_undo_index = -1
                curr_undo_list = []
        else:
            first_undo_index = -1
            curr_undo_list = []
    print dynamics.shape[0]
    dynamics.drop(undo_index_list,inplace=True)
    print dynamics.shape[0]
    dynamics.to_csv('stats/dynamics_undosReplaced.csv')


def add_aperture_values(dynamics):
    aperture_values = []
    open_path_values = []
    for index, row in dynamics.iterrows():
        if (row['move_number_in_path'] == 1) | (row['action'] != 'click'):
            aperture_values.append(0)
            open_path_values.append(False)
            continue
        path = np.array(ast.literal_eval(row['path']))
        prev_ind = len(path) - 1
        player = 'O'
        if row['player'] == 1:
            prev_ind -= 1
            player = 'X'
        # print index
        # print path
        prev_move_row = path[prev_ind][0]
        prev_move_col = path[prev_ind][1]
        # neighbors = get_neighboring_squares(row['board_size'], [prev_move_row,prev_move_col], 2)
        active_squares = get_open_paths_through_square(prev_move_row, prev_move_col, np.array(ast.literal_eval(row['board_state'])), 1)
        square = row['position'].split('_')
        square_row = int(square[0])
        square_col = int(square[1])
        square = [square_row,square_col]
        aperture = 1
        open_path = True
        if len(active_squares) == 0:
            open_path = False
            active_squares.append([prev_move_row, prev_move_col])
        while not check_square_in_list(square, active_squares):
            active_squares = expand_neighborhood(active_squares, row['board_size'])
            aperture += 1

        aperture_values.append(aperture)
        open_path_values.append(open_path)

    dynamics['shutter'] = aperture_values
    dynamics['open_path'] = open_path_values
    dynamics.to_csv('stats/moves_hueristic_scores_shutter230718.csv')


def fit_heuristic_user_moves(transitions,dynamics):
    epsilon = 0.0001
    userids = []
    likelihoods_block = []
    likelihoods_int = []
    likelihoods_block_dens = []
    likelihoods_int_dens = []
    likelihoods_dens = []
    heuristic = []
    boards = []
    move_numbers = []

    for userid in dynamics['userid'].unique():
        user_data = dynamics.loc[(dynamics['userid'] == userid) & (dynamics['action'] == 'click')]
        if user_data.shape[0] > 0:

            log_likelihoods_block = 0.0
            log_likelihoods_interaction = 0.0
            log_likelihoods_density = 0.0
            prob_user_block = 1.0
            prob_user_interaction = 1.0
            prob_user_density = 1.0
            paths = []
            curr_path = []
            counter = 0.0
            for index, row in user_data.iterrows():
                transitions_board = transitions.loc[transitions['sizeType'] == row['sizeType']]

                move = row['position'].split('_')
                row_pos = move[0]
                col_pos = move[1]
                board_state = row['board_state']
                state = np.array(ast.literal_eval(board_state))
                probs_data = transitions_board.loc[transitions_board['board_state'] == board_state]
                probs_block = np.array(ast.literal_eval(probs_data['probs_blocking'].iloc[0]))
                probs_interaction = np.array(ast.literal_eval(probs_data['probs_interaction'].iloc[0]))
                probs_density = np.array(ast.literal_eval(probs_data['probs_density'].iloc[0]))
                probs_block_dens = np.array(ast.literal_eval(probs_data['probs_blocking_dens'].iloc[0]))
                probs_interaction_dens = np.array(ast.literal_eval(probs_data['probs_interaction_dens'].iloc[0]))
                last_move = row['position'].split('_')
                row_pos = int(last_move[0])
                col_pos = int(last_move[1])
                prob_block = probs_block[row_pos][col_pos]
                if prob_block == 0:
                    prob_block = epsilon
                prob_interaction = probs_interaction[row_pos][col_pos]
                if prob_interaction == 0:
                    prob_interaction = epsilon

                prob_block_dens = probs_block_dens[row_pos][col_pos]
                if prob_block_dens == 0:
                    prob_block_dens = epsilon
                prob_int_dens = probs_interaction_dens[row_pos][col_pos]
                if prob_int_dens == 0:
                    prob_int_dens = epsilon

                prob_density = probs_density[row_pos][col_pos]
                if prob_density == 0:
                    prob_density = epsilon
                prob_user_block = prob_user_block*prob_block
                prob_user_interaction = prob_user_interaction*prob_interaction
                prob_user_density = prob_user_density*prob_density
                # sum_user_likelihoods +/= comm_prob
                # print prob_block
                likelihoods_block.append(math.log(prob_block))
                likelihoods_block_dens.append(math.log(prob_block_dens))
                boards.append(user_data['board_name'].iloc[0])
                userids.append(userid)
                move_numbers.append(row['move_number_in_path'])
                likelihoods_int.append(math.log(prob_interaction))
                likelihoods_int_dens.append(math.log(prob_int_dens))
                # boards.append(user_data['board_name'].iloc[0])
                # userids.append(userid)
                # move_numbers.append(row['move_number_in_path'])
                likelihoods_dens.append(math.log(prob_density))
                # boards.append(user_data['board_name'].iloc[0])
                # userids.append(userid)
                # move_numbers.append(row['move_number_in_path'])
                counter += 1.0
                # break

            likelihood_vals = []


            # print log_likelihoods_block/counter
            # print log_likelihoods_interaction/counter
            # print log_likelihoods_density/counter
            # if max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density]) == log_likelihoods_block:
            #     heuristic.append('blocking')
            # elif max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density]) == log_likelihoods_interaction:
            #     heuristic.append('interaction')
            # else:
            #     heuristic.append('density')
    heuristic_vals = {'board':boards, 'userid':userids,'likelihoods_block':likelihoods_block,'likelihoods_interaction':likelihoods_int,'likelihoods_block_dens':likelihoods_block_dens,'likelihoods_interaction_dens':likelihoods_int_dens,'likelihoods_density':likelihoods_dens, 'move_number_in_path':move_numbers}
    heuristics_df = pd.DataFrame(heuristic_vals)
    heuristics_df.to_csv('stats/heuristics_fitted_by_move_combinations.csv')


def fit_heuristic_user(transitions, dynamics, heuristics):
    epsilon = 0.0001
    userids = []
    likelihoods_block = []
    likelihoods_int = []
    likelihoods_linear = []
    likelihoods_dens = []
    likelihoods_block_dens = []
    likelihoods_int_dens = []
    likelihoods_linear_dens = []
    heuristic = []
    boards = []

    for userid in dynamics['userid'].unique():
        # if userid!= '49e6113d':
        #     continue
        user_data = dynamics.loc[(dynamics['userid'] == userid) & (dynamics['action'] == 'click')]
        if user_data.shape[0] > 0:
            boards.append(user_data['board_name'].iloc[0])
            log_likelihoods_block = 0.0
            log_likelihoods_interaction = 0.0
            log_likelihoods_linear = 0.0
            log_likelihoods_block_dens = 0.0
            log_likelihoods_interaction_dens = 0.0
            log_likelihoods_linear_dens = 0.0
            log_likelihoods_density = 0.0
            prob_user_block = 1.0
            prob_user_interaction = 1.0
            prob_user_linear = 1.0
            prob_user_block_dens = 1.0
            prob_user_interaction_dens = 1.0
            prob_user_linear_dens = 1.0
            prob_user_density = 1.0
            paths = []
            curr_path = []
            counter = 0.0
            for index, row in user_data.iterrows():
                transitions_board = transitions.loc[transitions['size_type'] == row['size_type']]

                move = row['position'].split('_')
                row_pos = move[0]
                col_pos = move[1]
                board_state = row['board_state']
                state = np.array(ast.literal_eval(board_state))
                probs_data = transitions_board.loc[transitions_board['board_state'] == board_state]
                probs_block = np.array(ast.literal_eval(probs_data['probs_blocking'].iloc[0]))
                probs_interaction = np.array(ast.literal_eval(probs_data['probs_interaction'].iloc[0]))
                probs_linear = np.array(ast.literal_eval(probs_data['probs_linear'].iloc[0]))
                probs_density = np.array(ast.literal_eval(probs_data['probs_density'].iloc[0]))
                probs_block_dens = np.array(ast.literal_eval(probs_data['probs_blocking_dens'].iloc[0]))
                probs_interaction_dens = np.array(ast.literal_eval(probs_data['probs_interaction_dens'].iloc[0]))
                probs_linear_dens = np.array(ast.literal_eval(probs_data['probs_linear_dens'].iloc[0]))
                last_move = row['position'].split('_')
                row_pos = int(last_move[0])
                col_pos = int(last_move[1])
                prob_block = probs_block[row_pos][col_pos]
                if prob_block == 0:
                    prob_block = epsilon
                prob_interaction = probs_interaction[row_pos][col_pos]
                if prob_interaction == 0:
                    prob_interaction = epsilon
                prob_linear= probs_linear[row_pos][col_pos]
                if prob_linear == 0:
                    prob_linear = epsilon

                prob_block_dens = probs_block_dens[row_pos][col_pos]
                if prob_block_dens == 0:
                    prob_block_dens = epsilon
                prob_int_dens = probs_interaction_dens[row_pos][col_pos]
                if prob_int_dens == 0:
                    prob_int_dens = epsilon
                prob_linear_dens= probs_linear_dens[row_pos][col_pos]
                if prob_linear_dens == 0:
                    prob_linear_dens = epsilon


                prob_density = probs_density[row_pos][col_pos]
                if prob_density == 0:
                    prob_density = epsilon

                prob_user_block = prob_user_block*prob_block
                prob_user_interaction = prob_user_interaction*prob_interaction
                prob_user_linear = prob_user_linear*prob_linear
                prob_user_block_dens = prob_user_block_dens*prob_block_dens


                prob_user_interaction_dens = prob_user_interaction_dens*prob_int_dens
                prob_user_linear_dens = prob_user_linear_dens*prob_linear_dens
                prob_user_density = prob_user_density*prob_density
                # sum_user_likelihoods +/= comm_prob
                # print prob_block
                log_likelihoods_block += math.log(prob_block)
                log_likelihoods_interaction += math.log(prob_interaction)
                log_likelihoods_linear += math.log(prob_linear)
                log_likelihoods_block_dens += math.log(prob_block_dens)
                log_likelihoods_interaction_dens += math.log(prob_int_dens)
                log_likelihoods_linear_dens += math.log(prob_linear_dens)
                # print prob_density
                log_likelihoods_density += math.log(prob_density)
                counter += 1.0
                # break

            userids.append(userid)
            likelihood_vals = []

            likelihoods_block.append(log_likelihoods_block/counter)
            likelihoods_int.append(log_likelihoods_interaction/counter)
            likelihoods_linear.append(log_likelihoods_linear/counter)
            likelihoods_dens.append(log_likelihoods_density/counter)
            likelihoods_block_dens.append(log_likelihoods_block_dens/counter)
            likelihoods_linear_dens.append(log_likelihoods_linear_dens/counter)
            likelihoods_int_dens.append(log_likelihoods_interaction_dens/counter)
            # print log_likelihoods_block/counter
            # print log_likelihoods_interaction/counter
            # print log_likelihoods_density/counter
            if max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density, log_likelihoods_block_dens, log_likelihoods_interaction_dens, log_likelihoods_linear, log_likelihoods_linear_dens]) == log_likelihoods_block:
                heuristic.append('blocking')
            elif max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density, log_likelihoods_block_dens, log_likelihoods_interaction_dens, log_likelihoods_linear, log_likelihoods_linear_dens]) == log_likelihoods_interaction:
                heuristic.append('interaction')
            elif max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density, log_likelihoods_block_dens, log_likelihoods_interaction_dens, log_likelihoods_linear, log_likelihoods_linear_dens]) == log_likelihoods_block_dens:
                heuristic.append('blocking_dens')
            elif max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density, log_likelihoods_block_dens, log_likelihoods_interaction_dens, log_likelihoods_linear, log_likelihoods_linear_dens]) == log_likelihoods_interaction_dens:
                heuristic.append('interaction_dens')
            elif max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density, log_likelihoods_block_dens, log_likelihoods_interaction_dens, log_likelihoods_linear, log_likelihoods_linear_dens]) == log_likelihoods_linear:
                heuristic.append('linear')
            elif max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density, log_likelihoods_block_dens, log_likelihoods_interaction_dens, log_likelihoods_linear, log_likelihoods_linear_dens]) == log_likelihoods_linear_dens:
                heuristic.append('linear_dens')
            else:
                heuristic.append('density')
    heuristic_vals = {'board':boards, 'userid':userids,'likelihoods_block':likelihoods_block,'likelihoods_interaction':likelihoods_int, 'likelihood_linear': likelihoods_linear, 'likelihoods_linear_dens':likelihoods_linear_dens,'likelihoods_block_dens':likelihoods_block_dens,'likelihoods_int_dens':likelihoods_int_dens,'likelihoods_density':likelihoods_dens,'heuristic':heuristic}
    heuristics_df = pd.DataFrame(heuristic_vals)
    heuristics_df.to_csv('stats/heuristics_fitted_combinations_o_blindness_29052018_4.csv')


def compute_log_likelihood_heuristic(transitions, user_data, heuristic):
    epsilon = 0.0001
    counter = 0.0
    log_likelihoods_heuristic = 0.0
    for index, row in user_data.iterrows():
        move = row['position'].split('_')
        row_pos = move[0]
        col_pos = move[1]
        board_state = row['board_state']
        state = np.array(ast.literal_eval(board_state))
        probs_data = transitions.loc[transitions['board_state'] == board_state]
        if probs_data[heuristic].shape[0] == 0:
            print board_state
        probs_heuristic = np.array(ast.literal_eval(probs_data[heuristic].iloc[0]))

        last_move = row['position'].split('_')
        row_pos = int(last_move[0])
        col_pos = int(last_move[1])
        prob_heuristic = probs_heuristic[row_pos][col_pos]
        if prob_heuristic == 0:
            prob_heuristic = epsilon

        log_likelihoods_heuristic += math.log(prob_heuristic)

        counter += 1.0

    return log_likelihoods_heuristic/counter


def fit_heuristic_user_list(transitions, dynamics, heuristics, filename):
    userids = []
    heuristic_fitted = []
    boards = []
    log_likelihoods_dict = {}
    for heuristic in heuristics:
        log_likelihoods_dict[heuristic] = []

    for userid in dynamics['userid'].unique():
         # & (dynamics['player'] == 2)
        user_data = dynamics.loc[(dynamics['userid'] == userid) & (dynamics['action'] == 'click')]
        # print userid
        if user_data.shape[0] == 0:
            continue
        boards.append(user_data['board_name'].iloc[0])
        userids.append(userid)
        max_log_likelihood = -1000
        best_heuristic = ""
        user_transitions = transitions.loc[transitions['size_type'] == user_data['size_type'].iloc[0]]
        for heuristic in heuristics:
            log_likelihood_heuristic = compute_log_likelihood_heuristic(user_transitions, user_data, heuristic)
            log_likelihoods_dict[heuristic].append(log_likelihood_heuristic)
            if log_likelihood_heuristic > max_log_likelihood:
                max_log_likelihood = log_likelihood_heuristic
                best_heuristic = heuristic
        heuristic_fitted.append(best_heuristic)
    # dynamics['heuristic'] = heuristic_fitted
    # dynamics.to_csv('stats/dynamics_heuristics_300618.csv')
    heuristic_vals = {'board': boards, 'userid': userids, 'heuristic': heuristic_fitted}
    for k,v in log_likelihoods_dict.iteritems():
        heuristic_vals[k] = v
    heuristics_df = pd.DataFrame(heuristic_vals)
    heuristics_df.to_csv(filename)


def fit_heuristic_user_path(transitions,dynamics):
    epsilon = 0.0001
    for userid in dynamics['userid'].unique():
        user_data = dynamics.loc[(dynamics['userid'] == userid) & (dynamics['action'] == 'click')]
        log_likelihoods_block = 0.0
        log_likelihoods_interaction = 0.0
        log_likelihoods_density = 0.0
        paths = []
        curr_path = []
        for index, row in user_data.iterrows():
            transitions_board = transitions.loc[transitions['size_type'] == row['size_type']]
            move = row['position'].split('_')
            row_pos = move[0]
            col_pos = move[1]
            board_state = row['board_state']
            state = np.array(ast.literal_eval(board_state))


            path_data = np.array(ast.literal_eval(row['path']))
            for pos in path_data:
                row_pos = pos[0]
                col_pos = pos[1]
                state[row_pos][col_pos] = 0

            probs_data = transitions_board.loc[transitions_board['board_state'] == str(state)]
            probs_block = np.array(ast.literal_eval(probs_data['probs_blocking'].iloc[0]))
            probs_interaction = np.array(ast.literal_eval(probs_data['probs_interaction'].iloc[0]))
            probs_density = np.array(ast.literal_eval(probs_data['probs_density'].iloc[0]))


            prob_path_block = 1.0
            prob_path_interaction = 1.0
            prob_path_density = 1.0
            for pos in path_data:
                row_pos = pos[0]
                col_pos = pos[1]
                prob_block = probs_block[row_pos][col_pos]
                if prob_block == 0:
                    prob_block = epsilon
                prob_interaction = probs_interaction[row_pos][col_pos]
                if prob_interaction == 0:
                    prob_interaction = epsilon
                prob_density = probs_density[row_pos][col_pos]
                if prob_density == 0:
                    prob_density = epsilon
                prob_path_block = prob_path_block*prob_block
                prob_path_interaction = prob_path_interaction*prob_interaction
                prob_path_density = prob_path_density*prob_density

            last_move = row['position'].split('_')
            row_pos = int(last_move[0])
            col_pos = int(last_move[0])
            prob_block = probs_block[row_pos][col_pos]
            if prob_block == 0:
                prob_block = epsilon
            prob_interaction = probs_interaction[row_pos][col_pos]
            if prob_interaction == 0:
                prob_interaction = epsilon
            prob_density = probs_density[row_pos][col_pos]
            if prob_density == 0:
                prob_density = epsilon
            prob_path_block = prob_path_block*prob_block
            prob_path_interaction = prob_path_interaction*prob_interaction
            prob_path_density = prob_path_density*prob_density
            # sum_user_likelihoods +/= comm_prob
            print prob_path_block
            log_likelihoods_block += math.log(prob_path_block)
            log_likelihoods_interaction += math.log(prob_path_interaction)
            log_likelihoods_density += math.log(prob_path_density)

        print userid
        print log_likelihoods_block
        print log_likelihoods_interaction
        print log_likelihoods_density


def compute_path_likelihood_mc():
    simulation_data = pd.read_csv("stats/paths_simulations2000.csv")
    participant_data = pd.read_csv("dynamics06042018")
    userids = []
    path_nums = []
    probs = []
    likelihoods = []
    sim_heuristics = []
    paths = []
    path_lengths = []
    boards = []

    for userid in participant_data['userid'].unique():
        user_data = participant_data.loc[(participant_data['userid'] == userid) & (participant_data['action'] == 'click')]
        if user_data.shape[0] == 0:  # no user clicks
            continue
        for path_num in user_data['path_number'].unique():
            path_data = user_data.loc[(user_data['move_number_in_path'] == user_data['move_number_in_path'].max())]
            path_str = path_data['path_after'].iloc[0]
            path = np.array(ast.literal_eval(path_str))
            board_sim_data = simulation_data.loc[simulation_data['board_name'] == path_data['board_name'].iloc[0]]
            prob_path = 0.0
            path_sim_data = board_sim_data.loc[board_sim_data['path'] == path_str]
            if path_sim_data.shape[0] > 0:
                prob_path = path_sim_data['probability']
            userids.append(userid)
            path_nums.append(path_num)
            paths.append(path)
            path_lengths.append(len(path))
            probs.append(prob_path)
            likelihoods.append(math.log(prob_path))
            boards.append(path_data['board_name'].iloc[0])

    data_dict = {'board':boards, 'userid':userids, 'path_number': path_nums, 'path':paths, 'path_length':path_lengths, 'probability_block':probs, 'likelihood_block':likelihoods}
    paths_probs_df = pd.DataFrame(data_dict)
    paths_probs_df.to_csv('stats/participants_path_probabilities_simulation.csv')


def compute_path_probabilities_participants():
    participant_data = pd.read_csv("stats/dynamics06042018.csv")
    board_names = ['6_easy_full','6_easy_pruned', '10_easy_full', '10_easy_pruned','6_hard_full','6_hard_pruned', '10_hard_full', '10_hard_pruned',  '10_medium_full', '10_medium_pruned']
    boards = []
    path_lengths = []
    probs = []
    counts = []
    paths = []
    # path_numbers = []

    for board_name in board_names:
        # for path_number in participant_data['path_number'].unique():
        # path_data_participants = participant_data.loc[(participant_data['action'] == 'click') & (participant_data['board_name'] == board_name) & (participant_data['path_number'] == path_number)]
        path_data_participants = participant_data.loc[(participant_data['action'] == 'click') & (participant_data['board_name'] == board_name)]
        for path_length in participant_data['move_number_in_path'].unique():
            # print path_length

            paths_counts = {}
            num_paths = 0.0
            paths_data = path_data_participants.loc[path_data_participants['move_number_in_path'] == path_length]

            max_vals = path_data_participants.groupby(['userid','path_number'], as_index=False)['move_number_in_path'].max()
            for index, row in paths_data.iterrows():
                max_val = max_vals.loc[(max_vals['userid'] == row['userid']) & (max_vals['path_number'] == row['path_number'])]
                if row['move_number_in_path'] != max_val['move_number_in_path'].iloc[0]:
                    continue
                path_str = row['path_after']

                if path_str in paths_counts:
                    paths_counts[path_str] += 1.0
                else:
                    paths_counts[path_str] = 1.0

                num_paths += 1.0

            for p, count in paths_counts.iteritems():
                path = np.array(ast.literal_eval(p))
                paths.append(p)
                path_lengths.append(len(path))
                counts.append(count)
                probs.append(count/num_paths)
                # path_numbers.append(path_number)
                boards.append(board_name)


    # data_dict = {'board':boards, 'path_length': path_lengths, 'path':paths, 'probability': probs, 'counts': counts, 'path_number':path_numbers}
    data_dict = {'board':boards, 'path_length': path_lengths, 'path':paths, 'probability': probs, 'counts': counts,}

    paths_probs_df = pd.DataFrame(data_dict)
    paths_probs_df.to_csv('stats/participants_path_probabilities_subpaths2.csv')


def compare_distributions_simulation_population():
    simulation_data = pd.read_csv("stats/paths_simulations_blocking_blocking_softmax_5.csv")
    participant_data = pd.read_csv("stats/participants_path_probabilities_subpaths2.csv")

    board_names = ['6_easy_full','6_easy_pruned', '10_easy_full', '10_easy_pruned','6_hard_full','6_hard_pruned', '10_hard_full', '10_hard_pruned',  '10_medium_full', '10_medium_pruned']

    boards = []
    path_lengths = []
    wass_dist = []

    for board in board_names:
        print board
        simulation_data_filtered = simulation_data.loc[(simulation_data['board_name'] == board)]
        for path_length in simulation_data_filtered['path_length'].unique():
            print path_length
            paths_dict = {}
            simulation_data_board = simulation_data.loc[(simulation_data['board_name'] == board) & (simulation_data['path_length'] == path_length)]
            participant_data_board = participant_data.loc[(participant_data['board_name'] == board) & (participant_data['path_length'] == path_length)]
            data_size = participant_data_board['counts'].sum()
            probs_participants = []
            probs_sim = []
            for index, row in participant_data_board.iterrows():
                p = row['path']
                if str(p) not in paths_dict.keys():
                    paths_dict[str(p)] = 1
                prob_participant = row['probability']
                prob_sim = 0.0

                path_sim = simulation_data_board.loc[simulation_data_board['path'] == str(p)]
                if path_sim.shape[0] > 0:
                    prob_sim = path_sim['probability'].iloc[0]
                # probs_participants.append(prob_participant*data_size)
                probs_participants.append(prob_participant)
                probs_sim.append(prob_sim)
                # probs_sim.append(prob_sim*data_size)

            for index, row in simulation_data_board.iterrows():
                p = row['path']
                if str(p) not in paths_dict.keys():
                    probs_participants.append(0.0)
                    # probs_sim.append(row['probability']*data_size)
                    probs_sim.append(row['probability'])

            boards.append(board)
            path_lengths.append(path_length)
            wass_dist.append(stats.wasserstein_distance(probs_participants, probs_sim))
            print stats.wasserstein_distance(probs_participants, probs_sim)

    data_dict = {'board':boards, 'path_length': path_lengths, 'wasserstein': wass_dist}
    paths_probs_df = pd.DataFrame(data_dict)
    paths_probs_df.to_csv('stats/wasserstein_blockingBlockingSoftmax5VsParticipants.csv')


def probability_of_continuing_path():
    participants_data_all = pd.read_csv("stats/dynamics06042018.csv")
    board_names = ['6_easy_full','6_easy_pruned', '10_easy_full', '10_easy_pruned','6_hard_full','6_hard_pruned', '10_hard_full', '10_hard_pruned',  '10_medium_full', '10_medium_pruned']
    boards = []
    states = []
    scores = []
    delta_scores = []
    path_lengths = []
    state_counts = []
    state_continued = []
    prob_continue = []

    for board in board_names:
        print board
        participants_data = participants_data_all.loc[(participants_data_all['board_name'] == board)]
        state_cont = {}
        for userid in participants_data['userid'].unique():
            user_data = participants_data.loc[participants_data['userid'] == userid]
            for i, (index, row) in enumerate(user_data.iterrows()):
                board_mat = np.array(ast.literal_eval(row['board_state']))
                board_str = str(board_mat)
                if board_str not in state_cont.keys():
                    state_cont[board_str] = {'path_length': row['move_number_in_path']-1, 'score':row['score_move_x'], 'delta_score':row['delta_score'],'count':0.0, 'continued':0.0}

                state_cont[board_str]['count'] += 1.0
                if row['action'] == 'click':
                    state_cont[board_str]['continued'] += 1.0
                if i == len(user_data)-1:
                    if row['action'] == 'click':
                        state = row['board_state']
                        board_mat = np.array(ast.literal_eval(state))
                        move = row['position'].split('_')

                        board_mat[int(move[0])][int(move[1])] = row['player']
                        new_state = str(board_mat)
                        # new_state = new_state.replace('\n',',')
                        if new_state not in state_cont.keys():
                            state_cont[new_state] = {'path_length': row['move_number_in_path'], 'score':row['score_move_x'], 'delta_score':row['delta_score'],'count':1.0, 'continued':0.0}
                        else:
                            state_cont[new_state]['count'] += 1.0

        for board_state, data in state_cont.iteritems():
            boards.append(board)
            states.append(board_state)
            # scores.append(data['score'])
            # delta_scores.append(data['delta_score'])
            path_lengths.append(data['path_length'])
            state_counts.append(data['count'])
            state_continued.append(data['continued'])
            prob_continue.append(data['continued']/data['count'])

    data_dict = {'board':boards, 'path_length': path_lengths, 'state':states, 'count':state_counts, 'continued':state_continued, 'probability':prob_continue}
    state_cont_data = pd.DataFrame(data_dict)
    state_cont_data.to_csv('stats/states_continued.csv')





# def compare_distributions_simulation_population():
#     simulation_data = pd.read_csv("stats/paths_simulations2000.csv")
#     participant_data = pd.read_csv("dynamics06042018")
#
#     probs_sim = []
#     probs_participants = []
#     paths_dict = {}
#     board_names = []
#
#     for board_name in board_names:
#         path_data_participants = participant_data.loc[(participant_data['move_number_in_path'] == participant_data['move_number_in_path'].max()) & (participant_data['action'] == 'click') & (participant_data['board_name'] == board_name)]
#         path_data_simulation = simulation_data.loc[simulation_data['board_name'] == board_name]
#
#         for index, row in path_data_participants.iterrows():
#
#
#
#         if user_data.shape[0] == 0:  # no user clicks
#             continue
#         for path_num in user_data['path_number'].unique():
#             path_data = user_data.loc[(user_data['move_number_in_path'] == user_data['move_number_in_path'].max())]
#             path_str = path_data['path_after'].iloc[0]
#             path = np.array(ast.literal_eval(path_str))
#             board_sim_data = simulation_data.loc[simulation_data['board_name'] == path_data['board_name'].iloc[0]]
#             prob_path = 0.0
#             path_sim_data = board_sim_data.loc[board_sim_data['path'] == path_str]
#             if path_sim_data.shape[0] > 0:
#                 prob_path = path_sim_data['probability']
#             userids.append(userid)
#             path_nums.append(path_num)
#             paths.append(path)
#             path_lengths.append(len(path))
#             probs.append(prob_path)
#             likelihoods.append(math.log(prob_path))
#             boards.append(path_data['board_name'].iloc[0])
#
#     data_dict = {'board':boards, 'userid':userids, 'path_number': path_nums, 'path':paths, 'path_length':path_lengths, 'probability_block':probs, 'likelihood_block':likelihoods,}
#     paths_probs_df = pd.DataFrame(data_dict)
#     paths_probs_df.to_csv('stats/participants_path_probabilities_simulation.csv')

def add_score_heuristic(dynamics, scores):
    scores_heuristic = []
    clicks = dynamics.loc[dynamics['action'] == 'click']
    scores_list = []
    potential_scores = []
    for i, (index, row) in enumerate(dynamics.iterrows()):
        if row['action'] == 'click':
            state = row['board_state']

            heuristic = row['heuristic']
            state_scores = scores.loc[scores['board_state'] == state]
            # print state
            board_mat = np.array(ast.literal_eval(state))
            move = row['position'].split('_')
            # print move
            row_pos = int(move[0])
            col_pos = int(move[1])
            score = 0
            potential_score = 0
            if heuristic == 'density':
                score_mat_str = state_scores['probs_density'].iloc[0]
                scores_mat = np.array(ast.literal_eval(score_mat_str))
                score = scores_mat[row_pos][col_pos]
                board_mat[row_pos][col_pos] = int(row['player'])
                board_str = str(board_mat).replace("\n", ",")
                board_str = str(board_str).replace(" ", ", ")
                board_str = str(board_str).replace(",,", ",")
                state_scores = scores.loc[scores['board_state'] == board_str]
                # if state_scores['board_state'].shape[0] == :
                # print board_str
                score_mat_str = state_scores['probs_density'].iloc[0]
                scores_mat = np.array(ast.literal_eval(score_mat_str))
                # potential_score = np.max(scores_mat)
            elif heuristic == 'blocking':
                score_mat_str = state_scores['probs_blocking'].iloc[0]
                scores_mat = np.array(ast.literal_eval(score_mat_str))
                score = scores_mat[row_pos][col_pos]

                board_mat[row_pos][col_pos] = int(row['player'])
                board_str = str(board_mat).replace("\n", ",")
                board_str = str(board_str).replace(" ", ", ")
                board_str = str(board_str).replace(",,", ",")
                state_scores = scores.loc[scores['board_state'] == str(board_str)]
                # print board_str
                score_mat_str = state_scores['probs_blocking'].iloc[0]
                scores_mat = np.array(ast.literal_eval(score_mat_str))
                # potential_score = np.max(scores_mat)
            elif heuristic == 'interaction_blind':
                score_mat_str = state_scores['probs_interaction_oBlind'].iloc[0]
                scores_mat = np.array(ast.literal_eval(score_mat_str))
                score = scores_mat[row_pos][col_pos]
                board_mat[row_pos][col_pos] = int(row['player'])
                board_str = str(board_mat).replace("\n", ",")
                board_str = str(board_str).replace(" ", ", ")
                board_str = str(board_str).replace(",,", ",")
                state_scores = scores.loc[scores['board_state'] == str(board_str)]
                # print board_str
                score_mat_str = state_scores['probs_interaction_oBlind'].iloc[0]
                scores_mat = np.array(ast.literal_eval(score_mat_str))
            elif heuristic == 'blocking_blind':
                score_mat_str = state_scores['probs_blocking_oBlind'].iloc[0]
                scores_mat = np.array(ast.literal_eval(score_mat_str))
                score = scores_mat[row_pos][col_pos]
                board_mat[row_pos][col_pos] = int(row['player'])
                board_str = str(board_mat).replace("\n", ",")
                board_str = str(board_str).replace(" ", ", ")
                board_str = str(board_str).replace(",,", ",")
                state_scores = scores.loc[scores['board_state'] == str(board_str)]
                # print board_str
                score_mat_str = state_scores['probs_blocking_oBlind'].iloc[0]
                scores_mat = np.array(ast.literal_eval(score_mat_str))
            # player = int(row['player'])
            scores_mat[scores_mat==-0.00001] = -100000
            scores_mat[scores_mat==-0.00002] = -200000
            potential_score = np.max(scores_mat)


        scores_list.append(score)
        potential_scores.append(potential_score)

        # else:

    dynamics['score_heuristic'] = scores_list
    dynamics['potential_score_heuristic'] = potential_scores
    dynamics.to_csv('stats/moves_heuristic_scores_060518.csv')


def add_score_heuristic_list(dynamics, scores, likelihoods, heuristic_list):
    scores_list = []
    potential_scores = []
    likelihoods['fitted_heuristic'] = likelihoods[heuristic_list].idxmax(axis=1)
    heuristics_users = likelihoods[['userid','fitted_heuristic']]
    #
    # heuristics_users.set_index(['userid'], inplace=True)
    # dynamics.set_index(['userid'], inplace=True)

    # dynamics.join(heuristics_users).reset_index()
    dynamics = pd.merge(dynamics, heuristics_users, on = 'userid', how = 'left')

    for i, (index, row) in enumerate(dynamics.iterrows()):
        if row['action'] == 'click':
            state = row['board_state']

            heuristic = row['fitted_heuristic']
            state_scores = scores.loc[scores['board_state'] == state]
            # print state
            board_mat = np.array(ast.literal_eval(state))
            move = row['position'].split('_')
            # print move
            row_pos = int(move[0])
            col_pos = int(move[1])
            score = 0
            potential_score = 0

            score_mat_str = state_scores[heuristic].iloc[0]
            scores_mat = np.array(ast.literal_eval(score_mat_str))
            score = scores_mat[row_pos][col_pos]
            board_mat[row_pos][col_pos] = int(row['player'])
            board_str = str(board_mat).replace("\n", ",")
            board_str = str(board_str).replace(" ", ", ")
            board_str = str(board_str).replace(",,", ",")
            state_scores = scores.loc[scores['board_state'] == board_str]
            # if state_scores['board_state'].shape[0] == :
            # print board_str
            score_mat_str = state_scores[heuristic].iloc[0]
            # score_mat_str.replace("X", "-100000")
            # score_mat_str.replace("O", "-200000")
            scores_mat = np.array(ast.literal_eval(score_mat_str))

            # player = int(row['player'])
            # scores_mat[scores_mat=='X'] = -100000
            # scores_mat[scores_mat=='O'] = -200000
            potential_score = np.max(scores_mat)

        scores_list.append(score)
        potential_scores.append(potential_score)

    dynamics['score_heuristic'] = scores_list
    dynamics['potential_score_heuristic'] = potential_scores
    dynamics.to_csv('stats/moves_heuristic_scores_230718.csv')


def tag_last_moves_in_path(dynamics):
    last = []

    for i, (index, row) in enumerate(dynamics.iterrows()):
        if row['action'] == 'click':
            if i == dynamics.shape[0]:
                last.append(True)
            elif (dynamics['action'].iloc[i+1] == 'reset') | (dynamics['action'].iloc[i+1] == 'undo'):
                last.append(True)
            elif (dynamics['userid'].iloc[i+1] != row['userid']):
                last.append(True)
            else:
                last.append(False)
        else:
            last.append(False)
    dynamics['last_move'] = last
    dynamics.to_csv('stats/moves_hueristic_scores_last.csv')


def fit_heuristic_player_bootstrap(player_data, heuristics_list, num_samples=50, fit_threshold=-3):
    heuristics_player_counts = {}
    for heuristic in heuristics_list:
        heuristics_player_counts[heuristic] = 0.0
    for i in range(num_samples):
        num_moves_player = player_data['move_number'].max()
        moves = np.arange(1,num_moves_player+1)
        train_moves, test_moves = train_test_split(moves, test_size=0.3)  # random split
        sampled_moves = player_data.loc[player_data['move_number'].isin(train_moves)]
        if sampled_moves.shape[0] < 1:
            return None
        max_log_likeilhood = -10000
        fitted_heuristic = None
        for heuristic in heuristics_list:
            # heuristic_likelhioods = player_data.loc[player_data['heuristic'] == heuristic]
            heuristic_likelhioods = sampled_moves.loc[sampled_moves['heuristic'] == heuristic]
            mean_likeilhood = heuristic_likelhioods['log_move'].mean()
            if mean_likeilhood > max_log_likeilhood:
                max_log_likeilhood = mean_likeilhood
                if heuristic is None:
                    print 'problem'
                fitted_heuristic = heuristic
        heuristics_player_counts[fitted_heuristic] += 1.0

    best_heuristic = None
    max_fitted_heuristic = -10000
    for heuristic in heuristics_list:
        if heuristics_player_counts[heuristic] > max_fitted_heuristic:
            heuristic_likelhioods = player_data.loc[player_data['heuristic'] == heuristic]
            best_heuristic = heuristic
            max_fitted_heuristic = heuristic_likelhioods['log_move'].mean()
    if max_fitted_heuristic > fit_threshold:
        return best_heuristic
    else:
        return None


def compute_blindness_player(player_moves_probs, player=2, include_forced=False):
    player_moves_probs = player_moves_probs.loc[player_moves_probs['player'] == player]
    if not include_forced:
        player_moves_probs = player_moves_probs.loc[player_moves_probs['prob_move'] < 1]

    return player_moves_probs['prob_move'].mean(), player_moves_probs['prob_move'].median(), player_moves_probs['rank_move'].mean(), player_moves_probs['rank_move'].median(), player_moves_probs['prob_move_to_best_ratio'].mean(), player_moves_probs['prob_move_to_best_ratio'].median()


if __name__== "__main__":
    sns.set(style="whitegrid")
    # states_cont = pd.read_csv("stats/states_continued.csv")
    # states_cont_filtered = states_cont.loc[(states_cont['count'] > 4) & (states_cont['path_length'] <7) & (states_cont['path_length'] > 0)]
    # s = states_cont_filtered['count']
    # v = states_cont_filtered['probability']
    # # s = np.float_(np.array(range(0,1201,1200/8)))
    # # v = np.round(120*s/(171+s) + np.random.uniform(size=9), 2)
    # p = Parameters()
    # p.add('vmax', value=1., min=0.)
    # p.add('km', value=1., min=0.)
    #
    # out = lmmin(residual, p, args=(s, v))
    # sns.regplot(s,v, fit_reg = False)
    # ss = np.float_(np.array(range(0,300,1)))
    # y = out.params['vmax'].value * ss / (out.params['km'].value + ss)
    #
    # # y =  ss / (out.params['km'].value + ss)
    # sns.regplot(ss,y, fit_reg = False, color='red')
    # # plt.show()
    # print out.params
    # plt.show()
    # print 1/0
    # plot(s, v, 'bo')
    # hold(True)
    #
    # ss = np.float_(np.array(range(0,1201,1200/100)))
    # y = p['vmax'].value * ss / (p['km'].value + ss)
    # plot(ss, y, 'r-')
    # hold(False)
    # compute_path_probabilities_participants();

    # print stats.wasserstein_distance([1,2,7], [3,1,6])
    # print stats.wasserstein_distance([0.1,0.2,0.7], [0.3,0.1,0.6])
    # compare_distributions_simulation_population();
    # probability_of_continuing_path()
    # states_cont = pd.read_csv("stats/states_continued.csv")
    # # # & (states_cont['board'] == '6_hard_full')
    # states_cont_filtered = states_cont.loc[(states_cont['count'] > 4) & (states_cont['path_length'] <7) & (states_cont['path_length'] > 0)]
    # ax = sns.regplot(x="count", y="probability", data=states_cont_filtered, n_boot=1000)
    # # plt.hist(states_cont_filtered['probability'])
    # # g = sns.FacetGrid(states_cont_filtered, col="path_length", legend_out=False)
    # # g.map(plt.hist, "probability", color="steelblue",  lw=0)
    # # bins = np.linspace(-100, 100, 200)
    # # g.map(plt.hist, "score_move_x", color="steelblue", bins=bins, lw=0)
    # plt.show()
    # print 1/0

    data = pd.read_csv("stats/cogSci.csv")
    mctsData = pd.read_csv("stats/mctsRuns.csv")
    dataEntropy = pd.read_csv("stats/cogSciEntropy.csv")
    alphaBetaFull = pd.read_csv("stats/cogsciAlphaBeta100000.csv")
    alphaBeta50 = pd.read_csv("stats/alphaBest50_success.csv")
    distances = pd.read_csv("stats/distanceFirstMoves.csv")
    population = pd.read_csv("stats/cogsciPopulation1.csv")
    likelihood = pd.read_csv("stats/logLikelihood.csv")
    dynamics = pd.read_csv("stats/moves_hueristic_scores_shutter230718.csv")
    # dynamics = pd.read_csv("stats/moves_hueristic_scores_aperture.csv")
    transitions = pd.read_csv("stats/state_scores_heuristics_180718normalized_all.csv",dtype = {'board_state':str})
    scores = pd.read_csv("stats/state_scores_heuristics_180718raw_all_xo.csv",dtype = {'board_state':str})
    likelihoods = pd.read_csv("stats/heuristics_fitted_230718_all.csv")
    cogsci_participants = pd.read_csv("stats/cogsci_participants.csv")
    heuristics_sensitivity = pd.read_csv("stats/cogsci_heuristics250718.csv")

    cogsci_part_list = cogsci_participants['userid'].unique().tolist()
    # heuristics_part_list = heuristics_sensitivity['userid'].unique().tolist()
    # dynamics_filtered = dynamics.loc[dynamics['userid'].isin(cogsci_part_list)]
    # dynamics_filtered.to_csv('stats/dynamics_cogscidata_230718.csv')

    # dynamics_filtered = pd.read_csv('stats/dynamics_cogscidata_220718.csv')
    # dynamics_list = dynamics_filtered['userid'].unique().tolist()
    # print len(heuristics_part_list)
    # print len(dynamics_list)
    # for part in heuristics_part_list:
    #     if part not in dynamics_list:
    #         print part
    #         # print heuristics_sensitivity.loc[heuristics_sensitivity['userid'] == part]

    # heuristics_sensitivity_filtered = heuristics_sensitivity.loc[heuristics_sensitivity['userid'].isin(cogsci_part_list)]
    # # # cogsci_heuristics = pd.merge(cogsci_participants, heuristics_sensitivity, on = 'userid', how = 'left')
    # heuristics_sensitivity_filtered.to_csv('stats/cogsci_heuristics250718.csv')

    # fit_heuristic_user(transitions,dynamics)
    # add_score_heuristic(dynamics,scores)
    # tag_last_moves_in_path(dynamics)
    # merge_undos_to_reset(dynamics)
    # add_aperture_values(dynamics)
    # fit_heuristic_user_list(transitions,dynamics,['density','linear','non-linear','interaction','blocking','interaction_blind','blocking_blind'],'stats/heuristics_fitted_230718_all.csv')
    # add_score_heuristic_list(dynamics,scores,likelihoods,['density','non-linear','interaction','blocking','interaction_blind','blocking_blind'])
    # unique_users = dynamics.drop_duplicates(subset='userid', keep='first', inplace=False)
    # unique_users.to_csv('stats/users_heuristics.csv')
    # print 1/0

    # states = pd.read_csv("stats/states.csv")
    dynamicsFirstMoves = pd.read_csv("stats/dynamicsFirstMoves1.csv")
    # compare_start_move(dynamics)
    # probs_clicks(dynamicsFirstMoves)
    # print 1/0

    # exploreExploit = pd.read_csv("stats/exploreExploit0311_avg.csv")
    # exploreExploit = pd.read_csv("stats/exploreExploitPath_avg.csv")
    exploreExploit = pd.read_csv("stats/explore_exploit_avg_1604.csv")
    # exploreExploit2 = pd.read_csv("stats/exploreExploitData2.csv")
    # exploreExploit2 = pd.read_csv("stats/exploreExploitDataNoUndo.csv")
    timeResets = pd.read_csv("stats/timeBeforeReset.csv")
    timeUndos = pd.read_csv("stats/timeBeforeUndo.csv")
    resetsData = pd.read_csv("stats/resetsData.csv")
    # resetsDelta = pd.read_csv("stats/resetsDeltaData.csv")
    # resetsDelta = pd.read_csv("stats/actionsLogDelta_blocking_abs.csv")
    resetsDelta = pd.read_csv("stats/resetsFiltered2.csv")
    move_probabilities_heuristics = pd.read_csv('stats/heuristics_byMove_player_cogsci_withPaths.csv')  # only cogsci data
    # move_probabilities_heuristics = pd.read_csv('stats/heuristics_byMove_player_allData.csv')  # all data

    userids_sig = []
    fitted_heuristics_sig = []





    # ----shutters alpha beta

 # #
 # #    # # non-linear and blockign heuristics data
 # #    # heuristic_name = 'interaction'
 # #    # for k in ['k7']:
 # #    #     first = True
 # #    #     for filename in os.listdir('stats/ab_pareto/' + heuristic_name):
 # #    #         if k in filename:
 # #    #             if first:
 # #    #                 print filename
 # #    #                 alpha_beta_shutter = pd.read_csv('stats/ab_pareto/' + heuristic_name + '/' + filename)
 # #    #                 first = False
 # #    #             else:
 # #    #                 print filename
 # #    #                 new_df = pd.read_csv('stats/ab_pareto/' + heuristic_name + '/' + filename)
 # #    #                 alpha_beta_shutter = alpha_beta_shutter.append(copy.deepcopy(new_df))
 # #    #     alpha_beta_shutter.to_csv('stats/ab_pareto/' + heuristic_name + '_' + k + '_agg.csv')
 # #    #
 # #    # exit()
 # #
 # # #    # alpha_beta_shutter = pd.read_csv('stats/ab_shutter_stochastic_080818_6boards.csv')
 # # #    # alpha_beta_shutter = pd.read_csv('stats/ab_config_shutter_cogsci_110818_allBoards_moveLimit.csv')
 # # #    # # alpha_beta_shutter = pd.read_csv('stats/ab_config_shutter_cogsci_120818_allBoards_limitHeuristic.csv')
 # # #    # alpha_beta_shutter = pd.read_csv('stats/ab_config_shutter_cogsci_130818_allBoards_limitHeuristicLimitMoves.csv')
 # #    # alpha_beta_shutter_k3 = pd.read_csv('stats/ab_pareto/ab_tradeoff_allBoards_moveLimit_agg_k3.csv')
 # #    # alpha_beta_shutter_k5 = pd.read_csv('stats/ab_pareto/ab_tradeoff_allBoards_moveLimit_agg_k5.csv')
 # #    # alpha_beta_shutter_k10 = pd.read_csv('stats/ab_pareto/ab_tradeoff_allBoards_moveLimit_agg_k10.csv')
 #    alpha_beta_shutter_k3 = pd.read_csv('stats/ab_pareto/ab_tradeoff_allBoards_moveLimit_agg_k3_noiseAnalysis.csv')
 #    alpha_beta_shutter_k5 = pd.read_csv('stats/ab_pareto/ab_tradeoff_allBoards_moveLimit_agg_k5_noiseAnalysis.csv')
 #    alpha_beta_shutter_k10 = pd.read_csv('stats/ab_pareto/ab_tradeoff_allBoards_moveLimit_agg_k10_noiseAnalysis.csv')
 #    alpha_beta_shutter_k7 = pd.read_csv('stats/ab_pareto/interaction_k7_agg.csv')
 # #    alpha_beta_shutter_k3 = pd.read_csv('stats/ab_pareto/nonlinear_k3_agg.csv')
 # #    alpha_beta_shutter_k5 = pd.read_csv('stats/ab_pareto/nonlinear_k5_agg.csv')
 # #    alpha_beta_shutter_k10 = pd.read_csv('stats/ab_pareto/nonlinear_k10_agg.csv')
 # # # #    # print alpha_beta_shutter_k3.columns.values
 # # # #    # print alpha_beta_shutter_k5.columns.values
 # # # #    # print alpha_beta_shutter_k10.columns.values
 #    frames = [alpha_beta_shutter_k3, alpha_beta_shutter_k5, alpha_beta_shutter_k7, alpha_beta_shutter_k10]
 #    # frames = [alpha_beta_shutter_k10, alpha_beta_shutter_k5]
 #    alpha_beta_shutter = pd.concat(frames, sort=False)
 # #
 # #
 #
 #    # inverse_computation = alpha_beta_shutter['numberOfHeuristicComp'].apply(lambda x: 1.0/x)
 #    # alpha_beta_shutter_pareto['inverse_computation'] = inverse_computation
 #
 #    aggregations = {
 #        'correct': ['mean','sem'],
 #        'numberOfHeuristicComp':['mean','sem']
 #    }
 #    # for board in
 #    # alpha_beta_shutter = alpha_beta_shutter.loc[alpha_beta_shutter['board'].isin(['10_medium_full'])]
 #    alpha_beta_shutter_pareto = alpha_beta_shutter.groupby(['max_moves','noise_sig','k','shutter_size','board']).agg(aggregations).reset_index()
 #    # print alpha_beta_shutter_pareto.columns.values
 #    alpha_beta_shutter_pareto.columns = ['max_moves','noise_sig','k','shutter_size','board','numberOfHeuristicComp', 'numberOfHeuristicComp_sem', 'correct', 'correct_sem']
 #    # print alpha_beta_shutter_pareto.columns.values
 #    # alpha_beta_shutter_pareto = alpha_beta_shutter_pareto.loc[(alpha_beta_shutter_pareto['board'] == '6_easy_full') | (alpha_beta_shutter_pareto['board'] == '6_easy_pruned')]
 # #    alpha_beta_shutter_pareto.map({"('max_moves', '')":'max_moves', "('noise_sig', '') ('k', '')":'noise_sig' ('shutter_size', '')
 # # ('board', '') ('numberOfHeuristicComp', 'mean')})
 # # ('numberOfHeuristicComp', 'sem') ('correct', 'mean') ('correct', 'sem')])
 #    inverse_computation = alpha_beta_shutter_pareto['numberOfHeuristicComp'].apply(lambda x: 1.0/x)
 #    alpha_beta_shutter_pareto['inverse_computation'] = inverse_computation
 #
 #    inverse_computation_all = alpha_beta_shutter['numberOfHeuristicComp'].apply(lambda x: 1.0/x)
 #    alpha_beta_shutter['inverse_computation'] = inverse_computation_all
 #
 #    # pareto board specific figures pnas
 #    alpha_beta_tradeoff_config = alpha_beta_shutter.loc[(alpha_beta_shutter['max_moves']==30) & (alpha_beta_shutter['noise_sig']==0.5) & (alpha_beta_shutter['k']==5) & (alpha_beta_shutter['board']=='6_easy_full')]
 #    first = True
 #    for i in range(1):
 #
 #        alpha_beta_tradeoff_config = alpha_beta_shutter.loc[(alpha_beta_shutter['max_moves']==30) & (alpha_beta_shutter['noise_sig']==0.5) & (alpha_beta_shutter['k']==5) & (alpha_beta_shutter['board']=='6_easy_full')]
 #        alpha_beta_tradeoff_config_sub = alpha_beta_tradeoff_config.sample(frac=1.0)
 #        alpha_beta_shutter_pareto_config_sub = alpha_beta_tradeoff_config_sub.groupby(['shutter_size']).agg(aggregations).reset_index()
 #        alpha_beta_shutter_pareto_config_sub.columns = ['shutter_size','numberOfHeuristicComp', 'numberOfHeuristicComp_sem', 'correct', 'correct_sem']
 #        # print alpha_beta_shutter_pareto_config_sub['numberOfHeuristicComp'].values
 #        inverse_computation_sub = alpha_beta_shutter_pareto_config_sub['numberOfHeuristicComp'].apply(lambda x: 1.0/x)
 #        # inverse_computation_sub = alpha_beta_shutter_pareto_config_sub['numberOfHeuristicComp'].apply(lambda x: 1-(x/alpha_beta_shutter_pareto_config_sub['numberOfHeuristicComp'].values.max()))
 #        alpha_beta_shutter_pareto_config_sub['inverse_computation_sub'] = copy.deepcopy(inverse_computation_sub)
 #        if first:
 #            alpha_beta_tradeoff_config_pareto = copy.deepcopy(alpha_beta_shutter_pareto_config_sub)
 #            first = False
 #        else:
 #            alpha_beta_tradeoff_config_pareto = alpha_beta_tradeoff_config_pareto.append(alpha_beta_shutter_pareto_config_sub)
 #    alpha_beta_no_tradeoff_config = alpha_beta_shutter_pareto.loc[(alpha_beta_shutter_pareto['max_moves']==30) & (alpha_beta_shutter_pareto['noise_sig']==0.5) & (alpha_beta_shutter_pareto['k']==5) & (alpha_beta_shutter_pareto['board']=='6_hard_full')]
 #
 #    alpha_beta_no_tradeoff_config_all = alpha_beta_shutter.loc[(alpha_beta_shutter['max_moves']==30) & (alpha_beta_shutter['noise_sig']==0.5) & (alpha_beta_shutter['k']==5) & (alpha_beta_shutter['board']=='6_hard_full')]
 #    # # alpha_beta_no_tradeoff_config.to_csv('stats/tradeoff_example.csv')
 #    # # alpha_beta_no_tradeoff_config = pd.read_csv('stats/tradeoff_example.csv')
 #    # inverse_computation_sub = alpha_beta_no_tradeoff_config['numberOfHeuristicComp'].apply(lambda x: 1-(x/alpha_beta_no_tradeoff_config['numberOfHeuristicComp'].values.max()))
 #    # # # inverse_computation_sub = alpha_beta_shutter_pareto_config_sub['numberOfHeuristicComp'].apply(lambda x: 1-(x/alpha_beta_shutter_pareto_config_sub['numberOfHeuristicComp'].values.max()))
 #    alpha_beta_no_tradeoff_config['inverse_computation'] = alpha_beta_no_tradeoff_config['numberOfHeuristicComp'].apply(lambda x: 1-(x/alpha_beta_no_tradeoff_config['numberOfHeuristicComp'].values.max()))
 #    alpha_beta_no_tradeoff_config_all['inverse_computation'] = alpha_beta_no_tradeoff_config_all['numberOfHeuristicComp'].apply(lambda x: 1.0-(float(x)/alpha_beta_no_tradeoff_config_all['numberOfHeuristicComp'].values.max()))
 #
 #
 #    # for index, row in alpha_beta_no_tradeoff_config_all.iterrows():
 #
 #
 #    aggregations = {
 #        'correct': ['mean','sem'],
 #        'inverse_computation':['mean','sem']
 #    }
 #    # for board in
 #    # alpha_beta_shutter = alpha_beta_shutter.loc[alpha_beta_shutter['board'].isin(['10_medium_full'])]
 #    alpha_beta_shutter_pareto_no_tradeoff = alpha_beta_no_tradeoff_config_all.groupby(['shutter_size']).agg(aggregations).reset_index()
 #    # alpha_beta_shutter_pareto_no_tradeoff.columns = ['shutter_size','inverseComputationMean', 'inverseComputationSem', 'correct', 'correct_sem']
 #
 #    sns.set(font_scale=1.2, style='whitegrid')
 #    sns.set(style='whitegrid')
 #
 #    sns.set(rc={"font.style":"normal",
 #                "axes.facecolor":(0.9, 0.9, 0.9),
 #                "figure.facecolor":'white',
 #                "grid.color":'black',
 #                "grid.linestyle":':',
 #                "axes.grid":True,
 #                'axes.labelsize':30,
 #                'figure.figsize':(20.0, 10.0),
 #                'xtick.labelsize':25,
 #                'ytick.labelsize':20})
 #    sns.set(style='whitegrid')
 #    # ax = sns.lmplot(x="inverse_computation", y="correct", hue='shutter_size', data=alpha_beta_no_tradeoff_config,  fit_reg=False, scatter_kws={"s": 70}, size=5, aspect=1, palette='colorblind')
 #    # ax.set(xticklabels=[0,0.5,1.0])
 #    # plt.xlim(-0.05,1)
 #    colors = sns.color_palette("hls", 4)
 #
 #    x = alpha_beta_shutter_pareto_no_tradeoff[('inverse_computation','mean')]
 #    print x
 #    y = alpha_beta_shutter_pareto_no_tradeoff[('correct','mean')]
 #    print y
 #    x_err = alpha_beta_shutter_pareto_no_tradeoff[('inverse_computation','sem')]
 #    print x_err
 #    y_err = alpha_beta_shutter_pareto_no_tradeoff[('correct','sem')]
 #    print y_err
 #    # plt.show()
 #    figure, axs = plt.subplots(2,1,figsize=(4.5,5))
 #
 #    # plt.figure(figsize=(4,2))
 #    # plt.errorbar(x, y, yerr=y_err, xerr=x_err,
 #    #         capthick=2, fmt='none', ecolor='#B2BABB',zorder=1)
 #    plt.subplot(2,1,1)
 #    for i in range(4):
 #        plt.errorbar(x[i], y[i], xerr=x_err[i], yerr=y_err[i], lw=2, capsize=5, capthick=2, color=colors[i],zorder=1)
 #
 #    # plt.scatter(x, y, s=20, color=colors)
 #    # ax = sns.scatterplot(x="inverse_computation", y="correct", hue='shutter_size', data=alpha_beta_no_tradeoff_config)
 #    # plt.ylim(0,1)
 #    # plt.xlim(-0.05,1)
 #    # plt.show()
 #
 #    color_dict = {0: colors[0], 0.5: colors[1], 1: colors[2], 2: colors[3]}
 #    alpha_beta_no_tradeoff_config['color'] = alpha_beta_no_tradeoff_config['shutter_size'].apply(lambda x: color_dict[x])
 #
 #
 #    # plt.show()
 #    # fig = plt.figure()
 #
 #
 #
 #    # figure, axs = plt.subplots(2,1,figsize=(3.5,5))
 #    # fig = plt.subplot(2,1,1)
 #    #
 #    # ax = sns.regplot(x="inverse_computation", y="correct", data=alpha_beta_no_tradeoff_config,  fit_reg=False, scatter_kws={"s": 70, 'facecolors':alpha_beta_no_tradeoff_config['color']})
 #    #
 #    #
 #    # legend_elements = [Line2D([0], [0], marker='o', color='w', label='0',
 #    #                   markerfacecolor=color_dict[0], markersize=10),
 #    #                Line2D([0], [0], marker='o', color='w', label='0.5',
 #    #                   markerfacecolor=color_dict[0.5], markersize=10),
 #    #               Line2D([0], [0], marker='o', color='w', label='1',
 #    #                   markerfacecolor=color_dict[1], markersize=10),
 #    #                Line2D([0], [0], marker='o', color='w', label='2',
 #    #                   markerfacecolor=color_dict[2], markersize=10)]
 #    # fig.legend(labels=['shutter=0','shutter=0.5','shutter=1','shutter=2'], handles=legend_elements)
 #    # ax.legend(loc="best")
 #    # # plt.show()
 #    # ax.set_xlabel('Computation', fontsize=14)
 #    # ax.set_ylabel('Correctness', fontsize=14)
 #    # ax.tick_params(labelsize=12)
 #    alpha_beta_tradeoff_config = alpha_beta_shutter_pareto.loc[(alpha_beta_shutter_pareto['max_moves']==30) & (alpha_beta_shutter_pareto['noise_sig']==0.5) & (alpha_beta_shutter_pareto['k']==5) & (alpha_beta_shutter_pareto['board']=='6_easy_full')]
 #
 #    alpha_beta_tradeoff_config_all = alpha_beta_shutter.loc[(alpha_beta_shutter['max_moves']==30) & (alpha_beta_shutter['noise_sig']==0.5) & (alpha_beta_shutter['k']==5) & (alpha_beta_shutter['board']=='6_easy_full')]
 #
 #    # inverse_computation_sub = alpha_beta_tradeoff_config['numberOfHeuristicComp'].apply(lambda x: 1-(x/alpha_beta_no_tradeoff_config['numberOfHeuristicComp'].values.max()))
 #    # inverse_computation_sub = alpha_beta_shutter_pareto_config_sub['numberOfHeuristicComp'].apply(lambda x: 1-(x/alpha_beta_shutter_pareto_config_sub['numberOfHeuristicComp'].values.max()))
 #    alpha_beta_tradeoff_config['inverse_computation'] = alpha_beta_tradeoff_config['numberOfHeuristicComp'].apply(lambda x: 1-(x/alpha_beta_tradeoff_config['numberOfHeuristicComp'].values.max()))
 #    alpha_beta_tradeoff_config_all['inverse_computation'] = alpha_beta_tradeoff_config_all['numberOfHeuristicComp'].apply(lambda x: 1.0-(float(x)/alpha_beta_tradeoff_config_all['numberOfHeuristicComp'].values.max()))
 #
 #    alpha_beta_shutter_pareto_tradeoff = alpha_beta_tradeoff_config_all.groupby(['shutter_size']).agg(aggregations).reset_index()
 #    colors = sns.color_palette("hls", 4)
 #    color_dict = {0: colors[0], 0.5: colors[1], 1: colors[2], 2: colors[3]}
 #    alpha_beta_tradeoff_config['color'] = alpha_beta_tradeoff_config['shutter_size'].apply(lambda x: color_dict[x])
 #
 #    x = alpha_beta_shutter_pareto_tradeoff[('inverse_computation','mean')]
 #    print x
 #    y = alpha_beta_shutter_pareto_tradeoff[('correct','mean')]
 #    print y
 #    x_err = alpha_beta_shutter_pareto_tradeoff[('inverse_computation','sem')]
 #    print x_err
 #    y_err = alpha_beta_shutter_pareto_tradeoff[('correct','sem')]
 #    print y_err
 #    # plt.show()
 #    # plt.figure(figsize=(4,2))
 #    # plt.errorbar(x, y, yerr=y_err, xerr=x_err,
 #    #         capthick=2, fmt='none', ecolor='#B2BABB',zorder=1)
 #    fig = plt.subplot(2,1,2)
 #    fig.tick_params(axis='both', which='major', labelsize=10)
 #    for i in range(4):
 #        plt.errorbar(x[i], y[i], xerr=x_err[i], yerr=y_err[i], lw=2, capsize=5, capthick=2, color=colors[i],zorder=1)
 #
 #    # plt.subplot(1,2,2)
 #    # plt.figure(figsize=(3,3))
 #    # ax = sns.lmplot(x="inverse_computation", y="correct", hue='shutter_size', data=alpha_beta_tradeoff_config,  fit_reg=False, scatter_kws={"s": 70}, size=5, aspect=1, palette='colorblind')
 #
 #    # fig = plt.figure()
 #    # fig = plt.subplot(2,1,2)
 #    # ax = sns.regplot(x="inverse_computation", y="correct", data=alpha_beta_tradeoff_config,  fit_reg=False, scatter_kws={"s": 70, 'facecolors':alpha_beta_tradeoff_config['color']})
 #    # ax.set_xlabel('Computation', fontsize=14)
 #    # ax.set_ylabel('Correctness', fontsize=14)
 #    # ax.tick_params(labelsize=12)
 #    # legend_elements = [Line2D([0], [0], marker='o', color='w', label='0',
 #    #                   markerfacecolor=color_dict[0], markersize=10),
 #    #                Line2D([0], [0], marker='o', color='w', label='0.5',
 #    #                   markerfacecolor=color_dict[0.5], markersize=10),
 #    #               Line2D([0], [0], marker='o', color='w', label='1',
 #    #                   markerfacecolor=color_dict[1], markersize=10),
 #    #                Line2D([0], [0], marker='o', color='w', label='2',
 #    #                   markerfacecolor=color_dict[2], markersize=10)]
 #    # fig.legend(labels=['shutter=0','shutter=0.5','shutter=1','shutter=2'], handles=legend_elements)
 #    # fig.legend(loc=2)
 #    # # plt.savefig("stats/pareto/grid_pareto_170818.png", format='png')
 #    # # plt.ylim(0.25,0.3)
 #    # # plt.xlim(0,0.015)
 #    # # plt.xlim(-0.05,1)
 #    plt.tight_layout(pad=2.5)
 #    # plt.rcParams.update({'font.size': 20})
 #    # plt.rc('xtick', labelsize=20)
 #    # plt.rc('ytick', labelsize=20)
 #    plt.show()
 #    exit()
 # #
 # #    # exit()
 #    move_limits = alpha_beta_shutter_pareto['max_moves'].unique()
 #    # move_limits = [30,50,100]
 #    noise_vals = alpha_beta_shutter_pareto['noise_sig'].unique()
 #    # noise_vals = [0.5,1.0,1.5,2.0,2.5]
 #    k_vals = alpha_beta_shutter_pareto['k'].unique()
 #    # k_vals = [3,5,10]
 #    shutter_sizes = alpha_beta_shutter_pareto['shutter_size'].unique()
 #    boards = alpha_beta_shutter_pareto['board'].unique()
 #    out = []
 #    moves = []
 #    noises = []
 #    boards_df = []
 #    shutters = []
 #    corrects = []
 #    comp = []
 #    ks = []
 #    for m in move_limits:
 #        for n in noise_vals:
 #            for k in k_vals:
 #                for b in boards:
 #                    alpha_beta_shutter_pareto_conf = alpha_beta_shutter_pareto.loc[(alpha_beta_shutter_pareto['max_moves']==m) & (alpha_beta_shutter_pareto['noise_sig']==n) & (alpha_beta_shutter_pareto['k']==k)& (alpha_beta_shutter_pareto['board']==b)]
 #                    # if alpha_beta_shutter_pareto_conf.shape[0] == 0:
 #                    #     continue
 #                    for i in range(len(shutter_sizes)):
 #                        dominated = False
 #                        shutter_data_i = alpha_beta_shutter_pareto_conf.loc[alpha_beta_shutter_pareto_conf['shutter_size'] == shutter_sizes[i]]
 #                        if shutter_data_i.shape[0] == 0:
 #                            print 'problem'
 #                        i_comp_s = shutter_data_i['inverse_computation'].iloc[0]
 #                        i_comp_s_raw = shutter_data_i['numberOfHeuristicComp'].iloc[0]
 #                        i_comp_s_upper = shutter_data_i['numberOfHeuristicComp'].iloc[0] + 1.*shutter_data_i['numberOfHeuristicComp_sem'].iloc[0]
 #                        i_comp_s_lower = shutter_data_i['numberOfHeuristicComp'].iloc[0] - 1.*shutter_data_i['numberOfHeuristicComp_sem'].iloc[0]
 #                        i_correct_s = shutter_data_i['correct'].iloc[0]
 #                        i_correct_s_upper = shutter_data_i['correct'].iloc[0] + 1.*shutter_data_i['correct_sem'].iloc[0]
 #                        i_correct_s_lower = shutter_data_i['correct'].iloc[0] - 1.*shutter_data_i['correct_sem'].iloc[0]
 #                        for j in range(len(shutter_sizes)):
 #                            if j!=i:
 #                                shutter_data_j = alpha_beta_shutter_pareto_conf.loc[alpha_beta_shutter_pareto_conf['shutter_size'] == shutter_sizes[j]]
 #                                if shutter_data_j.shape[0] == 0:
 #                                    print 'problem'
 #                                j_comp_s = shutter_data_j['inverse_computation'].iloc[0]
 #                                j_comp_s_raw = shutter_data_j['numberOfHeuristicComp'].iloc[0]
 #                                j_comp_s_upper = shutter_data_j['numberOfHeuristicComp'].iloc[0] + 1.*shutter_data_j['numberOfHeuristicComp_sem'].iloc[0]
 #                                j_comp_s_lower = shutter_data_j['numberOfHeuristicComp'].iloc[0] - 1.*shutter_data_j['numberOfHeuristicComp_sem'].iloc[0]
 #                                j_correct_s = shutter_data_j['correct'].iloc[0]
 #                                j_correct_s_upper = shutter_data_j['correct'].iloc[0] + 1.*shutter_data_j['correct_sem'].iloc[0]
 #                                j_correct_s_lower = shutter_data_j['correct'].iloc[0] - 1.*shutter_data_j['correct_sem'].iloc[0]
 #                                if i_comp_s_lower >= j_comp_s_upper:
 #                                     if not (i_correct_s_lower >= j_correct_s_upper):
 #                                        dominated = True
 #                                elif i_correct_s_upper <= j_correct_s_lower:
 #                                    if not i_comp_s_upper <= j_comp_s_lower:
 #                                        dominated = True
 #                                # elif shutter_sizes[i] == 2.0:
 #                                #     print 'large shutter'
 #                        if not dominated:
 #                            # if shutter_sizes[i] == 2.0:
 #                            #     print 'large shutter'
 #                            moves.append(m)
 #                            noises.append(n)
 #                            ks.append(k)
 #                            boards_df.append(b)
 #                            shutters.append(shutter_sizes[i])
 #                            corrects.append(i_correct_s)
 #                            comp.append(i_comp_s)
 #                        # else:
 #                        #     print 'removing shutter ' + str(shutter_sizes[i])
 #                            # if
 #
 #    data_dict = {'move_limit': moves, 'noise': noises, 'board': boards_df, 'shutter_size': shutters, 'correct': corrects, 'inverse_computation': comp, 'k': ks}
 #    data_df = pd.DataFrame(data_dict)
 #
 #    # alpha_beta_shutter_pareto.to_csv('stats/ab_pareto_no_board.csv')
 #    # alpha_beta_shutter_pareto = pd.read_csv('stats/ab_pareto_no_board.csv')
 #
 #    # data_df.to_csv('stats/filtered_pareto_tradeoffs.csv')
 #    # data_df = pd.read_csv('stats/filtered_pareto_tradeoffs.csv')
 # #
 #    data_df.to_csv('stats/pareto_interaction_all.csv')
 #    data_df = pd.read_csv('stats/pareto_interaction_all.csv')
 # # #
 # # # #    # # plot alpha-beta
 # # # #    # ax = sns.lmplot(x="inverse_computation", y="correct", hue='shutter_size', col='board', col_wrap=2, data=data_df, fit_reg=False)
 # # # #    # plt.savefig("stats/pareto/grid_pareto_170818.png", format='png')
 # # # #    # plt.ylim(0,1)
 # # # #    # plt.xlim(0,0.03)
 # # # #    # plt.show()
 # # # #
 # # # #    # compute pareto stats to see whether there is tradeoff or not
 # # # #
 #    pareto_tradeoffs_df = data_df.groupby(['move_limit','noise','k','board']).shutter_size.nunique().reset_index()
 #    pareto_tradeoffs_df.columns = ['move_limit','noise','k','board','tradeoff']
 #    pareto_tradeoffs_df['tradeoff'] = pareto_tradeoffs_df['tradeoff'].apply(lambda x: min(x-1,1))
 #    pareto_tradeoffs_df.to_csv('stats/pareto_tradeoffs_interaction_all.csv')
 # # # #
 # # #
 #    exit()
 # #
 #    alpha_beta_shutter_pareto = pd.read_csv('stats/pareto_tradeoffs_nonlinear100.csv')
 #    alpha_beta_shutter_pareto = pd.read_csv('stats/pareto_tradeoffs_blocking.csv')
 #    alpha_beta_shutter_pareto= pd.read_csv('stats/pareto_tradeoffs_noiseAnalysisAll.csv')
 #    alpha_beta_shutter_pareto= pd.read_csv('stats/pareto_tradeoffs_interaction_all.csv')
 #    # alpha_beta_shutter_pareto_with_complexity = pd.read_csv('stats/pareto_noises_test2.csv')
 #    alpha_beta_shutter_pareto_noises = alpha_beta_shutter_pareto.groupby(['board', 'noise']).tradeoff.mean().reset_index()
 #    alpha_beta_shutter_pareto_noises.to_csv('stats/pareto_noises_interaction_all.csv')
 #    # alpha_beta_shutter_pareto_noises.to_csv('stats/pareto_noises_test2.csv')
 #    # alpha_beta_shutter_pareto_noises.to_csv('stats/pareto_noises_test2_blocking.csv')
 #    exit()
 #    # alpha_beta_shutter_pareto_noises = pd.merge(alpha_beta_shutter_pareto_noises, alpha_beta_shutter_pareto_with_complexity[['board','complexity']], on = 'board', how = 'left')
 #    alpha_beta_shutter_pareto_noises.to_csv('stats/pareto_noises_test2.csv')
 #    exit()
 #    alpha_beta_shutter_pareto = pd.read_csv('stats/pareto_tradeoffs_noiseAnalysisCI.csv')
 #    alpha_beta_shutter_pareto_noises = alpha_beta_shutter_pareto.groupby(['board', 'noise']).tradeoff.mean().reset_index()
    # alpha_beta_shutter_pareto_noises.to_csv('stats/pareto_noises_test_CI.csv')
    # alpha_beta_shutter_pareto_noises = pd.read_csv('stats/pareto_noises_interaction_all.csv')
    # alpha_beta_shutter_pareto_noises = pd.read_csv('stats/pareto_noises_test2_blocking.csv')
    # alpha_beta_shutter_pareto_noises = alpha_beta_shutter_pareto_noises.loc[alpha_beta_shutter_pareto_noises['moves']!=200]
    # result = alpha_beta_shutter_pareto_noises.pivot(index='noise', columns='complexity', values='tradeoff')
    # grid_x, grid_y = np.mgrid[0:100000:100j, 0:2.5:100j]
    # result2 = copy.deepcopy(result)
    # result_mat = result2.as_matrix()
    # result_mat2 = copy.deepcopy(result_mat)
    # # guassian_kernel = []
    # for r in range(len(result_mat)):
    #     for c in range(len(result_mat[r])):
    #         sum_vals = result_mat[r][c]
    #         counts = 1.0
    #         if r+1 < len(result_mat)-1:
    #             sum_vals+=result_mat[r+1][c]
    #             counts += 1
    #             if c+1 < len(result_mat)-1:
    #                 sum_vals+=result_mat[r+1][c]
    #                 counts += 1
    #             if c-1 > 0:
    #                 sum_vals+=result_mat[r+1][c-1]
    #                 counts += 1
    #         if r-1 > 0:
    #             sum_vals+=result_mat[r-1][c]
    #             counts += 1
    #             if c+1 < len(result_mat)-1:
    #                 sum_vals+=result_mat[r-1][c]
    #                 counts += 1
    #             if c-1 > 0:
    #                 sum_vals+=result_mat[r-1][c-1]
    #                 counts += 1
    #         if c+1 < len(result_mat)-1:
    #             print c+1
    #             sum_vals+=result_mat[r][c+1]
    #             counts += 1
    #         if c-1 > 0:
    #             sum_vals+=result_mat[r][c-1]
    #             counts += 1
    #         result_mat2[r][c] = sum_vals/counts


    # for r in range(len(result_mat)):
    #     for c in range(len(result_mat[r])):
    #         for guas in guassian_kernel:
    #             result_mat[r][c] += guas[r][c]

    # grid_z2 = griddata((alpha_beta_shutter_pareto_noises['complexity'].unique(), alpha_beta_shutter_pareto_noises['noise'].unique()), result_mat, (grid_x, grid_y), method='cubic')
    # plt.subplot(2,1,1)
    # plt.figure(figsize=(4,4))
    # ax = sns.heatmap(result, annot=False, fmt="g", cmap='coolwarm', square=True)
    # ax.set_xlabel('Complexity', fontsize=16)
    # ax.set_ylabel('Noise', fontsize=16)
    # ax.tick_params(labelsize=10)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=60)
    # # plt.subplot(2,1,2)
    # # sns.heatmap(result_mat2, annot=False, fmt="g", cmap='coolwarm')
    # plt.show()
    # exit()
    # # x, y, z = np.random.rand(3, 100)
    # cmap = sns.cubehelix_palette(as_cmap=True)

    #
    # f, ax = plt.subplots()
    # points = ax.scatter(alpha_beta_shutter_pareto_noises['complexity'].values, alpha_beta_shutter_pareto_noises['noise'].values, c=alpha_beta_shutter_pareto_noises['tradeoff'].values, s=100, cmap='coolwarm')
    # f.colorbar(points)
    # ax.set_xscale('log')
    # plt.show()
    # exit()

    #
    # max_vals = alpha_beta_shutter['max_moves'].unique()
    # k_vals = alpha_beta_shutter['shutter_size'].unique()
    # noise_vals = alpha_beta_shutter['noise_sig'].unique()
    # boards = alpha_beta_shutter['board'].unique()
    #
    #
    #
    #
    # # alpha_beta_shutter_0 = alpha_beta_shutter.loc[(alpha_beta_shutter['shutter_size'] == 0) & (alpha_beta_shutter['board'] == '6_hard_full')]
    # # alpha_beta_shutter_1 = alpha_beta_shutter.loc[(alpha_beta_shutter['shutter_size'] == 1) & (alpha_beta_shutter['board'] == '6_hard_full')]
    # # alpha_beta_shutter_2 = alpha_beta_shutter.loc[(alpha_beta_shutter['shutter_size'] == 2) & (alpha_beta_shutter['board'] == '6_hard_full')]
    # # print bootstrap_t_pvalue(alpha_beta_shutter_0['correct'].values, alpha_beta_shutter_1['correct'].values)
    # # print bootstrap_t_pvalue(alpha_beta_shutter_0['correct'].values, alpha_beta_shutter_2['correct'].values)
    # # print bootstrap_t_pvalue(alpha_beta_shutter_1['correct'].values, alpha_beta_shutter_2['correct'].values)
    # for m in max_vals:
    #     for noise in noise_vals:
    #         alpha_beta_shutter_filtered = alpha_beta_shutter.loc[(alpha_beta_shutter['max_moves'] == m) & (alpha_beta_shutter['noise_sig'] == noise)]
    #         ax = sns.barplot(x = 'board', y = 'o_misses', hue="shutter_size",   n_boot=1000, data=alpha_beta_shutter_filtered)
    #         plt.title('alpha_beta shutter - correctness: noise =' + str(noise) + '; max moves = '+str(m))
    #         plt.show()
    #     # for board in boards:
    #     #     for k in k_vals:
    #     #         print 'board: ' + board + '; max moves=' + str(m) + '; shutter=' + str(k)
    #     #         # print board
    #     #         alpha_beta_shutter_k = alpha_beta_shutter.loc[(alpha_beta_shutter['max_moves'] == m) & (alpha_beta_shutter['shutter_size'] == k) & (alpha_beta_shutter['board'] == board)]
    #     #         alpha_beta_shutter_k_shutter = alpha_beta_shutter_k.loc[alpha_beta_shutter_k['heuristic_name'] == 'shutter']
    #     #         alpha_beta_shutter_k_heuristic = alpha_beta_shutter_k.loc[alpha_beta_shutter_k['heuristic_name'] == 'heuristic']
    #     #         print 'correctness shutter: ' + str(bs.bootstrap(alpha_beta_shutter_k_shutter['correct'].values, stat_func=bs_stats.mean, is_pivotal=False))
    #     #         # print 'correctness heuristic: ' + str(bs.bootstrap(alpha_beta_shutter_k_heuristic['correct'].values, stat_func=bs_stats.mean, is_pivotal=False))
    #     #         print 'search size shutter: ' + str(bs.bootstrap(alpha_beta_shutter_k_shutter['numberOfNodes'].values, stat_func=bs_stats.mean, is_pivotal=False))
    #     #         # print 'search size heuristic: ' + str(bs.bootstrap(alpha_beta_shutter_k_heuristic['numberOfNodes'].values, stat_func=bs_stats.mean, is_pivotal=False))
    # print 1/0
    # ------- shutter correlation blindness
    # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered030818.csv')
    # # players = players_heuristics['userid'].unique()
    # players = cogsci_participants
    # players_data = dynamics.loc[(dynamics['userid'].isin(players))]
    # # moves = players_data.loc[(players_data['action'] == 'click')]
    # moves = pd.read_csv('stats/test_moves.csv')
    # search_sizes = moves.groupby(['userid'], as_index=False)['action'].count()
    # correct = moves.groupby(['userid'], as_index=False)['correct'].first()
    # moves = moves.loc[moves['open_path'] == True]
    # # moves.to_csv('stats/test_moves.csv')
    # shutter_means = moves.groupby(['userid'], as_index=False)['shutter'].mean()
    #
    # shutter_medians = moves.groupby(['userid'], as_index=False)['shutter'].median()
    # # o_blindness_df = pd.read_csv('stats/blindness_metrics_players.csv')  # various noisy metrics
    # o_blindness_df = pd.read_csv('stats/o_blindness_misses.csv')  # missed wins metric
    #
    # o_blindness_df = o_blindness_df.loc[o_blindness_df['player'] == 2]
    #
    # o_blindness_df['player'] = o_blindness_df['player'].apply(lambda x: 'X' if x==1 else 'O')
    # shutter_blindness_df = pd.merge(o_blindness_df, shutter_means[['userid','shutter']], on = 'userid', how = 'left')
    # shutter_blindness_df = shutter_blindness_df.rename(columns={'shutter': 'shutter_mean'})
    # shutter_blindness_df = pd.merge(shutter_blindness_df, shutter_medians[['userid','shutter']], on = 'userid', how = 'left')
    # shutter_blindness_df = pd.merge(shutter_blindness_df, search_sizes[['userid','action']], on = 'userid', how = 'left')
    # shutter_blindness_df = pd.merge(shutter_blindness_df, correct[['userid','correct']], on = 'userid', how = 'left')
    # shutter_blindness_df = pd.merge(shutter_blindness_df, players_heuristics[['userid','fitted_heuristic']], on = 'userid', how = 'left')
    # # shutter_blindness_df = pd.merge(o_blindness_df, shutter_means[['userid','shutter']], on = 'userid', how = 'left')
    # # shutter_blindness_df = shutter_blindness_df.rename(columns={'shutter': 'shutter_mean'})
    # # shutter_blindness_df = pd.merge(shutter_blindness_df, shutter_medians[['userid','shutter']], on = 'userid', how = 'left')
    # # shutter_blindness_df = pd.merge(shutter_blindness_df, search_sizes[['userid','action']], on = 'userid', how = 'left')
    # shutter_blindness_df = shutter_blindness_df.rename(columns={'shutter': 'shutter_median'})
    # shutter_blindness_df = shutter_blindness_df.rename(columns={'action': 'search_size'})
    #
    #
    # # shutter_blindness_df = shutter_blindness_df.loc[shutter_blindness_df['fitted_heuristic'] == 'blocking']
    # shutter_blindness_correct = shutter_blindness_df.loc[shutter_blindness_df['correct'] == 1]
    # shutter_blindness_wrong = shutter_blindness_df.loc[shutter_blindness_df['correct'] == 0]
    # shutter_blindness_df['shutter_mean'] = shutter_blindness_df['shutter_mean'].apply(lambda x: x-1.0)
    # shutter_blindness_df['mean_shutter_cat'] = pd.qcut(shutter_blindness_df['shutter_mean'], 3, labels=['narrow\n(0-0.2)','medium\n(0.2-0.5)','wide\n(0.5-2)'])
    # print shutter_blindness_df.shape[0]
    # cats = shutter_blindness_df['mean_shutter_cat'].unique()
    # for i in range(len(cats)):
    #     s = cats[i]
    #     print s
    #     blindness_cat = shutter_blindness_df.loc[shutter_blindness_df['mean_shutter_cat'] == s]
    #     print blindness_cat.shape[0]
    #     print bs.bootstrap(blindness_cat['missed_win'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #     for j in range(i,len(cats)):
    #         s2 = cats[j]
    #         blindness_cat2 = shutter_blindness_df.loc[shutter_blindness_df['mean_shutter_cat'] == s2]
    #         if s != s2:
    #             print s2
    #             # print bootstrap_t_pvalue(blindness_cat['missed_win'].values, blindness_cat2['missed_win'].values)
    #             print stats.mannwhitneyu(blindness_cat['missed_win'].values, blindness_cat2['missed_win'].values)
    #             print rank_biserial_effect_size(blindness_cat['missed_win'].values, blindness_cat2['missed_win'].values)
    #         print s
    #
    #
    #
    # # #
    # # # # print stats.pearsonr(shutter_blindness_df['mean_shutter_cat'].values, shutter_blindness_df['missed_win'].values)
    # # # # figure 4 - shutter and correctness pnas
    # shutter_blindness_df['missed_win_percent'] = shutter_blindness_df['missed_win'].apply(lambda x: x*100)
    # f, axs = plt.subplots(1,3,figsize=(8.6,3), gridspec_kw = {'width_ratios':[2, 1, 2]})
    # plt.subplot(1,3,3)
    # ax = sns.barplot(x='mean_shutter_cat',y='missed_win_percent', ci=68, data=shutter_blindness_df)
    # # ax.set(xlabel='Shutter', ylabel='Missed O Wins')
    # ax.set_xlabel('Shutter size', fontsize=12)
    # ax.set_ylabel("Prob. missed 'O' wins [%]", fontsize=12)
    # ax.tick_params(labelsize=11)
    # plt.ylim(0,100)
    # # plt.show()
    # # ax = sns.regplot(x='shutter_mean', y='missed_win', data=shutter_blindness_df, x_jitter=.1)
    # # plt.show()
    # print stats.pearsonr(shutter_blindness_correct['shutter_mean'].values, shutter_blindness_correct['missed_win'].values)
    # print stats.pearsonr(shutter_blindness_wrong['shutter_mean'].values, shutter_blindness_wrong['missed_win'].values)
    # # print stats.pearsonr(shutter_blindness_df['o_prob_ratio_mean'].values, shutter_blindness_df['shutter_mean'].values)
    # # print stats.pearsonr(shutter_blindness_df['o_rank_mean'].values, shutter_blindness_df['shutter_mean'].values)
    # # print stats.pearsonr(shutter_blindness_correct['o_rank_mean'].values, shutter_blindness_correct['missed_win'].values)
    #
    # # print stats.pearsonr(shutter_blindness_df['search_size'].values, shutter_blindness_df['o_prob_mean'].values)
    # # print stats.pearsonr(shutter_blindness_correct['search_size'].values, shutter_blindness_correct['o_prob_mean'].values)
    # # print stats.pearsonr(shutter_blindness_wrong['search_size'].values, shutter_blindness_wrong['o_prob_mean'].values)
    # # # print stats.pearsonr(shutter_blindness_df['o_prob_ratio_mean'].values, shutter_blindness_df['shutter_mean'].values)
    # # # print stats.pearsonr(shutter_blindness_df['o_rank_mean'].values, shutter_blindness_df['shutter_mean'].values)
    # # print stats.pearsonr(shutter_blindness_correct['o_rank_mean'].values, shutter_blindness_correct['o_prob_mean'].values)
    # # print 1/0
    # ------- shutter correlation blindness end

    # --- shutter search size correlation (pnas)----
    # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered030818.csv')
    # players = players_heuristics['userid'].unique()
    # # players_data = dynamics.loc[(dynamics['userid'].isin(cogsci_participants))]
    # players_data = dynamics
    # moves = players_data.loc[(players_data['action'] == 'click')]
    # # moves = pd.read_csv('stats/test_moves.csv')
    # search_sizes = moves.groupby(['userid'], as_index=False)['action'].count()
    #
    # # correct = moves.groupby(['userid'], as_index=False)['correct'].first()
    # moves = moves.loc[moves['open_path'] == True]
    # # moves.to_csv('stats/test_moves.csv')
    # shutter_means = moves.groupby(['userid'], as_index=False)['shutter'].mean()
    # shutter_means_players = shutter_means['userid'].unique()
    # search_sizes = search_sizes.loc[(search_sizes['userid'].isin(shutter_means_players))]
    # # shutter_means = shutter_means.loc[shutter_means['shutter'] is not np.nan]
    #
    # shutter_medians = moves.groupby(['userid'], as_index=False)['shutter'].median()
    #
    # shutter_search_df = pd.merge(search_sizes, shutter_means, on='userid', how='left')
    # shutter_search_df = shutter_search_df.loc[shutter_search_df['shutter']<=1.5]
    # shutter_search_df['shutter'] = shutter_search_df['shutter'].apply(lambda x: x+ np.random.normal(0, 0.02))
    # shutter_search_df['action'] = shutter_search_df['action'].apply(lambda x: x+ np.random.normal(0, 0.02))
    # # for idx, row in shutter_search_df.iterrows():
    # #     row['shutter'] += np.random.normal(0, 0.1)
    # #     row['action'] += np.random.normal(0, 0.1)
    # # np.random.normal(0, 0.1)
    # shutter_search_df1 = shutter_search_df.loc[shutter_search_df['shutter']==1.0]
    # print shutter_search_df1.shape[0]
    #
    # # figure 4b --- moved to SI
    # ax = sns.regplot(x='shutter', y='action', data=shutter_search_df)
    # plt.show()
    # print stats.pearsonr(shutter_search_df['action'].values, shutter_search_df['shutter'].values)
    # print 1/0
    # --- shutter search size correlation (pnas) end----

    # ----------o blindness - non-forced moves quality---------------
    # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered030818.csv')
    # # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered_allData_070818.csv')
    #
    # heuristics = ['density','linear','non-linear','interaction', 'blocking']
    # players = players_heuristics['userid'].unique()
    # userids = []
    # boards = []
    # correct = []
    # heuristic_players = []
    # o_blind_prob_mean = []
    # o_blind_prob_best_mean = []
    # o_blind_rank_mean = []
    # o_blind_prob_median = []
    # o_blind_rank_median = []
    # o_blind_prob_best_median = []
    # x_blind_prob_mean = []
    # x_blind_rank_mean = []
    # x_blind_prob_best_mean = []
    # x_blind_prob_median = []
    # x_blind_rank_median = []
    # x_blind_prob_best_median = []
    # for player in players:
    #     player_heuristic = players_heuristics.loc[players_heuristics['userid'] == player]
    #     player_moves = move_probabilities_heuristics.loc[move_probabilities_heuristics['userid'] == player]
    #     fitted_heuristic_player = player_heuristic['fitted_heuristic'].iloc[0]
    #     if fitted_heuristic_player is not None:
    #         move_probs_player = player_moves.loc[player_moves['heuristic'] == fitted_heuristic_player]
    #         o_blind_metrics = compute_blindness_player(move_probs_player, player=2, include_forced=False)
    #         x_blind_metrics = compute_blindness_player(move_probs_player, player=1, include_forced=False)
    #         o_blind_prob_mean.append(o_blind_metrics[0])
    #         o_blind_prob_best_mean.append(o_blind_metrics[4])
    #         o_blind_prob_best_median.append(o_blind_metrics[5])
    #         o_blind_rank_mean.append(o_blind_metrics[2])
    #         o_blind_prob_median.append(o_blind_metrics[1])
    #         o_blind_rank_median.append(o_blind_metrics[3])
    #         x_blind_prob_mean.append(x_blind_metrics[0])
    #         x_blind_rank_mean.append(x_blind_metrics[2])
    #         x_blind_prob_median.append(x_blind_metrics[1])
    #         x_blind_rank_median.append(x_blind_metrics[3])
    #         x_blind_prob_best_mean.append(x_blind_metrics[4])
    #         x_blind_prob_best_median.append(x_blind_metrics[5])
    #         userids.append(player)
    #         player_dynamics = dynamics.loc[dynamics['userid'] == player]
    #         boards.append(player_dynamics['board_name'].iloc[0])
    #         correct.append(player_dynamics['correct'].iloc[0])
    #         heuristic_players.append(fitted_heuristic_player)
    #
    # blindness_metrics = {'userid': userids, 'board': boards, 'fitted_heuristic': heuristic_players,'correct': correct, 'o_prob_mean': o_blind_prob_mean,
    #                      'o_prob_median': o_blind_prob_median, 'o_prob_ratio_mean': o_blind_prob_best_mean, 'o_prob_ratio_median': o_blind_prob_best_median,
    #                      'o_rank_mean': o_blind_rank_mean, 'o_rank_median': o_blind_rank_median,
    #                      'x_prob_mean': x_blind_prob_mean, 'x_prob_median': x_blind_prob_median,
    #                      'x_prob_ratio_mean': x_blind_prob_best_mean, 'x_prob_ratio_median': x_blind_prob_best_median,
    #                      'x_rank_mean': x_blind_rank_mean, 'x_rank_median': x_blind_rank_median}
    # blindness_df = pd.DataFrame(blindness_metrics)
    # blindness_df.to_csv('stats/blindness_metrics_players_all.csv')
    # print blindness_df.shape[0]
    # print 1/0
    # ----------o blindness - non-forced moves quality end---------------


    # ----------o blindness and correctness---------------
    # # compare blocking with blocking-blind, interaction with interaction-blind
    # # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered_allData_070818.csv')
    # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered030818.csv')
    # players = players_heuristics['userid'].unique()
    # players_data = dynamics.loc[(dynamics['userid'].isin(players))]
    # moves = players_data.loc[(players_data['action'] == 'click')]
    # misses = []
    # players = []
    # users = []
    # boards = []
    # distances = []
    # correct = []
    # open = []
    # heuristics_fitted = []
    # for index, row in moves.iterrows():
    #     if row['top_possible_score'] == 100:
    #         if row['score_move'] < 100:
    #             misses.append(1)
    #         else:
    #             misses.append(0)
    #         players.append(row['player'])
    #         users.append(row['userid'])
    #         boards.append(row['board_name'])
    #         distances.append(row['shutter'])
    #         open.append(row['open_path'])
    #         correct.append(row['correct'])
    #         player_heuristic = players_heuristics.loc[players_heuristics['userid'] == row['userid']]
    #         heuristics_fitted.append(player_heuristic['fitted_heuristic'].iloc[0])
    #
    # misses_data = {'userid': users,'board': boards, 'player':players, 'missed_win':misses, 'shutter':distances, 'open_path':open, 'correct': correct, 'fitted_heuristic': heuristics_fitted}
    # # misses_data = {'userid': users,'board': boards, 'player':players, 'missed_win':misses, 'correct': correct, 'fitted_heuristic': heuristics_fitted}
    # misses_data = pd.DataFrame(misses_data)
    # # misses_data.to_csv('stats/misses_data_all.csv')
    # print misses_data.shape[0]
    # # compute mean missed wins per player
    # missed_wins_averages = misses_data.groupby(['userid','player'], as_index=False)['missed_win'].mean()
    # missed_wins_averages.to_csv('stats/o_blindness_misses.csv')
    # misses_data = misses_data[['userid','board','shutter','fitted_heuristic','correct']]
    # # misses_data = misses_data[['userid','board','fitted_heuristic','correct']]
    # misses_data = misses_data.drop_duplicates(subset='userid', keep='first', inplace=False)
    # misses_data = pd.merge(missed_wins_averages, misses_data[['userid','board','shutter','fitted_heuristic','correct']], on = 'userid', how = 'left')
    # # misses_data = pd.merge(missed_wins_averages, misses_data[['userid','board','fitted_heuristic','correct']], on = 'userid', how = 'left')
    #
    # # misses_data.to_csv('stats/misses_data_averages_all.csv')
    # print misses_data.shape[0]
    # misses_data = misses_data.loc[misses_data['fitted_heuristic'] == 'blocking']
    # # print missed_wins_averages['missed_win']
    # # for board in LOGFILE:
    # #     print '-----------------'
    # #     print board
    # #     misses_data_board = misses_data.loc[misses_data['board'] == board]
    # missed_wins_x_correct = misses_data.loc[(misses_data['player'] == 1)  & (misses_data['correct'] == 1)]
    # missed_wins_x_wrong = misses_data.loc[(misses_data['player'] == 1) & (misses_data['correct'] == 0)]
    # missed_wins_o_correct = misses_data.loc[(misses_data['player'] == 2)  & (misses_data['correct'] == 1)]
    # missed_wins_o_wrong = misses_data.loc[(misses_data['player'] == 2) & (misses_data['correct'] == 0)]
    # missed_wins_x = misses_data.loc[(misses_data['player'] == 1) ]
    # missed_wins_o = misses_data.loc[(misses_data['player'] == 2)]
    #
    # print bs.bootstrap(missed_wins_x['missed_win'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(missed_wins_o['missed_win'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print '---'
    # print missed_wins_x_correct.shape[0]
    # if missed_wins_x_correct.shape[0] > 0:
    #     print bs.bootstrap(missed_wins_x_correct['missed_win'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print missed_wins_o_correct.shape[0]
    # if missed_wins_o_correct.shape[0] > 0:
    #     print bs.bootstrap(missed_wins_o_correct['missed_win'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print missed_wins_x_wrong.shape[0]
    # if missed_wins_x_wrong.shape[0] > 0:
    #     print bs.bootstrap(missed_wins_x_wrong['missed_win'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print missed_wins_o_wrong.shape[0]
    # if missed_wins_o_wrong.shape[0] > 0:
    #     print bs.bootstrap(missed_wins_o_wrong['missed_win'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print bootstrap_t_pvalue(missed_wins_o_correct['missed_win'].values, missed_wins_o_wrong['missed_win'].values)
    # print 'mann whitney'
    # print stats.mannwhitneyu(missed_wins_o_correct['missed_win'].values, missed_wins_o_wrong['missed_win'].values)
    # # plt.hist(missed_wins_x['missed_win'])
    # # print np.std(missed_wins_x['missed_win'].values)
    # # plt.show()
    # # plt.hist(missed_wins_o['missed_win'])
    # # print np.std(missed_wins_o['missed_win'].values)
    # # plt.show()
    # print 1/0
    # # ----------o blindness and correctness end---------------
    #
    # # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered_withBlind_030818.csv')
    # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered_allData_070818.csv')
    # # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered030818.csv')
    # players = players_heuristics['userid'].unique()
    # players_data = dynamics.loc[(dynamics['userid'].isin(players))]
    # correctness_players = players_data[['userid','correct', 'board_name']]
    # correctness_players = correctness_players.drop_duplicates(subset='userid', keep='first', inplace=False)
    # #
    # # heuristics_users.set_index(['userid'], inplace=True)
    # # dynamics.set_index(['userid'], inplace=True)
    #
    # # dynamics.join(heuristics_users).reset_index()
    # data = pd.merge(players_heuristics, correctness_players, on = 'userid', how = 'left')
    # # data = data.loc[data['board_name'] == '10_medium_full']
    # heuristics = ['density','linear','non-linear','interaction', 'blocking']
    # # heuristics = ['interaction', 'blocking', 'interaction_blind', 'blocking_blind']
    # # heuristics = ['blocking', 'blocking_blind']
    # for heuristic in heuristics:
    #     print heuristic
    #     heuristic_data = data.loc[data['fitted_heuristic'] == heuristic]
    #     print bs.bootstrap(heuristic_data['correct'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # data = data.sample(frac=1).drop_duplicates(['userid'])
    # # data.to_csv('stats/filtered_users_heuristic_correct.csv')
    # heuristic1 = 'blocking'
    # heuristic2 = 'interaction'
    # boards = ['6_easy_full','6_easy_pruned', '10_easy_full', '10_easy_pruned','6_hard_full','6_hard_pruned', '10_hard_full', '10_hard_pruned',  '10_medium_full', '10_medium_pruned']
    # # for board in boards:
    # #     blocking_data = data.loc[(data['fitted_heuristic'] == 'blocking') & (data['board_name'] == board)]
    # #     blocking_blind_data = data.loc[(data['fitted_heuristic'] == 'blocking_blind') & (data['board_name'] == board)]
    # #     print board
    # #     print blocking_data.shape[0]
    # #     print blocking_blind_data.shape[0]
    # data1 = data.loc[data['fitted_heuristic'] == heuristic1]
    # data2 = data.loc[data['fitted_heuristic'] == heuristic2]
    # print bootstrap_t_pvalue(data1['correct'].values, data2['correct'].values)
    # print data.shape[0]
    #
    # print 1/0

    # ------- heuristic and search size
    # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered_allData_070818.csv')
    # # players_heuristics = pd.read_csv('stats/fitted_heuristics_notFiltered_060818.csv')
    #
    # # players_heuristics = players_heuristics.loc[players_heuristics['log_likelihood'] > -2]
    # players = players_heuristics['userid'].unique()
    # heuristics = players_heuristics['fitted_heuristic'].unique()
    # # heuristics = ['interaction', 'blocking']
    # search_size_players = []
    # heuristics_players_list = []
    # heuristics_log_list = []
    # userids = []
    # for player in players:
    #     player_heuristic = players_heuristics.loc[players_heuristics['userid'] == player]
    #     fitted_heuristic_player = player_heuristic['fitted_heuristic'].iloc[0]
    #     log_likelihood_player = player_heuristic['log_likelihood'].iloc[0]
    #     player_actions = dynamics.loc[(dynamics['userid'] == player) & (dynamics['action'] == 'click')]
    #     # if player_actions['correct'].iloc[0] != 1:
    #     #     continue
    #     userids.append(player)
    #     search_size_players.append(player_actions.shape[0])
    #     heuristics_players_list.append(fitted_heuristic_player)
    #     heuristics_log_list.append(log_likelihood_player)
    #
    # search_size_dict = {'userid': userids, 'heuristic': heuristics_players_list, 'log_likelihood': heuristics_log_list, 'search_size': search_size_players}
    # search_size_df = pd.DataFrame(search_size_dict)
    # for heuristic in heuristics:
    #     print heuristic
    #
    #     data_heuristic = search_size_df.loc[search_size_df['heuristic'] == heuristic]
    #     # if heuristic == 'blocking':
    #     #     print data_heuristic['search_size']
    #     print 'mean: ' + str(data_heuristic['search_size'].mean()) +';' + str(bs.bootstrap(data_heuristic['search_size'].values, stat_func=bs_stats.mean, is_pivotal=False))
    #     print 'median: ' + str(data_heuristic['search_size'].median()) + str(medianCI(data_heuristic['search_size'].values, 0.95, 0.5))
    #     print 'correlation: ' + str(stats.pearsonr(data_heuristic['search_size'].values, data_heuristic['log_likelihood'].values))
    # simple_heuristics = ['density', 'linear']
    # sophisticated_heuristics = ['interaction', 'blocking']
    # data_simple_heuristics = search_size_df.loc[search_size_df['heuristic'].isin(simple_heuristics)]
    # data_sophisticated_heuristics = search_size_df.loc[search_size_df['heuristic'].isin(sophisticated_heuristics)]
    # print bs.bootstrap(data_simple_heuristics['search_size'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(data_sophisticated_heuristics['search_size'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bootstrap_t_pvalue(data_simple_heuristics['search_size'].values, data_sophisticated_heuristics['search_size'].values)
    #
    # print 1/0
    # ------- shutter correlation correctness
    # players_heuristics = pd.read_csv('stats/fitted_heuristics_filtered030818.csv')
    # # players_heuristics_filtered = players_heuristics.loc[players_heuristics['fitted_heuristic'] == 'blocking_blind']
    # players_heuristics_filtered = players_heuristics
    # # players = players_heuristics_filtered['userid'].unique()
    # players = cogsci_participants['userid'].unique()
    # moves_on_path = dynamics.loc[(dynamics['open_path'] == True) & (dynamics['userid'].isin(players))]
    # moves_on_path = moves_on_path.loc[moves_on_path['action'] == 'click']
    # players = moves_on_path['userid'].unique()
    # # players = moves_on_path['userid'].unique()
    # print 'num=' + str(len(players))
    # mean_shutter_players = []
    # median_shutter_players = []
    # correctness = []
    # userids = []
    # # heuristic_log_likelihoods = []
    # # heuristics_fitted = []
    # for player in players:
    #     # print player
    #     player_data = moves_on_path.loc[(moves_on_path['userid'] == player)]
    #     player_heuristic_data = players_heuristics_filtered.loc[players_heuristics_filtered['userid'] == player]
    #     userids.append(player)
    #     mean_shutter_players.append(player_data['shutter'].mean())
    #     median_shutter_players.append(player_data['shutter'].median())
    #     # player_corr = moves_on_path.loc[(moves_on_path['userid'] == player)]
    #     # heuristic_log_likelihoods.append(player_heuristic_data['log_likelihood'].unique()[0])
    #     # heuristics_fitted.append(player_heuristic_data['fitted_heuristic'].unique()[0])
    #     correctness.append(player_data['correct'].unique()[0])
    #
    # shutter_correctness_dict = {'userid': userids, 'mean_shutter': mean_shutter_players, 'median_shutter': median_shutter_players, 'correct': correctness}
    # # shutter_correctness_dict = {'userid': userids, 'mean_shutter': mean_shutter_players, 'median_shutter': median_shutter_players, 'correct': correctness,'fitted_heuristic': heuristics_fitted, 'log_likelihood':heuristic_log_likelihoods}
    #
    # shutter_correctness_df = pd.DataFrame(shutter_correctness_dict)
    # correct_players = shutter_correctness_df.loc[shutter_correctness_df['correct'] == 1]
    # wrong_players = shutter_correctness_df.loc[shutter_correctness_df['correct'] == 0]
    #
    # shutter_correctness_df['mean_shutter'] = shutter_correctness_df['mean_shutter'].apply(lambda x: x-1.0)
    # #
    # # # ---code for figure 4a--
    # # bins = pd.IntervalIndex.from_tuples([(0, 0.2), (0.2, 0.5), (0.5, 1.9)])
    # # pd.Inter
    # shutter_correctness_df['mean_shutter_cat'] = pd.cut(shutter_correctness_df['mean_shutter'], bins=[0,0.2,0.5,2], right=True, include_lowest=True, labels=['narrow\n(0-0.2)','medium\n(0.2-0.5)','wide\n(0.5-2)'])
    # # shutter_correctness_df['mean_shutter_cat'] = pd.qcut(shutter_correctness_df['mean_shutter'], 3)
    # print shutter_correctness_df.shape[0]
    # cats = shutter_correctness_df['mean_shutter_cat'].unique()
    # print cats
    # # cats = cats[:3]
    # for i in range(len(cats)):
    #     # print i
    #     # print 's =' + str(s)
    #     print '-------------------'
    #     s = cats[i]
    #     print s
    #     blindness_cat = shutter_correctness_df.loc[shutter_correctness_df['mean_shutter_cat'] == s]
    #     print blindness_cat.shape[0]
    #     print bs.bootstrap(blindness_cat['correct'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #     for j in range(i+1,len(cats)):
    #         s2 = cats[j]
    #         print s2
    #         blindness_cat2 = shutter_correctness_df.loc[shutter_correctness_df['mean_shutter_cat'] == s2]
    #         # print blindness_cat2.shape[0]
    #
    #         print stats.mannwhitneyu(blindness_cat['correct'].values, blindness_cat2['correct'].values)
    #         print rank_biserial_effect_size(blindness_cat['correct'].values, blindness_cat2['correct'].values)
    #             # print stats.chi2_contingency()
    #         # print s
    #
    # # shutter_correctness_blindness_df = pd.merge(shutter_blindness_df, shutter_correctness_df, on='userid', how='left')
    # # shutter_correctness_df['correct'] = shutter_correctness_df['correct'].map({1: 'correct', 0: 'wrong'})
    # shutter_correctness_df['correct_percent'] = shutter_correctness_df['correct'].apply(lambda x: x*100)
    # plt.subplot(1,3,1)
    # ax = sns.barplot(x = 'mean_shutter_cat', y = 'correct_percent', n_boot=1000, data=shutter_correctness_df, ci=68)
    # # ax.set(xlabel='Shutter', ylabel='Correct Participants')
    # ax.set_xlabel('Shutter size', fontsize=12)
    # ax.set_ylabel('Prob. winning move [%]', fontsize=12)
    # ax.tick_params(labelsize=11)
    # # plt.ylim(0,100)
    # # # ax = sns.barplot(x = 'fitted_heuristic', y = 'heuristic_correct', hue="correct",   n_boot=1000, data = heuristics_sensitivity_filtered,  estimator=weighted_mean, orient="v",order=['density','linear','non-linear', 'interaction_blind', 'interaction','blocking_blind','blocking'])
    # # ax = sns.barplot(x = 'correct', y = 'mean_shutter', n_boot=1000, data=shutter_correctness_df, ci=68)
    # # change_width(ax, .35)
    # # for bar in ax.patches:
    # #     x = bar.get_x()
    # #     # x = bar[0]._x
    # #     width = bar.get_width()
    # #     # width = bar[0]._linewidth
    # #     newwidth = width/2.
    # #     centre = x+width/2.
    # #     # bar[0]._x = centre-newwidth/2.
    # #     # bar[0]._linewidth = newwidth
    # #     bar.set_x(centre-newwidth/2.)
    # #     bar.set_width(newwidth)
    # # plt.tight_layout(pad=2.5)
    # # plt.show()
    # # exit()
    # # #
    # # # # #
    # # # # print correct_players['mean_shutter'].mean()
    # # # # print correct_players.shape[0]
    # # # # print wrong_players['mean_shutter'].mean()
    # # # # print wrong_players.shape[0]
    # # # # print bootstrap_t_pvalue(correct_players['mean_shutter'].values, wrong_players['mean_shutter'].values)
    # # # # print bs.bootstrap(correct_players['mean_shutter'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # # # print bs.bootstrap(wrong_players['mean_shutter'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # # #
    # # # # print bs.bootstrap(correct_players['log_likelihood'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # # # print bs.bootstrap(wrong_players['log_likelihood'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # #
    # # print 1/0
    # ------- shutter correlation correctness end


    # ----- mcts distance people
    # boards = ['6_easy_full','6_easy_pruned', '10_easy_full', '10_easy_pruned','6_hard_full','6_hard_pruned', '10_hard_full', '10_hard_pruned',  '10_medium_full', '10_medium_pruned']
    # base_dir = 'data_matrices/cogsci/'
    # people_all = read_matrices_from_file(base_dir+'avg_people_first_moves_all.json')
    # paths_mcts = pd.read_csv('stats/mcts170918_paths_k3_stochastic_50nodes.csv')
    # first_moves_mcts = paths_mcts.loc[paths_mcts['length'] == 1]
    # sum_dists = 0.0
    # for board in boards:
    #     participants_dist = people_all[board]
    #     first_moves_mcts_board = first_moves_mcts.loc[paths_mcts['board'] == board]
    #     total_moves_mcts = first_moves_mcts_board['count'].sum()
    #     mcts_matrix = copy.deepcopy(participants_dist)
    #     for index, row in first_moves_mcts_board.iterrows():
    #         a = np.array(ast.literal_eval(row['path']))
    #         r = a[0][0]
    #         c = a[0][1]
    #         mcts_matrix[r][c] = float(row['count'])/total_moves_mcts
    #
    #     for row in range(len(participants_dist)):
    #         for col in range(len(participants_dist)):
    #             if (participants_dist[row][col] == -0.00001) | (participants_dist[row][col] == -0.00002):
    #                 participants_dist[row][col] = 0
    #                 mcts_matrix[row][col] = 0
    #             # else:
    #             #     mcts_matrix =
    #     print participants_dist
    #     print mcts_matrix
    #     print emd(participants_dist, mcts_matrix)
    #     sum_dists += emd(participants_dist, mcts_matrix)
    # print 'avg'
    # print sum_dists/len(boards)
    # exit()

    # ------ mcts path entropies
    mcts6 = pd.read_csv('stats/testCorrect.csv')
    print bs.bootstrap(mcts6['correct'].values,stat_func=bs_stats.mean, is_pivotal=False)
    exit()

    paths_mcts = pd.read_csv('stats/mcts170918_paths_k3_stochastic_1000nodes_test3.csv')
    # boards_raw = ['6_easy', '10_easy','6_hard', '10_hard', 'medium']
    board_states = paths_mcts['board_state'].unique()
    entropies_pruned = []
    entropies_full = []
    states_data = []
    boards = []
    solved = []
    conditions = []
    # heuristics = []
    entropies = []
    for s in board_states:
        moves_s = paths_mcts.loc[(paths_mcts['board_state'] == s)]
        if (len(moves_s['path'].unique()) > 1):
            check = True
            for p in moves_s['path'].unique():
                m_p = moves_s.loc[moves_s['path'] == p]
                if m_p.shape[0] < 5:
                    check = False
                    break
                # moves_pruned = m_p.loc[m_p['condition'] == 'pruned']
                # moves_full = m_p.loc[m_p['condition'] == 'full']
                # if (moves_full.shape[0] == 0) | (moves_pruned.shape[0] == 0):
                #     check = False
                #     break;
            if check:
                # vals = moves_s['position'].unique()
                #
                # g = sns.FacetGrid(moves_s, col="condition", legend_out=False)
                # g.map(sns.countplot, "position", order= vals, color="steelblue", lw=0)
                # print moves_s['board_name'].unique()
                # # plt.title("tt")
                # plt.show()

                pk = []
                moves_pruned = moves_s.loc[moves_s['condition'] == 'pruned']
                mp = moves_pruned['position'].unique()
                total = moves_pruned.shape[0] + 0.0
                for m in mp:
                    count = moves_pruned[moves_pruned['position'] == m].shape[0]
                    pk.append(float(count)/float(total))
                ent = stats.entropy(pk)
                entropies_pruned.append(ent)
                states_data.append(s)
                if moves_pruned.shape[0] == 0:
                    continue
                boards.append(moves_pruned['board'].unique()[0])
                # solvers = moves_pruned[moves_pruned['solved'] != 'validatedCorrect'].shape[0]
                # solved.append(float(solvers)/total)
                conditions.append('pruned')
                # heuristics.append()
                entropies.append(float(ent))
                pk = []

                moves_full = moves_s.loc[moves_s['condition'] == 'full']
                mf = moves_full['position'].unique()
                total = moves_full.shape[0] + 0.0
                for m in mf:
                    count = moves_full[moves_full['position'] == m].shape[0]
                    pk.append(float(count)/float(total))
                ent = stats.entropy(pk)
                entropies_full.append(ent)
                states_data.append(s)
                # solvers = moves_full[moves_full['solved'] == 'validatedCorrect'].shape[0]
                # solved.append(float(solvers)/total)
                if moves_full.shape[0] == 0:
                    continue
                boards.append(moves_full['board'].unique()[0])
                conditions.append('full')
                entropies.append(float(ent))
    entropies_data = {'board': boards, 'state': states_data, 'entropy':entropies, 'condition':conditions}
    entropies_data = pd.DataFrame(entropies_data)
    entropies_data['condition'] = entropies_data['condition'].map({'full': 'full', 'pruned': 'truncated'})
    print entropies_data['entropy']
    entropies_full = entropies_data.loc[entropies_data['condition'] == 'full']
    entropies_pruned = entropies_data.loc[entropies_data['condition'] == 'truncated']
    ax = sns.barplot(x='condition', y='entropy',  n_boot=1000, ci=68, data=entropies_data)
    print bootstrap_t_pvalue(entropies_full['entropy'].values, entropies_pruned['entropy'].values)
    print bs.bootstrap(entropies_full['entropy'].values, stat_func=bs_stats.mean, is_pivotal=False)
    print bs.bootstrap(entropies_pruned['entropy'].values, stat_func=bs_stats.mean, is_pivotal=False)
    plt.show()
    exit()
    # entropies_data.to_csv('stats/entropies_data_082118.csv')
    # ------ mcts path probabilities
    # path_counts_df = pd.read_csv('stats/mcts170918_paths_k5_stochastic_1000_6.csv')
    # path_total_counts = path_counts_df.groupby(['board','length']).sum().reset_index()
    # path_counts_df['mcts_prob'] = np.nan
    #
    # for index, row in path_counts_df.iterrows():
    #     path_total_counts_path = path_total_counts.loc[(path_total_counts['board'] == row['board']) & (path_total_counts['length'] == row['length'])]
    #     total_path = path_total_counts_path.iloc[0]['count']
    #     # print 'done'
    #     path_counts_df.loc[index,'mcts_prob'] = float(row['count'])/total_path
    #     # row[path_total_counts] = float(row['count'])/total_path
    # path_counts_df.to_csv('stats/path_probs_mcts_k5_stochastic_1000_6.csv')
    # path_counts_mcts_df = pd.read_csv('stats/path_probs_mcts_k5_stochastic_1000_6.csv')
    # path_counts_participants_df = pd.read_csv('stats/path_probs_participants.csv')
    #
    # path_counts_participants_df['mcts_prob'] = np.nan
    # for index, row in path_counts_participants_df.iterrows():
    #     path_mcts = path_counts_mcts_df.loc[(path_counts_mcts_df['board'] == row['board_name']) & (path_counts_mcts_df['path'] == row['path'])]
    #     if path_mcts.shape[0] == 0:
    #         path_counts_participants_df.loc[index,'mcts_prob'] = 0
    #     else:
    #         path_counts_participants_df.loc[index,'mcts_prob'] = path_mcts['mcts_prob'].values[0]
    #
    # path_counts_participants_df.to_csv('stats/path_probs_participants_mcts_k5_stochastic_1000_6.csv')
    #
    # path_probs_participants_mcts = pd.read_csv('stats/path_probs_participants_mcts_k5_stochastic_1000_6.csv')
    # path_probs_participants_mcts = path_probs_participants_mcts.loc[(path_probs_participants_mcts['participants_prob'] >= 0.05) & (path_probs_participants_mcts['move_number_in_path'] < 8) & (path_probs_participants_mcts['board_name'].isin(['6_easy_full','6_easy_pruned','6_hard_full','6_hard_pruned']))]
    # correlations = []
    # move_numbers = []
    # path_lenghts = path_probs_participants_mcts['move_number_in_path'].unique()
    # for i in range(50):
    #     for l in path_lenghts:
    #         # path_probs_participants_mcts_l = path_probs_participants_mcts.loc[path_probs_participants_mcts['move_number_in_path']==l]
    #
    #         df_sample = path_probs_participants_mcts.sample(frac=0.7)
    #         # df_sample_h = path_probs_participants_mcts_l.sample(frac=0.7)
    #         df_sample_h = df_sample.loc[(df_sample['move_number_in_path'] == l)]
    #         move_numbers.append(l)
    #         print df_sample_h
    #         correlations.append(stats.pearsonr(df_sample_h['participants_prob'].values, df_sample_h['mcts_prob'].values)[0])
    #         print stats.pearsonr(df_sample_h['participants_prob'].values, df_sample_h['mcts_prob'].values)[0]
    # correlations_df = pd.DataFrame({'path_length': move_numbers, 'correlation': correlations})
    # correlations_df.to_csv('stats/heuristics_participants_paths_mcts_correlations_pearsonr_k5_stochastic_1000_6.csv')
    # correlations_df = pd.read_csv('stats/heuristics_participants_paths_mcts_correlations_pearsonr_k5_stochastic_1000_6.csv')
    # ax = sns.barplot(x='path_length',y='correlation', ci=68, data=correlations_df)
    # plt.show()
    # exit()
    # ------- path probabilities population vs. heuristic prediction
    # # players = move_probabilities_heuristics['userid'].unique()
    # heuristics = ['density', 'linear', 'non-linear', 'interaction', 'blocking']
    # # moves_df = dynamics.loc[(dynamics['action'] == 'click')]
    # # moves_df = moves_df[['board_name','userid','path_after', 'move_number_in_path']]
    # # # path_counts_df = moves_df.groupby(['board_name','path_after','move_number_in_path']).count()
    # # # path_counts_df.to_csv('stats/test_path_counts.csv')
    # # path_counts_df = pd.read_csv('stats/test_path_counts.csv')
    # # path_total_counts = path_counts_df.groupby(['board_name','move_number_in_path']).sum()
    # # path_total_counts.to_csv('stats/test_path_totals.csv')
    # # path_total_counts = pd.read_csv('stats/test_path_totals.csv')
    # # move_probabilities_heuristics = move_probabilities_heuristics[['board_name', 'heuristic',  'move_number_in_path', 'path', 'prob_path']]
    # # heuristics_path_probs_df = move_probabilities_heuristics.groupby(['board_name', 'heuristic', 'path', 'move_number_in_path']).first()
    # # heuristics_path_probs_df.to_csv('stats/path_probs_heuristics.csv')
    # # heuristics_path_probs_df = pd.read_csv('stats/path_probs_heuristics.csv')
    # #
    # # # path_counts_df['participants_prob'] = np.nan
    # # #
    # # # for index, row in path_counts_df.iterrows():
    # # #     path_total_counts_path = path_total_counts.loc[(path_total_counts['board_name'] == row['board_name']) & (path_total_counts['move_number_in_path'] == row['move_number_in_path'])]
    # # #     total_path = path_total_counts_path.iloc[0]['count']
    # # #     # print 'done'
    # # #     path_counts_df.loc[index,'participants_prob'] = float(row['count'])/total_path
    # # #     # row[path_total_counts] = float(row['count'])/total_path
    # #
    # # # path_counts_df.to_csv('stats/path_probs_participants.csv')
    # # path_counts_df = pd.read_csv('stats/path_probs_participants.csv')
    # # path_probs_participants_heuristic_df = pd.merge(path_counts_df, heuristics_path_probs_df, on=['board_name','move_number_in_path','path'], how='left')
    # # path_probs_participants_heuristic_df.to_csv('stats/probs_paths_participants_heuristic.csv')
    # path_probs_participants_heuristic_df = pd.read_csv('stats/probs_paths_participants_heuristic.csv')
    # path_probs_participants_heuristic_df = path_probs_participants_heuristic_df.loc[(path_probs_participants_heuristic_df['participants_prob'] >= 0.05) & (path_probs_participants_heuristic_df['move_number_in_path'] < 8) & (path_probs_participants_heuristic_df['board_name'].isin(['6_easy_full','6_easy_pruned']))]
    # heuristic_vals = []
    # correlations = []
    # move_numbers = []
    # path_lenghts = path_probs_participants_heuristic_df['move_number_in_path'].unique()
    # for i in range(50):
    #     for l in path_lenghts:
    #         df_l = path_probs_participants_heuristic_df.loc[path_probs_participants_heuristic_df['move_number_in_path']==l]
    #         df_sample = df_l.sample(frac=0.7)
    #         # df_sample = path_probs_participants_heuristic_df.sample(frac=0.7)
    #         for heuristic in heuristics:
    #             df_sample_h = df_sample.loc[(df_sample['heuristic'] == heuristic) & (df_sample['move_number_in_path'] == l)]
    #             heuristic_vals.append(heuristic)
    #             move_numbers.append(l)
    #             correlations.append(stats.pearsonr(df_sample_h['participants_prob'].values, df_sample_h['prob_path'].values)[0])
    # correlations_df = pd.DataFrame({'heuristic': heuristic_vals, 'path_length': move_numbers, 'correlation': correlations})
    # correlations_df.to_csv('stats/heuristics_participants_paths_correlations_pearson005_l_6easy.csv')
    # exit()
    # for i in range(1,10):
    #     print 'path length = ' + str(i)
    #     path_probs_participants_heuristic_df_i = path_probs_participants_heuristic_df.loc[path_probs_participants_heuristic_df['move_number_in_path'] == i]
    #     for heuristic in heuristics:
    #         print heuristic
    #         path_probs_participants_heuristic_df_h = path_probs_participants_heuristic_df_i.loc[path_probs_participants_heuristic_df_i['heuristic'] == heuristic]
    #         # print path_probs_participants_heuristic_df_h['count'].values.sum()
    #         print stats.spearmanr(path_probs_participants_heuristic_df_h['participants_prob'].values, path_probs_participants_heuristic_df_h['prob_path'].values)
    #         # print stats.pearsonr(path_probs_participants_heuristic_df_h['participants_prob'].values, path_probs_participants_heuristic_df_h['prob_path'].values)
    # exit()
    # #
    #
    # # path_probs_participants_heuristic_df = path_probs_participants_heuristic_df.loc[(path_probs_participants_heuristic_df['move_number_in_path'] > 4) & (path_probs_participants_heuristic_df['move_number_in_path'] < 8)]
    # # path_probs_participants_heuristic_df = path_probs_participants_heuristic_df.loc[(path_probs_participants_heuristic_df['move_number_in_path'] < 4)]
    # # path_probs_participants_heuristic_df = path_probs_participants_heuristic_df.loc[(path_probs_participants_heuristic_df['heuristic'].isin(['blocking'])) & (path_probs_participants_heuristic_df['participants_prob'] >= 0.05) & (path_probs_participants_heuristic_df['move_number_in_path'] < 4)]
    # # path_probs_participants_heuristic_df = path_probs_participants_heuristic_df.loc[(path_probs_participants_heuristic_df['move_number_in_path'] < 4) & (path_probs_participants_heuristic_df['participants_prob'] >= 0.05)]
    # path_probs_participants_heuristic_df = path_probs_participants_heuristic_df.loc[(path_probs_participants_heuristic_df['participants_prob'] >= 0.05) & (path_probs_participants_heuristic_df['move_number_in_path'] < 10)]
    #
    # # heuristic_vals = []
    # # correlations = []
    # # move_numbers = []
    # # path_lenghts = path_probs_participants_heuristic_df['move_number_in_path'].unique()
    # # for i in range(50):
    # #     for l in path_lenghts:
    # #         df_sample = path_probs_participants_heuristic_df.sample(frac=0.7)
    # #         for heuristic in heuristics:
    # #             df_sample_h = df_sample.loc[(df_sample['heuristic'] == heuristic) & (df_sample['move_number_in_path'] == l)]
    # #             heuristic_vals.append(heuristic)
    # #             move_numbers.append(l)
    # #             correlations.append(stats.spearmanr(df_sample_h['participants_prob'].values, df_sample_h['prob_path'].values)[0])
    # # correlations_df = pd.DataFrame({'heuristic': heuristic_vals, 'path_length': move_numbers, 'correlation': correlations})
    # # correlations_df.to_csv('stats/heuristics_participants_paths_correlations_spearman005.csv')
    # # exit()
    # correlations_df = pd.read_csv('stats/heuristics_participants_paths_correlations.csv')
    # # density = correlations_df.loc[correlations_df['heuristic'] == 'density']
    # # linear = correlations_df.loc[correlations_df['heuristic'] == 'linear']
    # # nonlinear = correlations_df.loc[correlations_df['heuristic'] == 'non-linear']
    # # interaction = correlations_df.loc[correlations_df['heuristic'] == 'interaction']
    # # blocking = correlations_df.loc[correlations_df['heuristic'] == 'blocking']
    # # print np.mean(blocking.correlation.values)
    # # print np.mean(interaction.correlation.values)
    # # print stats.mannwhitneyu(blocking.correlation.values, interaction.correlation.values)
    # # print stats.mannwhitneyu(blocking.correlation.values, nonlinear.correlation.values)
    # # print stats.mannwhitneyu(blocking.correlation.values, linear.correlation.values)
    # # print stats.mannwhitneyu(blocking.correlation.values, density.correlation.values)
    # # print 'interaction'
    # # print stats.mannwhitneyu(interaction.correlation.values, nonlinear.correlation.values)
    # # print stats.mannwhitneyu(interaction.correlation.values, linear.correlation.values)
    # # print stats.mannwhitneyu(interaction.correlation.values, density.correlation.values)
    # # print 'nonlinear'
    # # print stats.mannwhitneyu(nonlinear.correlation.values, linear.correlation.values)
    # # print stats.mannwhitneyu(nonlinear.correlation.values, density.correlation.values)
    # # print 'linear'
    # # print stats.mannwhitneyu(linear.correlation.values, density.correlation.values)
    # # exit()
    # # # ax = sns.lmplot(x='participants_prob', y='prob_path', data=path_probs_participants_heuristic_df, fit_reg=False, col='heuristic', col_order=['density','linear','non-linear','interaction','blocking'])
    # # # ax = sns.lmplot(x='participants_prob', y='prob_path', data=path_probs_participants_heuristic_df, fit_reg=False, hue='heuristic')
    # # sns.set(font_scale=1.5, style="whitegrid")
    # # path_probs_participants_heuristic_df = path_probs_participants_heuristic_df.loc[path_probs_participants_heuristic_df['heuristic'] == 'blocking']
    # # ax = sns.lmplot(x='participants_prob', y='prob_path', data=path_probs_participants_heuristic_df, fit_reg=True, scatter_kws={"s": 35})
    # # ax = ax.set_axis_labels('Observed Likelihood', 'Predicted Likelihood')
    # # # ax.set_xlabel('Observed Likelihood', fontsize=16)
    # # # ax.tick_params(labelsize=14)
    # # # ax.set(xlabel='Observed Path Likelihood', ylabel='Predicted Path Likelihood')
    # # # ax.set(yscale="log")
    # # # ax.set(xscale="log")
    # # # plt.xlim(0,0.05)
    # # plt.show()
    # # exit()
    # #
    # correlations_df['heuristic'] = correlations_df['heuristic'].map({'density':'Density', 'linear':  'Linear','non-linear':'Non-linear', 'interaction': 'Interaction','blocking':'Forcing'})
    # ax = sns.barplot(x='path_length',y='correlation', ci=68, data=correlations_df, hue='heuristic')
    # plt.gca().legend().set_title('')
    #
    #
    # ax.set_ylabel('Correlation', fontsize=16)
    # ax.set_xlabel('Path length', fontsize=16)
    # ax.tick_params(labelsize=14)
    # plt.subplots(2,1,figsize=(4,7))
    # plt.subplot(2,1,2)
    # # correlations_df = correlations_df.loc[correlations_df['path_length'] < 8]
    # ax = sns.barplot(x='heuristic',y='correlation', ci=68, data=correlations_df)
    # # ax.set(xlabel='', ylabel='Correlation - Observed & Predicted Path Likelihood', fontsize=16)
    # ax.set_ylabel('Correlation', fontsize=16)
    # ax.set_xlabel('', fontsize=16)
    # ax.tick_params(labelsize=14)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    # plt.legend(loc='best')
    # plt.show()
    # exit()
    # ------- end path probabilities population vs. heuristic prediction


    # --------heuristic fit population confidence intervals --------
    # print 'start'
    # #
    # # # move_probabilities_heuristics = pd.read_csv('stats/heuristics_byMove_winScoresChanged.csv')
    # players = move_probabilities_heuristics['userid'].unique()
    # heuristics = ['density', 'linear', 'non-linear', 'interaction', 'blocking']
    # #
    # # players_to_include = []
    # #
    # # # filter out players with bad fit
    # # log_likelihood_threshold = -3
    # # for player in players:
    # #     # player_data = move_probabilities_heuristics.loc[(move_probabilities_heuristics['userid'] == player) & (move_probabilities_heuristics['move_number'] % 2 == 0)]
    # #     player_data = move_probabilities_heuristics.loc[(move_probabilities_heuristics['userid'] == player)]
    # #     max_log_likeilhood = -10000
    # #
    # #     for heuristic in heuristics:
    # #         heuristic_likelhioods = player_data.loc[player_data['heuristic'] == heuristic]
    # #         mean_likeilhood = heuristic_likelhioods['log_move'].mean()
    # #         if mean_likeilhood > max_log_likeilhood:
    # #             max_log_likeilhood = mean_likeilhood
    # #     if (max_log_likeilhood > log_likelihood_threshold) & (player_data['move_number'].max() > 4):
    # #         players_to_include.append(player)
    # #
    # # # for each player, sample from moves, fit heuristic. store proportion of population for each heuristic
    # # move_probabilities_heuristics_filtered = move_probabilities_heuristics.loc[move_probabilities_heuristics['userid'].isin(players_to_include)]
    # # players = move_probabilities_heuristics_filtered['userid'].unique()
    # # num_players = len(players)
    # # num_samples = 100
    # # sample_size = 0.7
    # # # heuristics = ['density','non-linear','blocking']
    # # # heuristics = ['density', 'linear', 'non-linear', 'interaction', 'blocking']
    # # population_distributions = {}
    # # heuristic_list = []
    # # prop_list = []
    # # for heuristic in heuristics:
    # #     population_distributions[heuristic] = []
    # # players_list = []
    # # fitted_heuristic_list = []
    # # fitted_heuristic_likelihood = []
    # # for i in range(num_samples):
    # #
    # #     heuristics_player_counts = {}
    # #     for heuristic in heuristics:
    # #         heuristics_player_counts[heuristic] = 0.0
    # #     train_test_same_counter = 0.0
    # #     player_counter = 0.0
    # #     for player in players:
    # #         player_data = move_probabilities_heuristics_filtered.loc[(move_probabilities_heuristics_filtered['userid'] == player)]
    # #         num_moves_player = player_data['move_number'].max()
    # #         moves = np.arange(1,num_moves_player+1)
    # #         train_moves, test_moves = train_test_split(moves, test_size=0.3)  # random split
    # #         sampled_moves = player_data.loc[player_data['move_number'].isin(train_moves)]
    # #         max_log_likeilhood = -10000
    # #         fitted_heuristic = None
    # #         for heuristic in heuristics:
    # #             # heuristic_likelhioods = player_data.loc[player_data['heuristic'] == heuristic]
    # #             heuristic_likelhioods = sampled_moves.loc[sampled_moves['heuristic'] == heuristic]
    # #             mean_likeilhood = heuristic_likelhioods['log_move'].mean()
    # #             if mean_likeilhood > max_log_likeilhood:
    # #                 max_log_likeilhood = mean_likeilhood
    # #                 fitted_heuristic = heuristic
    # #         heuristics_player_counts[fitted_heuristic] += 1.0
    # #         players_list.append(player)
    # #         fitted_heuristic_list.append(fitted_heuristic)
    # #         fitted_heuristic_likelihood.append(max_log_likeilhood)
    # #     sum_distribution_check = 0.0
    # #     for heuristic in heuristics:
    # #         prop_heuristic = heuristics_player_counts[heuristic]/num_players
    # #         sum_distribution_check += prop_heuristic
    # #         population_distributions[heuristic].append(prop_heuristic)
    # #         heuristic_list.append(heuristic)
    # #         prop_list.append(prop_heuristic*100)
    # #     # print sum_distribution_check
    # #
    # # population_distributions_df = pd.DataFrame(population_distributions)
    # # population_distributions_df2 = pd.DataFrame({'heuristic': heuristic_list, 'percent': prop_list})
    # # population_distributions_df2.to_csv('stats/population_distributions210818.csv')
    # # heuristic_fitted_df = pd.DataFrame({'userid': players_list, 'fitted_heuristic': fitted_heuristic_list, 'log_likelihood': fitted_heuristic_likelihood})
    # # heuristic_fitted_df.to_csv('stats/fitted_heuristics_filtered_cogsci_210818.csv')
    #
    # population_distributions_df2 = pd.read_csv('stats/population_distributions210818.csv')
    # heuristic_fitted_df = pd.read_csv('stats/fitted_heuristics_filtered_cogsci_210818.csv')
    # # generate table 2 in pnas
    # # for heuristic in heuristics:
    # #     print heuristic
    # #     print stats.t.interval(0.95, len(population_distributions_df[heuristic].values)-1, loc=np.mean(population_distributions_df[heuristic].values), scale=stats.sem(population_distributions_df[heuristic].values))
    # #
    # #     print bs.bootstrap(population_distributions_df[heuristic].values, stat_func=bs_stats.mean, is_pivotal=False)
    # #     heuristic_fitted_df_filtered = heuristic_fitted_df.loc[heuristic_fitted_df['fitted_heuristic'] == heuristic]
    # #     print bs.bootstrap(heuristic_fitted_df_filtered['log_likelihood'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # #     # print population_distributions_df[heuristic].median()
    # #     # print medianCI(population_distributions_df[heuristic], 0.95, 0.5)
    # #     # print bs.bootstrap(population_distributions_df[heuristic].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # # # generate figure for table 2 pnas
    # # plt.subplot(1, 2, 1)
    # # ax = sns.barplot(x='heuristic', y='percent', n_boot=1000, ci=68, data=population_distributions_df2, order=['density','linear','non-linear','interaction','blocking'])
    # # # ax = sns.regplot(x="moves", y="solutionAndValidationCorrectPercent", x_estimator=np.mean, data=alphaBetaBehvaior, color="r", fit_reg=False,  ci=68)
    # # # ax.set(xscale="log")
    # # ax.set(ylim=(0, 100))
    # # # ax.set(xlabel='Board Complexity', ylabel='Percent Correct')
    # # # # ax2 = ax.twinx()
    # # plt.subplot(1, 2, 2)
    # # ax = sns.barplot(x='fitted_heuristic', y='log_likelihood', n_boot=1000, ci=68, data=heuristic_fitted_df, order=['density','linear','non-linear','interaction','blocking'])
    # # # ax = sns.regplot(x="moves", y="moves", data=alphaBetaBehvaior,  x_estimator=np.mean, ci=68, color="silver", fit_reg=False)
    # # # # ax3 = ax.twinx()
    # # # sns.regplot(x="moves", y="actionsSolution", data=alphaBetaBehvaior,  x_estimator=np.mean, ci=68, color="b", fit_reg=False, ax=ax)
    # # # ax.set(xscale="log")
    # # # ax.set(yscale="log")
    # # # ax.set(xlabel='Board Complexity', ylabel='Search Size')
    # # plt.tight_layout()
    # # plt.show()
    #
    # population_distributions_df2['heuristic'] = population_distributions_df2['heuristic'].map({'density':'Density', 'linear':  'Linear','non-linear':'Non-linear', 'interaction': 'Interaction','blocking':'Forcing'})
    # heuristic_fitted_df['fitted_heuristic'] = heuristic_fitted_df['fitted_heuristic'].map({'density':'Density', 'linear':  'Linear','non-linear':'Non-linear', 'interaction': 'Interaction','blocking':'Forcing'})
    # sns.set(style='white')
    #
    # plt.subplot(1, 2, 2)
    # ax = sns.barplot(y='heuristic', x='percent', n_boot=1000, ci=68, data=population_distributions_df2,  order=['Density','Linear','Non-linear','Interaction','Forcing'])
    # ax.set_xlabel('Percent participants', fontsize=14)
    # ax.set_ylabel('Scoring strategy', fontsize=14)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.set_ticks_position('right')
    # # ax = sns.regplot(x="moves", y="solutionAndValidationCorrectPercent", x_estimator=np.mean, data=alphaBetaBehvaior, color="r", fit_reg=False,  ci=68)
    # # ax.set(xscale="log")
    # # ax.set(ylim=(0, 100))
    # # ax.set(xlim=(0,100))
    # # ax.set(xlabel='Board Complexity', ylabel='Percent Correct')
    # # # ax2 = ax.twinx()
    # plt.subplot(1, 2, 1)
    # ax = sns.barplot(y='fitted_heuristic', x='log_likelihood', n_boot=1000, ci=68, data=heuristic_fitted_df, order=['Density','Linear','Non-linear','Interaction','Forcing'])
    #
    # ax.set_xlabel('Log Likelihood', fontsize=14)
    # ax.set_ylabel('Scoring strategy', fontsize=14)
    # # heuristic_fitted_df.plot("fitted_heuristic", "log_likelihood", kind="barh", color=sns.color_palette("deep", 3))
    # # ax = sns.regplot(x="moves", y="moves", data=alphaBetaBehvaior,  x_estimator=np.mean, ci=68, color="silver", fit_reg=False)
    # # # ax3 = ax.twinx()
    # # sns.regplot(x="moves", y="actionsSolution", data=alphaBetaBehvaior,  x_estimator=np.mean, ci=68, color="b", fit_reg=False, ax=ax)
    # # ax.set(xscale="log")
    # # ax.set(yscale="log")
    # # ax.set(xlabel='Board Complexity', ylabel='Search Size')
    # plt.tight_layout(pad=2.5)
    # # plt.savefig('figures_pnas/heuristic_fit_population.jpg')
    # plt.show()
    #
    # print 1/0


    # --------heuristic fit significance split --------
    # print 'start'
    # move_probabilities_heuristics = pd.read_csv('stats/move_probs_heuristics020818.csv')
    # # move_probabilities_heuristics = pd.read_csv('stats/heuristics_byMove_winScoresChanged.csv')
    # players = move_probabilities_heuristics['userid'].unique()
    # heuristics = ['density','non-linear','blocking']
    # # heuristics = ['density','linear','non-linear','interaction']
    # train_test_same_counter = 0.0
    # player_counter = 0.0
    # for player in players:
    #     # player_data = move_probabilities_heuristics.loc[(move_probabilities_heuristics['userid'] == player) & (move_probabilities_heuristics['move_number'] % 2 == 0)]
    #     player_data = move_probabilities_heuristics.loc[(move_probabilities_heuristics['userid'] == player)]
    #     # num_x_moves_player = player_data.loc[player_data['heuristic'] == 'density'].shape[0]
    #     num_moves_player = player_data['move_number'].max()
    #     # # moves = np.arange(1,num_x_moves_player+1)
    #     moves = player_data['path_number'].unique()
    #     if num_moves_player < 5:
    #         continue
    #     train_moves, test_moves = train_test_split(moves, test_size=0.3)  # random split
    #     print train_moves
    #     print test_moves
    #
    #     # ordered splity
    #     # num_train = int(0.7*len(moves))
    #     # train_moves = moves[:num_train]
    #     # test_moves = moves[num_train:len(moves)]
    #
    #
    #     train = player_data.loc[player_data['path_number'].isin(train_moves)]
    #     test = player_data.loc[player_data['path_number'].isin(test_moves)]
    #
    #
    #     fitted_heuristic_train = None
    #     max_log_likeilhood_train = -10000
    #     fitted_heuristic_test = None
    #     max_log_likeilhood_test = -10000
    #     for heuristic in heuristics:
    #         heuristic_likelhioods_train = train.loc[train['heuristic'] == heuristic]
    #         mean_likeilhood = heuristic_likelhioods_train['log_move'].mean()
    #         if mean_likeilhood > max_log_likeilhood_train:
    #             max_log_likeilhood_train = mean_likeilhood
    #             fitted_heuristic_train = heuristic
    #
    #         heuristic_likelhioods_test = test.loc[test['heuristic'] == heuristic]
    #         mean_likeilhood = heuristic_likelhioods_test['log_move'].mean()
    #         if mean_likeilhood > max_log_likeilhood_test:
    #             max_log_likeilhood_test = mean_likeilhood
    #             fitted_heuristic_test = heuristic
    #
    #         # print fitted_heuristic_test
    #     # print '--------'
    #     # print fitted_heuristic_train +';'+fitted_heuristic_test
    #
    #     if fitted_heuristic_train is None:
    #         print 'here'
    #     if fitted_heuristic_test is None:
    #         print 'here'
    #     if fitted_heuristic_train == fitted_heuristic_test:
    #         train_test_same_counter += 1.0
    #         userids_sig.append(player)
    #         fitted_heuristics_sig.append(fitted_heuristic_train)
    #         # print fitted_heuristic_train
    #     else:
    #         # print fitted_heuristic_train +';'+fitted_heuristic_test
    #         # print num_moves_player
    #         # print player
    #         print player_data['board_name'].unique()
    #     player_counter += 1
    # print train_test_same_counter
    # print player_counter
    # print train_test_same_counter/player_counter
    # sig_heuristic = {'userids':userids_sig, 'fitted_heuristic':fitted_heuristics_sig}
    # sig_heuristic_df = pd.DataFrame(sig_heuristic)
    # for heuristic in heuristics:
    #     print heuristic
    #     df_fitted_heuristic = sig_heuristic_df.loc[sig_heuristic_df['fitted_heuristic'] == heuristic]
    #     print float(df_fitted_heuristic.shape[0])/sig_heuristic_df.shape[0]
    # print 1/0


    # --------heuristic sensitivity analysis-------------
    # heuristics = ['Density','Linear','Non-linear','Interaction','Forcing']
    #
    # # win_scores = [25, 250, 500, 750, 1000,1250, 1500, 1750, 2000, 2250, 2500]
    # win_scores = [250, 500, 750, 1000,1250, 1500, 1750, 2000, 2250, 2500]
    # # blocking_vals = [2,20,40,60,80,100,120,140,160,180,200]
    # blocking_vals = [20,40,60,80,100,120,140,160,180,200]
    # # log_changes = []
    #
    # # heuristics_sensitivity_scores =  heuristics_sensitivity.groupby(['userid','win_score']).mean().reset_index()
    # # heuristics_sensitivity_scores.to_csv('stats/heuristics_sensitivity_scores.csv')
    # heuristics_sensitivity_scores = pd.read_csv('stats/heuristics_sensitivity_scores.csv')
    #
    # # exit()
    # changes = []
    # folds = []
    # heuristics_names = []
    # heuristics_sensitivity_score_25 = heuristics_sensitivity_scores.loc[heuristics_sensitivity_scores['win_score'] == 25]
    # for win_score in win_scores:
    #     heuristics_sensitivity_score = heuristics_sensitivity_scores.loc[heuristics_sensitivity_scores['win_score'] == win_score]
    #     fold = win_score/25.0
    #     sum_changes = 0.0
    #     for user in heuristics_sensitivity_score['userid'].unique():
    #         user_scores_25 = heuristics_sensitivity_score_25.loc[heuristics_sensitivity_score_25['userid'] == user]
    #         user_scores = heuristics_sensitivity_score.loc[heuristics_sensitivity_score['userid'] == user]
    #         for h in heuristics:
    #             # change = abs(np.log(user_scores[h].values[0]/user_scores_25[h].values[0]))
    #             change = abs((user_scores[h].values[0]/user_scores_25[h].values[0]))
    #             if change < 1:
    #                 change = 1.0/change
    #             change = change - 1
    #             changes.append(change)
    #             folds.append(fold)
    #             heuristics_names.append(h)
    #
    # changes_dict = {'fold': folds, 'changes':changes, 'heuristic': heuristics_names}
    # changes_df = pd.DataFrame(changes_dict)
    #
    # for h in heuristics:
    #     print h
    #     max_fold_df = changes_df.loc[(changes_df['fold'] == 100) & (changes_df['heuristic'] == h) ]
    #     print bs.bootstrap(max_fold_df['changes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # exit()
    # plt.subplot(1,2,1)
    # ax = sns.pointplot('fold', 'changes', hue='heuristic', data=changes_df, legend_out=False, legend=False)
    # plt.gca().legend().set_title('')
    # plt.legend(loc='best')
    # ax.set_ylabel('Mean percent change in log-likelihood', fontsize=16)
    # ax.set_xlabel('Fold change in winning score', fontsize=16)
    # ax.tick_params(labelsize=14)
    # # plt.show()
    #
    # # heuristics_sensitivity_blocking = heuristics_sensitivity.groupby(['userid','blocking_score']).mean().reset_index()
    # # heuristics_sensitivity_blocking.to_csv('stats/heuristics_sensitivity_blocking_scores.csv')
    # heuristics_sensitivity_blocking= pd.read_csv('stats/heuristics_sensitivity_blocking_scores.csv')
    # changes = []
    # folds = []
    # heuristics_names = []
    # heuristics_blocking_2 = heuristics_sensitivity_blocking.loc[heuristics_sensitivity_blocking['blocking_score'] == 2]
    # for blocking_score in blocking_vals:
    #     heuristics_sensitivity_score = heuristics_sensitivity_blocking.loc[heuristics_sensitivity_scores['blocking_score'] == blocking_score]
    #     fold = blocking_score/2.0
    #     sum_changes = 0.0
    #     for user in heuristics_sensitivity_score['userid'].unique():
    #         user_scores_2 = heuristics_blocking_2.loc[heuristics_blocking_2['userid'] == user]
    #         user_scores = heuristics_sensitivity_score.loc[heuristics_sensitivity_score['userid'] == user]
    #         for h in heuristics:
    #             # change = abs(np.log(user_scores[h].values[0]/user_scores_25[h].values[0]))
    #             change = abs((user_scores[h].values[0]/user_scores_2[h].values[0]))
    #             if change < 1:
    #                 change = 1.0/change
    #             change = change - 1
    #             changes.append(change)
    #             folds.append(fold)
    #             heuristics_names.append(h)
    # plt.subplot(1,2,2)
    # changes_dict_blocking = {'fold': folds, 'changes':changes, 'heuristic': heuristics_names}
    # changes_blocking_df = pd.DataFrame(changes_dict_blocking)
    # ax = sns.pointplot('fold', 'changes', data=changes_df)
    # ax.set_ylabel('Mean percent change in log-likelihood', fontsize=16)
    # ax.set_xlabel('Fold change in blocking score', fontsize=16)
    # ax.tick_params(labelsize=14)
    # plt.tight_layout()
    # plt.show()
    # #
    # #
    # #
    # # for h in heuristics:
    # #     print h
    # #     print bs.bootstrap(heuristics_sensitivity[h].values, stat_func=bs_stats.mean, is_pivotal=False)
    # exit()
    # --------heuristic fit analysis-------------
    # aggregations = {
    #     'userid': 'count'
    #     # 'userid': 'sum'
    # }
    #
    # # heuristics_fit_paramters = heuristics_sensitivity.groupby(['win_score','blocking_score','fitted_heuristic']).agg(aggregations)
    # # heuristics_fit_paramters.to_csv("stats/heuristic_fit_agg250718.csv")
    # # print 1/0
    # # sensitivity analysis
    # heuristics_fit_paramters = pd.read_csv("stats/heuristic_fit_agg250718.csv")
    # heuristics = ['density','linear','non-linear','interaction','blocking','interaction_blind','blocking_blind']
    # # win_scores = [25,50,100,200,400,800,1600,3200,6400,12800]
    # # blocking_vals = [0.05, 0.1, 0.2, 0.4, 0.8]
    # win_scores = [25, 250, 500, 750, 1000,1250, 1500, 1750, 2000, 2250, 2500]
    # blocking_vals = [2,20,40,60,80,100,120,140,160,180,200]
    # prob_changes = []
    # paramaeter_changes = []
    # heuristics_for_data = []
    # heuristics_fit = pd.read_csv("stats/cogsci_heuristics250718_1.csv")
    # heuristics_fit_correct = heuristics_fit.loc[heuristics_fit['correct'] == 1]
    # heuristics_fit_wrong = heuristics_fit.loc[heuristics_fit['correct'] == 0]
    #
    # # print 'all'
    # # for heuristic in heuristics:
    # #     print heuristic
    # #     print bs.bootstrap(heuristics_fit[heuristic].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # #correct
    # print 'correct'
    # for heuristic in heuristics:
    #     print heuristic
    #     print bs.bootstrap(heuristics_fit_correct[heuristic].values, stat_func=bs_stats.mean, is_pivotal=False)
    # #wrong
    # print 'wrong'
    # for heuristic in heuristics:
    #     print heuristic
    #     print bs.bootstrap(heuristics_fit_wrong[heuristic].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print 1/0
    # probs = []
    # for heuristic in heuristics:
    #     probs = []
    #     print heuristic
    #     for i in range(len(win_scores)):
    #         print win_scores[i]
    #         # proprtions_heuristic = heuristics_fit_paramters.loc[(heuristics_fit_paramters['fitted_heuristic'] == heuristic) & (heuristics_fit_paramters['win_score'] == win_scores[i])]
    #         proprtions_heuristic = heuristics_fit_paramters.loc[(heuristics_fit_paramters['fitted_heuristic'] == heuristic) & (heuristics_fit_paramters['win_score'] == win_scores[i]) & (heuristics_fit_paramters['blocking_score'] == 20)]
    #         avg_proportion_heuristic = proprtions_heuristic.proportion_participants.mean()
    #         print avg_proportion_heuristic
    #         probs.append(avg_proportion_heuristic)
    #         # heuristics_for_data.append(heuristic)
    #         # paramaeter_changes.append(win_scores[i])
    #         if (i > 0) & (avg_proportion_heuristic > 0.1):
    #         # if (i > 0) :
    #             # print i
    #             prob_changes.append(abs(math.log((avg_proportion_heuristic/probs[i-1]),2)))
    #             # prob_changes.append(abs(avg_proportion_heuristic/probs[i-1]))
    #             # prob_changes.append((abs(avg_proportion_heuristic-probs[i-1])))
    #             print probs[i-1]
    #             print math.log(avg_proportion_heuristic/probs[i-1])
    #             paramaeter_changes.append((win_scores[i]))
    #             heuristics_for_data.append(heuristic)
    #
    # changes_data = {'parameter_change': paramaeter_changes, 'proportion_change':prob_changes, 'heuristic':heuristics_for_data}
    # changes_df = pd.DataFrame(changes_data)
    # ax = sns.pointplot(x="parameter_change", y="proportion_change", hue='heuristic',data=changes_data, legend_out=False, legend=False)
    # # ax.set(yscale="log")
    # # ax.set(xscale="log")
    # plt.ylim([0,1])
    # plt.legend(loc='best')
    # plt.show()
    #         # print heuristic
    #         # heuristics_fit_paramters_filter = heuristics_fit_paramters.loc[heuristics_fit_paramters['fitted_heuristic'] == heuristic]
    #         # print bs.bootstrap(heuristics_fit_paramters_filter['proportion_participants'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # # participant heuristic fit figure-----
    # # heuristics_sensitivity_filtered = heuristics_sensitivity.loc[(heuristics_sensitivity['win_score'] == 100) & (heuristics_sensitivity['blocking_score'] == 10)]
    # #
    # # heuristics_sensitivity_filtered["heuristic_correct"] = zip(heuristics_sensitivity_filtered.user_count, heuristics_sensitivity_filtered.num_participants)
    # #
    # #
    # # def weighted_mean(x, **kws):
    # #     user_count, correct = map(np.asarray, zip(*x))
    # #
    # #     return user_count.sum() / float(correct[0])
    # #
    # # # lambda x: True if x % 2 == 0 else False
    # # ax = sns.barplot(x = 'fitted_heuristic', y = 'heuristic_correct', hue="correct",   n_boot=1000, data = heuristics_sensitivity_filtered,  estimator=weighted_mean, orient="v",order=['density','linear','non-linear', 'interaction_blind', 'interaction','blocking_blind','blocking'])
    # # # ax = sns.barplot(x = 'fitted_heuristic', y = 'blocking', hue="correct",   n_boot=1000, data = heuristics_sensitivity_filtered)
    # #
    # # # ax = sns.factorplot(x="board", y="solutionAndValidationCorrectPercent", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 MC', '10 MC', '6 HC', '10 HC', '10 DC'],  markers=['o','^'], legend_out=False, legend=False)
    # # # data['board'] = data['board'].map({'6 MC': 'MC6','10 MC': 'MC10','6 HC': 'HC6','10 HC': 'HC10','10 DC': 'DC10'})
    # # # ax = sns.barplot(x="board", y="solutionAndValidationCorrectPercent",  hue="condition", data=data, n_boot=1000, order=['MC6', 'MC10', 'HC6', 'HC10', 'DC10'])
    # # #
    # # # ax.set(xlabel='Board', ylabel='Percent Correct')
    # # # lw = ax.ax.lines[0].get_linewidth()
    # # # plt.setp(ax.ax.lines,linewidth=lw)
    # # plt.ylim([0, 1])
    # # plt.legend(loc='best')
    # # plt.show()
    # # participant success rate figure end-----
    #
    #
    # # print heuristics_fit_paramters
    # print 1/0
    # --------end heuristic fit analysis------


    # get_user_stats(dynamics,exploreExploit)
    #
    # # exploreExploitRaw = pd.read_csv("stats/exploreExploitTimesPathLength0416.csv")
    # # f = {'explore_time':['mean','std'], 'exploit_time':['mean','std'], 'solved': ['first'], 'board_name': ['first']}
    # # exploreExploitAvg = pd.read_csv("stats/explore_exploit_avg_1604.csv")
    # # exploreExploitAvg = exploreExploitAvg.sort_values(by='explore_time')
    # # ax = sns.pointplot(x="userid", y="explore_time", data=exploreExploitRaw,join=False, order=exploreExploitAvg['userid'])
    # # plt.show()
    # # #
    # # # exploreExploitAvg = exploreExploitRaw.groupby('userid').agg(f)
    # # # exploreExploitAvg.to_csv('stats/explore_exploit_avg_1604.csv')
    # # print 1/0
    #
    # # stop conditions exploration
    # # & (dynamics['move_number_in_path'] == 6)
    # # dynamics_filtered = dynamics.loc[(dynamics['action'] == 'click') & (dynamics['board_name'] == '6_hard_full') & (dynamics['state_score_x'] > 9) & (dynamics['state_score_x'] < 100) &  (dynamics['last_move'] == True)]
    # # & (dynamics['move_number_in_path'] == 6)
    # board_size = 10
    # moves_to_win = 10
    # # & (dynamics['board_size'] == board_size) & (dynamics['moves_to_win'] == moves_to_win)
    # # & (dynamics['move_number_in_path'] >2) & (dynamics['move_number_in_path'] < 4)
    # # & (dynamics['state_score_x'] > -100) & (dynamics['state_score_x'] < 100)
    # # & ((dynamics['move_number_in_path'] % 2) == 1)
    # dynamics_filtered = dynamics.loc[(dynamics['action'] == 'click') & (dynamics['move_number_in_path'] < 9) & (dynamics['loss_x']==1) & (dynamics['last_move_ind']==1)]
    # # dynamics_filtered = dynamics.loc[(dynamics['action'] == 'click') & (dynamics['move_number_in_path'] < 9) & (dynamics['loss_x']==1)]
    #
    # print dynamics_filtered.shape[0]
    # print 1/0
    # lr = LogisticRegression(class_weight='balanced')
    # rf = RandomForestClassifier(n_estimators=25,max_depth=10, class_weight='balanced')
    # # lr = LogisticRegression()
    # y = dynamics_filtered.last_move_ind
    # # df = dynamics_filtered[['state_score_x','loss','win', 'explore','move_number_in_path']]
    # # df = dynamics_filtered[['move_number_in_path']]
    # # print np.mean(y)
    # # predicted = cross_val_predict(lr, df, y, cv=10)
    # # scores_lr = cross_val_score(lr, df, y, cv=10)
    # # print np.mean(predicted)
    # # print scores_lr
    # # print 'done'
    # # df = dynamics_filtered[['explore_norm','moves_to_win','board_size']]
    # df = dynamics_filtered[['explore_norm','move_number_in_path','win_x','loss_x','moves_to_win', 'state_score_x','blocking','blocking_density','density']]
    # # df = dynamics_filtered[[ 'explore_norm','move_number_in_path','state_score_x']]
    # poly = PolynomialFeatures(interaction_only=True,include_bias = False)
    # df1 = poly.fit_transform(df)
    # gkf = GroupKFold(n_splits=10)
    # cv= gkf.split(df1, y, groups=dynamics_filtered[[ 'userid']])
    #     # print("%s %s" % (train, test))
    #
    # # scores_lr = cross_val_score(lr, df1, y, cv=cv, scoring='roc_auc')
    # # print scores_lr
    #
    #
    #
    # # lr.fit(df1, y)
    # # print lr.coef_
    # # df = dynamics_filtered[['explore','state_score_x', 'path_number']]
    # cv= gkf.split(df1, y, groups=dynamics_filtered[[ 'userid']])
    # # scores_rf = cross_val_score(rf, df1, y, cv=cv, scoring='roc_auc')
    # # print scores_rf
    #
    # tprs = []
    # base_fpr = np.linspace(0, 1, 101)
    # y = dynamics_filtered[['last_move_ind']].values
    # plt.figure(figsize=(5, 5))
    # cv= gkf.split(df1, y, groups=dynamics_filtered[[ 'userid']])
    # aucs = []
    # for i, (train, test) in enumerate(cv):
    #     model = lr.fit(df1[train], y[train].ravel())
    #     print model.coef_
    #     # print model.summary()
    #     y_score = model.predict_proba(df1[test])
    #
    #     fpr, tpr, _ = roc_curve(y[test], y_score[:, 1])
    #     aucs.append(metrics.auc(fpr, tpr))
    #     plt.plot(fpr, tpr, 'b', alpha=0.15)
    #     tpr = interp(base_fpr, fpr, tpr)
    #     tpr[0] = 0.0
    #     tprs.append(tpr)
    #
    # tprs = np.array(tprs)
    # mean_tprs = tprs.mean(axis=0)
    # std = tprs.std(axis=0)
    #
    # tprs_upper = np.minimum(mean_tprs + std, 1)
    # tprs_lower = mean_tprs - std
    #
    #
    # plt.plot(base_fpr, mean_tprs, 'b')
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    #
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.title(np.mean(aucs))
    # plt.show()
    #
    # print np.mean(dynamics_filtered.last_move_ind)
    # print 1/0
    # # six = ['6_hard_full', '6_easy_full', '6_hard_pruned', '6_easy_pruned']
    # #  # & (dynamics['move_number_in_path'] == 5)
    # dynamics_filtered = dynamics.loc[(dynamics['state_score_x'] < 90) & (dynamics['state_score_x'] > 4) & (dynamics['action'] == 'click')  &  (dynamics['move_number_in_path'] < 10)]
    # # dynamics_filtered = dynamics.loc[(dynamics['score_heuristic'] < 90) & (dynamics['score_heuristic'] > -90) & (dynamics['action'] == 'click')  & (dynamics['move_number_in_path'] == 2) & (dynamics['last_move'] == True)]
    # # dynamics_filtered = dynamics.loc[ (dynamics['action'] == 'click')]
    # # g = sns.FacetGrid(dynamics_filtered, row="board_size", col="last_move",  legend_out=False)
    # sns.barplot(data=dynamics_filtered, x='state_score_x_5', y='last_move',n_boot=1000)
    # # bins = np.linspace(0, 30, 30)
    # # g.map(plt.hist, "state_score_x", color="steelblue", bins=bins, lw=0)
    # # ax = g.ax_joint
    # # g.set_yscale('symlog')
    # # g.map(sns.distplot, "score_heuristic_x")
    # plt.show()
    # # g.map()
    # # # ten_to_win = ['6_hard_full', '10_hard_full', '10_medium_full']
    # # # eight_to_win = ['6_hard_pruned', '10_hard_pruned', '10_medium_pruned', '10_easy_full', '6_easy_full']
    # # # six_to_win = ['6_easy_pruned', '10_easy_pruned']
    # # # dynamics_filtered = dynamics.loc[dynamics['board_name'] in ten_to_win]
    # # # max_vals = dynamics.groupby(['userid','path_number', 'board_name', 'sizeType','condition','moves_to_win'], as_index=False)['move_number_in_path'].max()
    # # max_vals = dynamics.groupby(['userid','path_number'], as_index=False)['move_number_in_path'].max()
    # # resets = dynamics.loc[(dynamics['action'] == 'reset')]
    # # # g = sns.FacetGrid(max_vals, row="sizeType", col="condition", legend_out=False)
    # # # g = sns.FacetGrid(dynamics, row="move_number_in_path", legend_out=False)
    # # g = sns.FacetGrid(dynamics, col="moves_to_win", legend_out=False)
    # # # g = g.map(sns.distplot, "move_number_in_path")
    # # # g = g.map(sns.distplot, "score_move_x")
    # # bins = np.linspace(0, 15, 15)
    # # g.map(plt.hist, "move_number_in_path", color="steelblue", bins=bins, lw=0)
    # # # bins = np.linspace(-100, 100, 200)
    # # # g.map(plt.hist, "score_move_x", color="steelblue", bins=bins, lw=0)
    # #
    # # plt.show()
    # # print 1/0

    # -- explore-exploit correlation line
    # user_stats_exploration = pd.read_csv("stats/user_stats1604.csv")
    # df = user_stats_exploration[['explore_time','exploit_time']]
    # lof = LocalOutlierFactor()
    # outliers =  lof.fit_predict(df)
    #
    #
    # # ev = EllipticEnvelope(contamination=0.05)
    # # print ev.fit(df)
    # # outliers = ev.predict(df)
    # print len(outliers)
    # user_stats_exploration['outliers'] = outliers
    # # print user_stats_exploration['outliers']
    # user_stats_exploration_filtered = user_stats_exploration.loc[user_stats_exploration['outliers']!=-1]
    # # print user_stats_exploration_filtered['explore_time']
    # ax = sns.regplot(x="explore_time", y="exploit_time", data=user_stats_exploration_filtered, n_boot=1000)
    # plt.show()
    # l = ax.get_lines()
    # x1 = l[0]._path._vertices[0][0]
    # y1 = l[0]._path._vertices[0][1]
    #
    # x2 = l[0]._path._vertices[len(l[0]._path._vertices)-1][0]
    # y2 = l[0]._path._vertices[len(l[0]._path._vertices)-1][1]
    # new_x = []
    # new_y = []
    # for index, row in user_stats_exploration_filtered.iterrows():
    #     x3 = row['explore_time']
    #     y3 = row['exploit_time']
    #     dx = x2 - x1
    #     dy = y2 - y1
    #     d2 = dx*dx + dy*dy
    #     nx = ((x3-x1)*dx + (y3-y1)*dy) / d2
    #     point = (dx*nx + x1, dy*nx + y1)
    #     new_x.append(point[0])
    #     new_y.append(point[1])
    #     # print point
    # user_stats_exploration_filtered['new_x'] = new_x
    # user_stats_exploration_filtered['new_y'] = new_y
    # # ax = sns.regplot(x="new_x", y="new_y", data=user_stats_exploration_filtered, n_boot=1000)
    # min_x_val = user_stats_exploration_filtered['new_x'].min()
    # min_y_val = user_stats_exploration_filtered['new_y'].min()
    # print min_x_val
    # min_explore = math.sqrt((math.pow(min_x_val,2) + math.pow(min_y_val,2)))
    # print min_explore
    # exploration = []
    # for index, row in user_stats_exploration_filtered.iterrows():
    #     distance = math.sqrt((math.pow(row['explore_time']-min_x_val,2) + math.pow(row['exploit_time']-min_y_val,2)))
    #     exploration.append(distance+min_explore)
    # user_stats_exploration_filtered['exploration'] = exploration
    # user_stats_exploration_filtered.to_csv('stats/exploreExploitCombined1604.csv')
    # print 1/0
    #
    # lr = LinearRegression()
    # y = user_stats_exploration_filtered.exploration
    # df = user_stats_exploration_filtered[['median_score']]
    # predicted = cross_val_predict(lr, df, y, cv=10)
    # #
    # fig, ax = plt.subplots()
    # ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    # ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # plt.show()

    # plt.clf()
    # ax = sns.distplot(user_stats_exploration_filtered['exploration'], )
    # g = sns.FacetGrid(user_stats_exploration_filtered, hue="condition", legend_out=False)
    # g = g.map(sns.distplot, "exploration")
    #
    # X = user_stats_exploration_filtered
    # y = # Some classes
    #
    # clf = linear_model.Lasso()
    # scores = cross_val_score(clf, X, y, cv=10)
    # g = sns.FacetGrid(user_stats_exploration_filtered, col="correctness", margin_titles=True)
    # user_stats_exploration = pd.read_csv("stats/exploreExploitCombined1604.csv")
    # # user_stats_exploration_filtered =user_stats_exploration.loc[user_stats_exploration['exploration']]
    # # user_stats_exploration_correct = user_stats_exploration.loc[user_stats_exploration['solved']=='validatedCorrect']
    # # user_stats_exploration_6hard = user_stats_exploration.loc[user_stats_exploration['typeSize']=='6hard']
    # g = sns.FacetGrid(user_stats_exploration, row="board", margin_titles=True)
    # # # #
    # # bins = np.linspace(0, 120, 20)
    # # g.map(plt.hist, "exploration", color="steelblue", bins=bins, lw=0)
    # # g.map(sns.distplot, "exploration", bins=bins)
    # test_char = "avg_first_move_score"
    # g.map(sns.regplot,"exploration", test_char);
    # print stats.spearmanr(user_stats_exploration['exploration'], user_stats_exploration[test_char])
    # #
    # # ax = sns.regplot(x="exploration", y="num_resets",data=user_stats_exploration, n_boot=1000)
    # plt.show()

    # feature_names = ["num_moves", "solved"]
    # df = pd.DataFrame(user_stats_exploration_filtered, columns=feature_names)
    # print df
    # target = pd.DataFrame(user_stats_exploration_filtered, columns=["exploration"])
    # print target


    # print reg.get_params()
    # print m
    # print b
    # plt.ylim(0,80)
    # plt.show()
    # get_user_stats()

    # log-likelhood
    # density = likelihood.loc[likelihood['heuristic'] == 'density']
    # linear = likelihood.loc[likelihood['heuristic'] == 'linear']
    # nonLinear = likelihood.loc[likelihood['heuristic'] == 'non-linear']
    # nonLinearInteraction  = likelihood.loc[likelihood['heuristic'] == 'non-linear_interaction']
    # blocking = likelihood.loc[likelihood['heuristic'] == 'blocking']
    # chance = likelihood.loc[likelihood['heuristic'] == 'chance']
    #
    # densityCorrect = density.loc[density['participants'] == 'correct']
    # densityWrong = density.loc[density['participants'] == 'wrong']
    # linearCorrect = linear.loc[linear['participants'] == 'correct']
    # linearWrong = linear.loc[linear['participants'] == 'wrong']
    # nonLinearCorrect = nonLinear.loc[nonLinear['participants'] == 'correct']
    # nonLinearWrong = nonLinear.loc[nonLinear['participants'] == 'wrong']
    # nonLinearInteractionCorrect = nonLinearInteraction.loc[nonLinearInteraction['participants'] == 'correct']
    # nonLinearInteractionWrong = nonLinearInteraction.loc[nonLinearInteraction['participants'] == 'wrong']
    # blockingCorrect = blocking.loc[blocking['participants'] == 'correct']
    # blockingWrong = blocking.loc[blocking['participants'] == 'wrong']
    #
    # print bs.bootstrap(data['timeMinutes'].values, stat_func=bs_stats.mean, is_pivotal=False)


    # print bootstrap_t_pvalue(wrong['actionsSolution'].values, correct['actionsSolution'].values)
    # print bs.bootstrap(density['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(linear['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinear['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearInteraction['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(blocking['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(chance['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print 'correct'
    # print bs.bootstrap(densityCorrect['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(linearCorrect['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearCorrect['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearInteractionCorrect['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(blockingCorrect['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(chance['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print 'wrong'
    # print bs.bootstrap(densityWrong['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(linearWrong['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearWrong['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearInteractionWrong['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(blockingWrong['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(chance['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    #
    # print 'p-values'
    # print bootstrap_t_pvalue(densityCorrect['value'].values, densityWrong['value'].values)
    # print bootstrap_t_pvalue(linearCorrect['value'].values, linearWrong['value'].values)
    # print bootstrap_t_pvalue(nonLinearCorrect['value'].values, nonLinearWrong['value'].values)
    # print bootstrap_t_pvalue(nonLinearInteractionCorrect['value'].values, nonLinearInteractionWrong['value'].values)
    # print bootstrap_t_pvalue(blockingCorrect['value'].values, blockingWrong['value'].values)
    # print 'hello'
    # sns.set_style("darkgrid")
    # ax = sns.factorplot(x="size_type", y="actionsSolution",col="condition", hue="solutionAndValidationCorrect", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax = sns.factorplot(x="board", y="solutionAndValidationCorrect", hue="condition", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'], markers=['o','^'], linestyles=["-", "--"], legend=False)
    sns.set(style="whitegrid")

    # --------------dynamics analysis----------------
    # user_stats_exploration = exploreExploit
    # print stats.spearmanr(user_stats_exploration['explore_time'], user_stats_exploration['exploit_time'])
    # exploreExploit_filtered1 = user_stats_exploration.loc[(user_stats_exploration['explore_time'] < 1000) & (user_stats_exploration['exploit_time'] < 1000) & (user_stats_exploration['solved']=='validatedCorrect')]
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # print exploreExploit_filtered1.shape[0]
    # print stats.spearmanr(exploreExploit_filtered1['explore_time'], exploreExploit_filtered1['exploit_time'])
    # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, marker='+', color='green')
    #
    # # plt.xlim(0,100)
    # # plt.ylim(0,100)
    # # plt.show()
    #
    # exploreExploit_filtered2 = user_stats_exploration.loc[(user_stats_exploration['explore_time'] < 1000) & (user_stats_exploration['exploit_time'] < 1000) & ((user_stats_exploration['solved']=='wrong')  | (user_stats_exploration['solved']=='solvedCorrect'))]
    # print stats.spearmanr(exploreExploit_filtered2['explore_time'], exploreExploit_filtered2['exploit_time'])
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered2, n_boot=1000, color='red')
    # # plt.xlim(0,100)
    # # plt.ylim(0,100)
    # plt.show()

    # boards = ['6_easy_full','6_easy_pruned', '10_easy_full', '10_easy_pruned','6_hard_full','6_hard_pruned', '10_hard_full', '10_hard_pruned',  '10_medium_full', '10_medium_pruned']
    # for board in boards:
    #     print board
    #     # sns.load_dataset
    #     exploreExploit_filtered1 = user_stats_exploration.loc[(user_stats_exploration['board']==board)]
    #     colors = {'solvedCorrect':'blue','validatedCorrect':'green', 'wrong':'red' }
    #     plt.scatter(exploreExploit_filtered1.explore_time, exploreExploit_filtered1.exploit_time,
    #                 c = exploreExploit_filtered1.solved_num, s=(exploreExploit_filtered1.num_resets**2), cmap="viridis")
    #     ax = plt.gca()
    #
    #     plt.colorbar(label="solved")
    #     plt.xlabel("explore_time")
    #     plt.ylabel("exploit_time")
    #
    #     #make a legend:
    #     # pws = [0.5, 1, 1.5, 2., 2.5]
    #     # for pw in pws:
    #     #     plt.scatter([], [], s=(pw**2), c="k",label=str(pw))
    #     #
    #     # h, l = plt.gca().get_legend_handles_labels()
    #     # plt.legend(h[1:], l[1:], labelspacing=1.2, title="num_resets", borderpad=1,
    #     #             frameon=True, framealpha=0.6, )
    #
    #     plt.show()
    # for board in boards:
    # exploreExploit_filtered1 = user_stats_exploration.loc[user_stats_exploration['board']==board]
    # exploreExploit_filtered1 = user_stats_exploration
    # exploreExploit_filtered1 = user_stats_exploration.loc[user_stats_exploration['explore_time']<100]
    # ax = sns.lmplot(x="explore_time", y="exploit_time",data=exploreExploit_filtered1,hue='condition', n_boot=1000,fit_reg=False)
    # plt.show()
    # # exploreExploit_filtered1 = user_stats_exploration.loc[user_stats_exploration['board'].str.endswith('full')]
    # colors = {'correct':'green', 'wrong':'red'}
    # cols = ['num_resets','num_unique_first_moves','num_moves_win_score','mean_score','solution_time','median_score','number_of_moves']
    # # mult_vals = [10.0, 50.0,50.0,50.0,5.0,10.0,10.0]
    # for i in range(len(cols)):
    #     # f, (ax1, ax2) = plt.subplots(2)
    #     p = cols[i]
    #     # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, scatter_kws={"s": (exploreExploit_filtered1.num_resets**2)})
    #     print p
    #     max_value = exploreExploit_filtered1[p].max()*1.0
    #     filt_min = exploreExploit_filtered1[exploreExploit_filtered1[p]>0.001]
    #     min_value = filt_min[p].min()*1.0
    #     # print exploreExploit_filtered1['norm_val']
    #
    #     mult = ((min_value+0.)**2)/30.0
    #     print mult
    #     ax = sns.lmplot(x="explore_time", y="exploit_time",data=exploreExploit_filtered1,hue='board', n_boot=1000,palette=colors,scatter_kws={"s": ((exploreExploit_filtered1[p])**2 )/mult},fit_reg=False, legend=False)
    #     # plt.scatter(exploreExploit_filtered1.explore_time, exploreExploit_filtered1.exploit_time,
    #     #             c = exploreExploit_filtered1['solved'].apply(lambda x: colors[x]), s=(exploreExploit_filtered1.num_resets**2), cmap="Paired")
    #     # ax = plt.gca()
    #
    #     # plt.colorbar(label="solved")
    #     plt.xlabel("explore_time")
    #     plt.ylabel("exploit_time")
    #
    #     #make a legend:
    #     # pws = [0.5, 1, 1.5, 2., 2.5]
    #     # for pw in pws:
    #     #     plt.scatter([], [], s=(pw**2), c="k",label=str(pw))
    #     #
    #     # h, l = plt.gca().get_legend_handles_labels()
    #     # plt.legend(h[1:], l[1:], labelspacing=1.2, title="num_resets", borderpad=1,
    #     #             frameon=True, framealpha=0.6, )
    #     plt.xlim(0, min(exploreExploit_filtered1['explore_time'].max()+10,100))
    #     plt.ylim(0, min(exploreExploit_filtered1['exploit_time'].max()+10,100))
    #     # plt.xlim(0,60)
    #     # plt.ylim(0,60)
    #     # title = p + '_' +board
    #     title = p + '_full'
    #     plt.title(title)
    #     # plt.show()
    #     # plt.figure(figsize=(20,10))
    #     # ax.set(yscale="symlog")
    #     # ax.set(xscale="symlog")
    #     plt.show()
        # plt.savefig("dynamics/explore_exploit/test/15_explore_exploit_allBoards60_"+ title +".png", format='png')
        # plt.clf()
        # c = Chart(exploreExploit_filtered1)
        # c.mark_circle().encode(
        #     x='explore_time',
        #     y='exploit_time',
        #     color='solved',
        #     size='num_resets',
        # )
        # c.serve()
        # break
        # display(c)
        # print(c.to_json(indent=2))
        # plt.show()
        # exploreExploit_filtered1 = user_stats_exploration.loc[(user_stats_exploration['solved']=='validatedCorrect') & (user_stats_exploration['board']==board)]

        # print stats.spearmanr(exploreExploit_filtered1['explore_time'], exploreExploit_filtered1['exploit_time'])
        # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
        # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, color='green')
        # # plt.gca().set_xlim(left=0)
        # # plt.gca().set_ylim(left=0)
        # plt.xlim(0, 130)
        # plt.ylim(0, 180)
        #
        # exploreExploit_filtered2 = user_stats_exploration.loc[((user_stats_exploration['solved']=='wrong') | (user_stats_exploration['solved']=='solvedCorrect')) & (user_stats_exploration['board']==board)]
        # print stats.spearmanr(exploreExploit_filtered2['explore_time'], exploreExploit_filtered2['exploit_time'])
        # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
        # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered2, n_boot=1000, color='red')
        # plt.xlim(0, 130)
        # plt.ylim(0, 180)
        # # plt.gca().set_xlim(left=0)
        # # plt.gca().set_ylim(left=0)
        # plt.show()

    # print len(users)
    # for user in users:
    #     print user
    # for board in boards:
    #     print board
    #     f, (ax1, ax2) = plt.subplots(2)
    #     exploreExploit_filtered = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100)   & (exploreExploit['board']==board)]
    #     print stats.spearmanr(exploreExploit_filtered['explore_time'], exploreExploit_filtered['exploit_time'])
    #     spear = stats.spearmanr(exploreExploit_filtered['explore_time'], exploreExploit_filtered['exploit_time'])
    #     # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    #     sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered, n_boot=1000, color='blue', ax=ax1)
    #     ax1.set_xlim(0,100)
    #     ax1.set_ylim(0,100)
    #
    #     exploreExploit_filtered1 = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100) & (exploreExploit['solved']=='validatedCorrect') & (exploreExploit['board']==board)]
    #     print stats.spearmanr(exploreExploit_filtered1['explore_time'], exploreExploit_filtered1['exploit_time'])
    #     # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    #     sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, color='green', ax=ax2)
    #     plt.xlim(0,100)
    #     plt.ylim(0,100)
    #
    #     exploreExploit_filtered2 = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100)  & ((exploreExploit['solved']=='wrong') | (exploreExploit['solved']=='solvedCorrect')) & (exploreExploit['board']==board)]
    #     print stats.spearmanr(exploreExploit_filtered2['explore_time'], exploreExploit_filtered2['exploit_time'])
    #     # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    #     sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered2, n_boot=1000, color='red', ax=ax2)
    #     plt.xlim(0,100)
    #     plt.ylim(0,100)
    #
    #
    #     title = board + '_' + "spearman = " + str(round(spear.correlation,2))
    #     ax1.set_title(title)
    #     # plt.show()
    #     plt.savefig("dynamics/explore_exploit/explore_exploit_paths_"+ title +".png", format='png')

    # f, (ax1, ax2) = plt.subplots(2)
    # exploreExploit_filtered = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100)]
    # print stats.spearmanr(exploreExploit_filtered['explore_time'], exploreExploit_filtered['exploit_time'])
    # spear = stats.spearmanr(exploreExploit_filtered['explore_time'], exploreExploit_filtered['exploit_time'])
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered, n_boot=1000, color='blue', ax=ax1)
    # ax1.set_xlim(0,100)
    # ax1.set_ylim(0,100)
    #
    # exploreExploit_filtered1 = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100) & (exploreExploit['solved']=='validatedCorrect')]
    # # exploreExploit_filtered1 = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100)]
    #
    # print stats.spearmanr(exploreExploit_filtered1['explore_time'], exploreExploit_filtered1['exploit_time'])
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, color='green', ax=ax2)
    # # ax = sns.regplot(x="explore_exploit_ratio", y="solution_time", data=exploreExploit_filtered1, n_boot=1000, color='green')
    #
    # # plt.xlim(0,100)
    # # plt.ylim(0,100)
    # # plt.show()
    #
    # exploreExploit_filtered2 = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100)  & ((exploreExploit['solved']=='wrong') | (exploreExploit['solved']=='solvedCorrect'))]
    # print stats.spearmanr(exploreExploit_filtered2['explore_time'], exploreExploit_filtered2['exploit_time'])
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered2, n_boot=1000, color='red', ax=ax2)
    # plt.xlim(0,100)
    # plt.ylim(0,100)
    #
    #
    # title = 'all_boards' + '_' + "spearman = " + str(round(spear.correlation,2))
    # ax1.set_title(title)
    # # plt.show()
    # plt.savefig("dynamics/explore_exploit/explore_exploit_paths_"+ title +".png", format='png')

        # plt.show()




    # ----------o blindness - missing winning moves ---------------
    # moves = dynamics.loc[(dynamics['action'] == 'click')]
    # misses = []
    # players = []
    # boards = []
    # distances = []
    # open = []
    #
    # for index, row in moves.iterrows():
    #     if row['top_possible_score'] == 100:
    #         if row['score_move'] < 100:
    #             misses.append(1)
    #         else:
    #             misses.append(0)
    #         players.append(row['player'])
    #         boards.append(row['board_name'])
    #         distances.append(row['shutter'])
    #         open.append(row['open_path'])
    #
    # misses_data = {'board': boards, 'player':players, 'missed_win':misses, 'shutter':distances, 'open_path':open}
    # misses_data = pd.DataFrame(misses_data)
    #
    #
    # # missed_wins_x = misses_data.loc[(misses_data['player'] == 1) & (misses_data['open_path'] == True)]
    # # missed_wins_o = misses_data.loc[(misses_data['player'] == 2) & (misses_data['open_path'] == True)]
    # missed_wins_x = misses_data.loc[(misses_data['player'] == 1) ]
    # missed_wins_o = misses_data.loc[(misses_data['player'] == 2)]
    # print bootstrap_t_pvalue(missed_wins_x['missed_win'].values, missed_wins_o['missed_win'].values)
    # print bs.bootstrap(missed_wins_x['missed_win'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(missed_wins_o['missed_win'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    #
    #
    # print bootstrap_t_pvalue(missed_wins_x['shutter'].values, missed_wins_o['shutter'].values)
    # print bs.bootstrap(missed_wins_x['shutter'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(missed_wins_o['shutter'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # print bs.bootstrap(entropies_pruned, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # # figure x misses vs o misses pnas
    # # plt.subplots(2,1,figsize=(3,6))
    # plt.subplot(1,3,2)
    #
    # misses_data['player'] = misses_data['player'].map({1: 'X', 2:'O'})
    # misses_data['missed_win'] = misses_data['missed_win'].apply(lambda x: x*100)
    # ax = sns.barplot(x='player', y='missed_win', n_boot=1000, ci=68, data=misses_data)
    # ax.set_xlabel('Player', fontsize=12)
    # ax.set_ylabel('Missed Wins Percent', fontsize=12)
    # ax.tick_params(labelsize=11)
    # plt.ylim(0,100)
    # plt.tight_layout(pad=2.5)
    # plt.show()
    #
    # print '----'
    # # unmissed_wins_x = missed_wins_x.loc[missed_wins_x['missed_win'] == 0]
    # # unmissed_wins_o = missed_wins_o.loc[missed_wins_o['missed_win'] == 0]
    # # print unmissed_wins_x.shape[0]
    # # print unmissed_wins_o.shape[0]
    # # print bs.bootstrap(unmissed_wins_x['shutter'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # print bs.bootstrap(unmissed_wins_o['shutter'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # unmissed_wins_x = missed_wins_x.loc[missed_wins_x['missed_win'] == 1]
    # # unmissed_wins_o = missed_wins_o.loc[missed_wins_o['missed_win'] == 1]
    # # print unmissed_wins_x.shape[0]
    # # print unmissed_wins_o.shape[0]
    # # print bs.bootstrap(unmissed_wins_x['shutter'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # print bs.bootstrap(unmissed_wins_o['shutter'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # # misses_data.to_csv('stats/misses_data230718.csv')
    # print 1/0
    # ----------end o blindness - missing winning moves ---------------

    # ------ move entropy full vs truncated boards ------
    # # entropies = pd.read_csv("stats/entropies_data.csv")
    # # # g = sns.FacetGrid(entropies, col="sizeType", legend_out=False)
    # # ax = sns.barplot(x="condition",y="entropy", data = entropies)
    # # # print moves_s['board_name'].unique()
    # # # plt.title("tt")
    # # plt.show()
    # # states = states.loc[(states['action'] == 'click') & (states['sizeType'] == '10medium')]
    # states = dynamics.loc[(dynamics['action'] == 'click') & (dynamics['solved'] != 'validatedCorrect')]
    # # states = dynamics.loc[(dynamics['action'] == 'click')]
    # board_states = states['board_state'].unique()
    # entropies_pruned = []
    # entropies_full = []
    # states_data = []
    # boards = []
    # solved = []
    # conditions = []
    # # heuristics = []
    # entropies = []
    # for s in board_states:
    #     moves_s = states.loc[(states['board_state'] == s)]
    #     if (len(moves_s['path'].unique()) > 1):
    #         check = True
    #         for p in moves_s['path'].unique():
    #             m_p = moves_s.loc[moves_s['path'] == p]
    #             if m_p.shape[0] < 5:
    #                 check = False
    #                 break
    #         if check:
    #             # vals = moves_s['position'].unique()
    #             #
    #             # g = sns.FacetGrid(moves_s, col="condition", legend_out=False)
    #             # g.map(sns.countplot, "position", order= vals, color="steelblue", lw=0)
    #             # print moves_s['board_name'].unique()
    #             # # plt.title("tt")
    #             # plt.show()
    #
    #             pk = []
    #             moves_pruned = moves_s.loc[moves_s['condition'] == 'pruned']
    #             mp = moves_pruned['position'].unique()
    #             total = moves_pruned.shape[0] + 0.0
    #             for m in mp:
    #                 count = moves_pruned[moves_pruned['position'] == m].shape[0]
    #                 pk.append(float(count)/float(total))
    #             ent = stats.entropy(pk)
    #             entropies_pruned.append(ent)
    #             states_data.append(s)
    #             boards.append(moves_pruned['board_name'].unique()[0])
    #             solvers = moves_pruned[moves_pruned['solved'] != 'validatedCorrect'].shape[0]
    #             solved.append(float(solvers)/total)
    #             conditions.append('pruned')
    #             # heuristics.append()
    #             entropies.append(float(ent))
    #             pk = []
    #
    #             moves_full = moves_s.loc[moves_s['condition'] == 'full']
    #             mf = moves_full['position'].unique()
    #             total = moves_full.shape[0] + 0.0
    #             for m in mf:
    #                 count = moves_full[moves_full['position'] == m].shape[0]
    #                 pk.append(float(count)/float(total))
    #             ent = stats.entropy(pk)
    #             entropies_full.append(ent)
    #             states_data.append(s)
    #             solvers = moves_full[moves_full['solved'] == 'validatedCorrect'].shape[0]
    #             solved.append(float(solvers)/total)
    #             boards.append(moves_full['board_name'].unique()[0])
    #             conditions.append('full')
    #             entropies.append(float(ent))
    # entropies_data = {'board': boards, 'state': states_data, 'entropy':entropies, 'solvers':solved, 'condition':conditions}
    # entropies_data = pd.DataFrame(entropies_data)
    # entropies_data['condition'] = entropies_data['condition'].map({'full': 'full', 'pruned': 'truncated'})
    # # entropies_data.to_csv('stats/entropies_data_082118.csv')
    # # plt.subplot(2,1,1)
    # plt.figure(figsize=(5,3))
    # ax = sns.barplot(x='condition', y='entropy',  n_boot=1000, ci=68, data=entropies_data)
    # # ax = sns.barplot(x='condition', y='solvers', n_boot=1000, ci=68, data=entropies_data)
    # ax.set_xlabel('Condition', fontsize=14)
    # ax.set_ylabel('Entropy (non-solvers)', fontsize=14)
    #
    # ax.tick_params(labelsize=12)
    # # plt.title("tt")
    # # plt.tight_layout()
    # # plt.show()
    #
    # #
    # # # print entropies_data
    # # print np.mean(entropies_full)
    # # print np.mean(entropies_pruned)
    # # print np.mean(entropies_pruned)
    # #
    # # print np.std(entropies_full)
    # # print np.std(entropies_pruned)
    # #
    # entropies_full = np.asarray(entropies_full)
    # # entropies_full = entropies_data.loc[entropies_data['condition']=='full']
    # # entropies_pruned = entropies_data.loc[entropies_data['condition']=='truncated']
    # entropies_pruned = np.asarray(entropies_pruned)
    # # print bootstrap_t_pvalue(entropies_pruned['entropy'].values, entropies_full['entropy'].values)
    # print bs.bootstrap(entropies_full, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(entropies_pruned, stat_func=bs_stats.mean, is_pivotal=False)
    # print stats.mannwhitneyu(entropies_full, entropies_pruned)
    # print bootstrap_t_pvalue(entropies_full, entropies_pruned)
    # exit()
    # print 1/0
    #--------- reset and undo distributions-----------

    # ax = sns.distplot(timeResets['time_before_sec'])
    # resetsDelta = resetsDelta.loc[resetsDelta['action'] == 'reset']
    # resetsDelta = pd.read_csv('stats/resetsPotential.csv')
    # delta_filtered = dynamics.loc[(dynamics['score_curr_move'] > -100) & (dynamics['score_curr_move'] < 100)]
    # delta_filtered = dynamics.loc[(dynamics['action'] == 'reset')]
    # delta_filtered = dynamics.loc[dynamics['move_number_in_path']==5]
    # # delta_filtered = dynamics.loc[(dynamics['move_number_in_path']>4) & (dynamics['board_name']=='6_hard_full')]
    # # delta_filtered = dynamics.loc[(dynamics['delta_score']>-100) & (dynamics['delta_score']<100) & (dynamics['board_name']=='6_hard_full')]
    # # delta_filtered = dynamics.loc[(dynamics['move_number_in_path']<11) &(dynamics['score_move']>-100) & (dynamics['score_move']<100) ]
    # # delta_filtered = dynamics.loc[ (dynamics['board_name']=='6_hard_full')]
    # #
    # # ax = sns.distplot(delta_filtered['potential_score'])
    #
    # g = sns.FacetGrid(delta_filtered, row="action", legend_out=False)
    # g = g.map(sns.distplot, "score_move_x")
    # bins = np.linspace(-20,20,num=100)
    # g.map(plt.hist, "score_move", color="steelblue",  bins=bins,lw=0)
    # g.map(plt.hist, "score_move", color="steelblue",  lw=0)
    # g.map(sns.regplot,"move_number_in_path", "delta_score");
    # # g.map(plt.hist, "deltaScoreByScore", color="steelblue",  lw=0)
    # ax = sns.regplot(x="move_number_in_path",y= "potential_score", data=delta_filtered)
    # ax = g.ax_joint
    # ax.set_yscale('symlog')
    # # g.set(yscale="symlog")
    # plt.show()
    # timeUndos_filtered = timeUndos.loc[(timeUndos['time_before_sec'] < 10)]
    # timeResets_filtered = timeResets.loc[(timeResets['time_before_sec'] < 10)]
    # # ax = sns.distplot(timeUndos_filtered['time_before_sec'])
    # ax = sns.distplot(timeResets_filtered['time_before_sec'])
    # plt.show()

    # reset events
    #
    # for board in boards:
    #     moves_to_win = 0
    #     if '6' in board:
    #         moves_to_win = 4
    #         if 'pruned' in board:
    #             moves_to_win = 3
    #
    #     elif '10' in board:
    #         moves_to_win = 5
    #         if 'pruned' in board:
    #             moves_to_win = 4
    #
    #     print board
    #     # resetsData_filtered = resetsData.loc[(resetsData['board_name'] == board) & (resetsData['delta_score'] != 99999) & (resetsData['delta_score'] < 1000) & (resetsData['delta_score'] > -1000)]
    #     # resetsData_filtered = resetsData.loc[(resetsData['board_name'] == board)]
    #     resetsData_filtered = resetsData.loc[(resetsData['board_name'] == board) & (resetsData['delta_score'] != 99999) & (resetsData['delta_score'] < 1000) & (resetsData['delta_score'] > -1000) & (resetsData['move_number_in_path']<moves_to_win)]
    #
    #     # print len(resetsData_filtered)
    #     # ax = sns.distplot(resetsData_filtered['move_number_in_path'])
    #     ax = sns.distplot(resetsData_filtered['delta_score'])
    #     # timeUndos_filtered = timeUndos.loc[(timeUndos['time_before_sec'] < 10)]
    #     # timeResets_filtered = timeResets.loc[(timeResets['time_before_sec'] < 10)]
    #     # # ax = sns.distplot(timeUndos_filtered['time_before_sec'])
    #     # ax = sns.distplot(timeResets_filtered['time_before_sec'])
    #     plt.show()

    # dynamics_filtered = dynamics.loc[(dynamics['move_number_in_path'] < 11) & (dynamics['move_number_in_path'] > 1) & (dynamics['player'] == 2)]
    # userids = dynamics['userid'].unique()
    #
    # for user in userids:
    #     # print user
    #     f, (ax1, ax2) = plt.subplots(2, figsize = (20,10))
    #     clicks_filtered = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='click')]
    #     clicks_filtered_p1 = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='click') & (dynamics['player']==1)]
    #     clicks_filtered_p2 = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='click') & (dynamics['player']==2)]
    #
    #     # ax = sns.FacetGrid(dynamics_filtered, row="userid")
    #     # ax = ax.map_dataframe(sns.tsplot, time='time_rel_sec', value='time_between', unit='userid', data=clicks_filtered, interpolate=False)
    #     if (len(clicks_filtered_p1) < 2) | (len(clicks_filtered_p2) < 2):
    #         continue
    #     sns.tsplot(time='time_rel_sec', value='time_from_action', unit='userid', data=clicks_filtered, interpolate=False, ax=ax1)
    #     sns.tsplot(time='time_rel_sec', value='score_heuristic_x', unit='userid', data=clicks_filtered, interpolate=False, color='blue', ax=ax2)
    #     # sns.tsplot(time='time_rel_sec', value='top_possible_score', unit='userid', data=clicks_filtered_p1, interpolate=False, color='orange',  ax=ax2)
    #
    #     # sns.tsplot(time='time_rel_sec', value='score_move', unit='userid', data=clicks_filtered_p2, interpolate=False, color='black', ax=ax2)
    #     # sns.tsplot(time='time_rel_sec', value='top_possible_score', unit='userid', data=clicks_filtered_p2, interpolate=False, color='orange',  ax=ax2)
    #
    #     solved = clicks_filtered['solved'].iloc[0]
    #     board_name = clicks_filtered['board_name'].iloc[0]
    #     resets = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='reset')]
    #
    #     for index, event in resets.iterrows():
    #         time_reset = int(event['time_rel_sec'])
    #         # print time_reset
    #         ax1.axvline(time_reset, color="red", linestyle="--");
    #         ax2.axvline(time_reset, color="red", linestyle="--");
    #         # ax3.axvline(time_reset, color="red", linestyle="--");
    #
    #         solved = event['solved']
    #         board_name = event['board_name']
    #
    #     undos = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='undo')]
    #
    #     for index, event in undos.iterrows():
    #         time_undo = int(event['time_rel_sec'])
    #         # print time_undo
    #         ax1.axvline(time_undo, color="purple", linestyle="--");
    #         ax2.axvline(time_undo, color="purple", linestyle="--");
    #         # ax3.axvline(time_undo, color="purple", linestyle="--");
    #
    #
    #
    #     ax2.set(yscale="symlog")
    #     ax2.set_ylim(-110,110)
    #
    #     # ax3.set(yscale="symlog")
    #     # ax3.set_ylim(-100000,100000)
    #     # plt.show()
    #
    #     ax2.set_ylabel('score_heuristic_x')
    #     # ax3.set_ylabel('o score vs. best')
    #
    #     title = user + '_' + solved + '_' + board_name
    #     ax1.set_title(title)
    #     plt.show()
    #     # plt.savefig("dynamics/time_series3/timeSeries_"+ title +".png", format='png')
    #
    #     plt.clf()
    #     plt.close()

    # participant actions figure-----
    # # ax = sns.factorplot(x="board", y="actionsSolution", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 medium', '10 medium', '6 hard', '10 hard', '10 CV'],  markers=['o','^'], legend_out=False, legend=False)
    # print 'here'
    # name_mapping = {'6 MC': 'I', '6 HC': 'II', '10 MC': 'III', '10 HC': 'IV', '10 DC': 'V'}
    # print spearmanr_ci(data['timeMinutes'], data['actionsSolution'])
    #
    # plt.subplot(1,2,1)
    # data['board_name'] = data['board'].apply(lambda x: name_mapping[x])
    # ax = sns.barplot(x="board_name", y="timeMinutes", hue="condition", data=data, ci=68, n_boot=1000, order=['I','II','III','IV','V'])
    # # ax.set(xlabel='Board', ylabel='Solution Time')
    # ax.set_xlabel('Board', fontsize=14)
    # ax.set_ylabel('Solution Time (min.)', fontsize=14)
    # # lw = ax.ax.lines[0].get_linewidth()
    # # plt.setp(ax.ax.lines,linewidth=lw)
    # plt.legend(loc='best')
    # plt.subplot(1,2,2)
    # ax = sns.barplot(x="board_name", y="actionsSolution", hue="condition", data=data, ci=68, n_boot=1000, order=['I','II','III','IV','V'])
    # # ax.set(xlabel='Board', ylabel='Solution Time')
    # ax.set_xlabel('Board', fontsize=14)
    # ax.set_ylabel('Search Size', fontsize=14)
    # # lw = ax.ax.lines[0].get_linewidth()
    # # plt.setp(ax.ax.lines,linewidth=lw)
    # plt.legend(loc='best')
    # plt.tight_layout(pad=2)
    # plt.show()


    # ----- participants demographics
    # cogsci_log = pd.read_csv('logs/fullLogCogSci.csv')
    # cogsci_log = cogsci_log.loc[cogsci_log['userid'].isin(cogsci_part_list)]
    #
    # males = cogsci_log.loc[(cogsci_log['key'] == 'gender') & (cogsci_log['value'] == 'male')]
    # males = males['userid'].unique()
    # print males.shape[0]
    # females = cogsci_log.loc[(cogsci_log['key'] == 'gender') & (cogsci_log['value'] == 'female')]
    # females = females['userid'].unique()
    # print females.shape[0]
    # age = cogsci_log.loc[cogsci_log['key'] == 'age']
    # age = age.drop_duplicates(subset='userid', keep='first', inplace=False)
    # age['value'] = age['value'].apply(lambda x: int(x))
    # print np.mean(age['value'].values)
    # print np.std(age['value'].values)
    # exit()
    # ----- participants demographic end
    # ---- complexity correctness & search size figures (pnas) ---------
    # alphaBetaFull['heuristic_name'] = alphaBetaFull['heuristic_name'].map({'density':'density', 'linear':  'linear','non-linear':'non-linear', 'non-linear-interaction': 'interaction','blocking':'blocking', 'participants':'participants'})
    # # alphaBetaFull['board'] = alphaBetaFull['board'].map({'6 MC full': 'MC6 full', '6 MC truncated': 'MC6 truncated','10 MC full': 'MC10 full','10 MC truncated':'MC10 truncated','6 HC full': 'HC6 full', '6 HC truncated':'HC6 truncated','10 HC full':'HC10 full','10 HC truncated':'HC10 truncated', '10 DC full': 'DC10 full','10 DC truncated':'DC10 truncated'})
    # alphaBetaBlocking = alphaBetaFull.loc[alphaBetaFull['heuristic_name'] == 'interaction']
    # # data['board'] = data['board'].map({'6 MC': 'MC6','10 MC': 'MC10','6 HC': 'HC6','10 HC': 'HC10','10 DC': 'DC10'})
    #
    # alphaBetaBehvaior = pd.merge(alphaBetaBlocking, data, on=['size','type','condition'], how='left')
    # max_moves = float(alphaBetaBehvaior['moves'].max())
    # moves_norm = []
    # for index, row in alphaBetaBehvaior.iterrows():
    #     moves_norm.append(float(row['actionsSolution'])/float(row['moves']))
    # alphaBetaBehvaior['actions_normalized'] = moves_norm
    # # alphaBetaBehvaior['moves_blocking'] = moves_norm
    # # alphaBetaBehvaior['actionsSolution'] = alphaBetaBehvaior['actionsSolution'].apply(lambda x: np.round(float(x)/alphaBetaBehvaior['moves'],5))
    # # alphaBetaBehvaior.to_csv('stats/alpha_beta_participants_normalized.csv')
    #
    # # ax = sns.regplot(x="moves", y="moves", data=alphaBetaBehvaior,  x_estimator=np.mean, ci=68, color="r", fit_reg=False)
    # # ax.set(yscale="log")
    # # ax.set(xscale="log")
    # # plt.show()
    # plt.subplots(1,2, figsize=(8.6,4))
    # plt.subplot(1, 2, 2)
    # ax = sns.regplot(x="moves", y="solutionAndValidationCorrectPercent", x_estimator=np.mean, data=alphaBetaBehvaior, color="r", fit_reg=False,  ci=68)
    # ax.set(xscale="log")
    # ax.set(ylim=(0, 100))
    # ax.set(xlim=(10, 200000))
    # ax.set_xlabel('Board Complexity', fontsize=12)
    # ax.set_ylabel('Percent Correct', fontsize=12)
    # ax.tick_params(labelsize=11)
    # # ax.set(xlabel='Board Complexity', ylabel='Percent Correct')
    # # ax2 = ax.twinx()
    # fig = plt.subplot(1, 2, 1)
    # # alphaBetaBehvaior2 = pd.read_csv('stats/alpha_beta_participants_ab.csv')
    # # pal = {'participant':"silver", 'alpha-beta':"blue"}
    # # g = sns.FacetGrid(alphaBetaBehvaior2, hue='moves_type', palette=pal, size=5);
    # # g.map(sns.regplot, "moves", "actionsSolution", ci=68,  x_estimator=np.mean, fit_reg=False)
    # # # g.map(sns.regplot, "moves", "actionsSolution", ci=None, robust=1)
    # # g.add_legend();
    # ax = sns.regplot(x="moves", y="actionsSolution", data=alphaBetaBehvaior,  x_estimator=np.mean, ci=68, color="b", fit_reg=False)
    # sns.regplot(x="moves", y="moves", data=alphaBetaBehvaior,  x_estimator=np.mean, ci=68, color="silver", fit_reg=False, ax=ax)
    # # print stats.spearmanr(alphaBetaBehvaior.actionsSolution.values, alphaBetaBehvaior.moves.values)
    # # print spearmanr_ci(alphaBetaBehvaior.actionsSolution.values, alphaBetaBehvaior.moves.values)
    # # print spearmanr_ci(alphaBetaBehvaior.solutionAndValidationCorrectPercent.values, alphaBetaBehvaior.moves.values)
    # # print spearmanr_ci(alphaBetaBehvaior.solutionAndValidationCorrectPercent.values, alphaBetaBehvaior.actionsSolution.values)
    # # ax3 = ax.twinx()
    # # g.set(xscale="log", yscale="log")
    # ax.set(xscale="log")
    # ax.set(yscale="log")
    # ax.set(xlim=(10, 200000))
    # ax.set_xlabel('Board Complexity', fontsize=12)
    # ax.set_ylabel('Search Size', fontsize=12)
    # ax.tick_params(labelsize=11)
    #
    #
    #
    # # fig.legend(labels=['Participants Search Size','Algorithmic Search Size'], handles=[])
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label='Participants Search Size',
    #                       markerfacecolor='b', markersize=9),
    #                    Line2D([0], [0], marker='o', color='w', label='Alpha-Beta Search Size',
    #                       markerfacecolor='silver', markersize=9)]
    # fig.legend(handles=legend_elements, fontsize=9)
    #
    # ax.set(xlabel='Board Complexity', ylabel='Search Size')
    # plt.tight_layout(pad=2)
    # plt.show()
    #
    #
    # # search size and correctness tests
    # correct_participants = alphaBetaBehvaior.loc[alphaBetaBehvaior['solutionAndValidationCorrect'] == 1]
    # wrong_participants = alphaBetaBehvaior.loc[alphaBetaBehvaior['solutionAndValidationCorrect'] == 0]
    # print stats.mannwhitneyu(correct_participants['actionsSolution'].values, wrong_participants['actionsSolution'].values)
    # print cohen_d(wrong_participants['actionsSolution'].values, correct_participants['actionsSolution'].values)
    # exit()
    #
    # # ax2.set(ylim=(0, 1))
    # # plt.ylim(0,100)
    #
    # print 1/0
    # aggregations = {
    #     'solutionAndValidationCorrect': 'mean',
    #     'actionsSolution':'mean'
    # }
    # alpha_beta_shutter_pareto = alpha_beta_shutter.groupby(['max_moves','noise_sig','shutter_size','board']).agg(aggregations).reset_index()
    # ---- complexity correctness & search size figures (pnas) end ---------

    # participant actions figure-----
    # alphaBetaFull['heuristic_name'] = alphaBetaFull['heuristic_name'].map({'density':'Density', 'linear':  'Linear','non-linear':'Non-linear', 'non-linear-interaction': 'Interaction','blocking':'Blocking', 'participants':'Participants'})
    # sns.set(font_scale=1.2, style='whitegrid')
    #
    # # plt.figure(figsize=(10, 3))
    # # alpha beta and participant actions figure-----
    # # ax = sns.factorplot(x="board", y="moves",, hue="heuristic_name", data=alphaBetaFull, n_boot=1000, order=['6 MC', '10 MC', '6 HC', '10 HC', '10 DC'],  markers=["1","2","3","4","8","o"], legend_out=False, legend=False)
    # # data['board'] = data['board'].map({'6 MC full': 'MC6 full','10 MC': 'MC10','6 HC': 'HC6','10 HC': 'HC10','10 DC': 'DC10'})
    # alphaBetaFull['board'] = alphaBetaFull['board'].map({'6 MC full': 'I full', '6 MC truncated': 'I truncated','10 MC full': 'III full','10 MC truncated':'III truncated','6 HC full': 'II full', '6 HC truncated':'II truncated','10 HC full':'IV full','10 HC truncated':'IV truncated', '10 DC full': 'V full','10 DC truncated':'V truncated'})
    # ax = sns.factorplot(x="board", y="moves",  scale= 0.5, data=alphaBetaFull, hue="heuristic_name", n_boot=1000, order=['I truncated', 'I full', 'II truncated', 'III truncated', 'V truncated','II full', 'IV truncated', 'III full',   'IV full', 'V full'],  markers=["<","1","2","3","4","*"],linestyles=["-","-","-","-","-", "--"], legend_out=False, legend=False)
    # ax.fig.get_axes()[0].set_yscale('log')
    # # ax.set_xlabel('Board', fontsize=14)
    # # ax.set_ylabel('Number of Moves', fontsize=14)
    # # ax.tick_params(labelsize=12)
    # # print alphaBetaFull['moves']
    # # ax.ax.show()
    # plt.ylim(0, 200000)
    # # sns.plt.xlim(0, None)
    #
    #
    # ax.set(xlabel='Board', ylabel='Number of Moves')
    # lw = ax.ax.lines[0].get_linewidth()
    # plt.setp(ax.ax.lines,linewidth=lw)
    # plt.legend(loc='best')
    # plt.show()
    # exit()
    # #heatmap distance----
    # distances['scoring'] = distances['scoring'].map({'mcts':'mcts','density':'Density', 'linear':  'Linear','non-linear':'Non-linear', 'non-linear-interaction': 'Interaction','blocking':'Forcing'})
    # distances = distances.loc[distances['scoring']!='mcts']
    #
    # plt.subplot(2,1,1)
    # ax = sns.barplot(x="scoring", y="distance", ci=68, data=distances, order=['Density', 'Linear', 'Non-linear','Interaction','Blocking'])
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    #
    # # ax = sns.factorplot(x="board", y="moves",  data=alphaBetaFull, hue="heuristic_name", n_boot=1000, order=['6 MC full', '6 MC truncated','10 MC full','10 MC truncated','6 HC full', '6 HC truncated','10 HC full','10 HC truncated', '10 DC full','10 DC truncated'],  markers=["<","1","2","3","4","*"],linestyles=["-","-","-","-","-", "--"], legend_out=False, legend=False)
    # ax.set_xlabel('', fontsize=16)
    # ax.set_ylabel('Distance', fontsize=16)
    # ax.tick_params(labelsize=14)
    # # # plt.xticks(x,tick_label,rotation = 30,fontsize=12)
    # # # ax.fig.get_axes()[0].set_yscale('log')
    # # # # print alphaBetaFull['moves']
    # # # # ax.ax.show()
    # # # plt.ylim(0, 200000)
    # # # sns.plt.xlim(0, None)
    # #
    # #
    # # # ax.set(xlabel='Board', ylabel='Distance From Participants First Moves')
    # # # lw = ax.ax.lines[0].get_linewidth()
    # # # plt.setp(ax.ax.lines,linewidth=lw)
    # # # plt.legend(loc='best')
    # # # lw = ax.ax.lines[0].get_linewidth()
    # # # plt.setp(ax.ax.lines,linewidth=lw)
    # # # # plt.legend(loc='best')
    # # # lw = ax.ax.lines[0].get_linewidth()
    # # # plt.setp(ax.ax.lines,linewidth=lw)
    # # # change_width(ax, .35)
    # plt.tight_layout()
    # plt.show()
    # exit()
    #  Wasserstein distances stats
    # mcts = distances.loc[distances['scoring'] == 'mcts']
    # density = distances.loc[distances['scoring'] == 'density']
    # linear = distances.loc[distances['scoring'] == 'linear']
    # nonlinear = distances.loc[distances['scoring'] == 'non-linear']
    # nonlinearInteraction = distances.loc[distances['scoring'] == 'non-linear-interaction']
    # blocking = distances.loc[distances['scoring'] == 'blocking']
    # density = distances.loc[distances['scoring'] == 'density']
    #
    # print bs.bootstrap(density['distance'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(linear['distance'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonlinear['distance'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonlinearInteraction['distance'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(blocking['distance'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # # print 'blocking vs mcts'
    # # print bootstrap_t_pvalue(blocking['distance'].values, mcts['distance'].values)
    # print 'blocking vs density'
    # print bootstrap_t_pvalue(blocking['distance'].values, density['distance'].values)
    # print 'blocking vs linear'
    # print bootstrap_t_pvalue(blocking['distance'].values, linear['distance'].values)
    # print 'blocking vs nonlinear'
    # print bootstrap_t_pvalue(blocking['distance'].values, nonlinear['distance'].values)
    # print 'blocking vs interaction'
    # # print stats.mannwhitneyu(blocking['distance'].values, nonlinearInteraction['distance'].values)
    # print bootstrap_t_pvalue(blocking['distance'].values, nonlinearInteraction['distance'].values)
    #
    # print 'interaction'
    # # print bootstrap_t_pvalue(nonlinearInteraction['distance'].values, mcts['distance'].values)
    # print bootstrap_t_pvalue(nonlinearInteraction['distance'].values, density['distance'].values)
    # print bootstrap_t_pvalue(nonlinearInteraction['distance'].values, linear['distance'].values)
    # print bootstrap_t_pvalue(nonlinearInteraction['distance'].values, nonlinear['distance'].values)
    #
    # print 'nonlinear'
    # # print bootstrap_t_pvalue(nonlinear['distance'].values, mcts['distance'].values)
    # print bootstrap_t_pvalue(nonlinear['distance'].values, density['distance'].values)
    # print bootstrap_t_pvalue(nonlinear['distance'].values, linear['distance'].values)
    # # #
    #
    # print 'linear'
    # # print bootstrap_t_pvalue(linear['distance'].values, mcts['distance'].values)
    # print bootstrap_t_pvalue(linear['distance'].values, density['distance'].values)
    #
    # exit()
    #
    # #heatmap distance----
    #
    # # ----participant success rate figure cogsci-----
    # # ax = sns.factorplot(x="board", y="solutionAndValidationCorrectPercent", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 MC', '10 MC', '6 HC', '10 HC', '10 DC'],  markers=['o','^'], legend_out=False, legend=False)
    # data['board'] = data['board'].map({'6 MC': 'MC6','10 MC': 'MC10','6 HC': 'HC6','10 HC': 'HC10','10 DC': 'DC10'})
    # ax = sns.barplot(x="board", y="solutionAndValidationCorrectPercent",  hue="condition", data=data, n_boot=1000, order=['MC6', 'MC10', 'HC6', 'HC10', 'DC10'])
    # #
    # # ax.set(xlabel='Board', ylabel='Percent Correct')
    # # # lw = ax.ax.lines[0].get_linewidth()
    # # # plt.setp(ax.ax.lines,linewidth=lw)
    # #
    # plt.legend(loc='best')
    # plt.show()

    # participant success rate figure end-----

    # # alpha-beta 50 moves success rate-----
    # ax = sns.factorplot(x="board", y="solutionAndValidationCorrectPercent", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 medium', '10 medium', '6 hard', '10 hard', '10 CV'],  markers=['o','^'], legend_out=False, legend=False)
    #
    # alphaBeta50['scoring'].rename_categories(['density','linear','non-linear' + '\n'+ 'interaction','non-linear','blocking'],inplace=True)
    # ax = sns.barplot(x="percent correct", y="scoring", n_boot=1000, data=alphaBeta50)
    # # ax.set(axis_labels=["a","b","c","d","e"])
    # # set_axis_labels("a","b","c","d","e")
    # # ax.set_axis_labels("a","b","c","d","e")
    # # ax.set(xlabel='Board', ylabel='Percent Correct')
    # # lw = ax.ax.lines[0].get_linewidth()
    # # plt.setp(ax.ax.lines,linewidth=lw)
    # # plt.legend(loc='best')
    # # change_width(ax, .30)
    # plt.xlim(0, 100)
    # plt.show()
    # alpha-beta 50 moves success rate-----

    # mcts num Nodes CI


    # ax.axes[0][0].legend(loc=1)
    # ax = sns.factorplot(x="size_type", y="entropyNormalized",col="condition", hue="solutionAndValidationCorrect", data=dataEntropy, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax = sns.factorplot(x="solutionAndValidationCorrect", y="actionsSolution", data=data, n_boot=1000)
    # correct = data.loc[data['solutionAndValidationCorrect'] == 1]
    # wrong = data.loc[data['solutionAndValidationCorrect'] == 0]
    # print bs.bootstrap(correct['timeMinutes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(wrong['timeMinutes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bootstrap_t_pvalue(wrong['timeMinutes'].values, correct['timeMinutes'].values)

    # print bs.bootstrap(correct['timePerMove'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(wrong['timePerMove'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bootstrap_t_pvalue(wrong['timePerMove'].values, correct['timePerMove'].values)

    # densityPop = population.loc[population['heuristic'] == 'density']
    # linearPop = population.loc[population['heuristic'] == 'linear']
    # nonLinearPop = population.loc[population['heuristic'] == 'non-linear']
    # nonLinearInteractionPop = population.loc[population['heuristic'] == 'non-linear-interaction']
    # blockingPop = population.loc[population['heuristic'] == 'blocking']
    # # print bootstrap_t_pvalue(wrong['actionsSolution'].values, correct['actionsSolution'].values)
    # print bs.bootstrap(densityPop['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(linearPop['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearPop['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearInteractionPop['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # print bs.bootstrap(blockingPop['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # print bs.bootstrap(wrong['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # correct = correct['actionsSolution']
    # wrong = wrong['actionsSolution']
    #
    #
    # #
    # easy_full_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'medium') & (data['condition'] == 'full')]
    # easy_pruned_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'medium') & (data['condition'] == 'truncated')]
    # hard_full_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'hard') & (data['condition'] == 'full')]
    # hard_pruned_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'hard') & (data['condition'] == 'truncated')]
    # easy_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'medium') & (data['condition'] == 'full')]
    # easy_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'medium') & (data['condition'] == 'truncated')]
    # hard_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'hard') & (data['condition'] == 'full')]
    # hard_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'hard') & (data['condition'] == 'truncated')]
    # medium_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'CV') & (data['condition'] == 'full')]
    # medium_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'CV') & (data['condition'] == 'truncated')]
    # #
    # easy_full = data.loc[(data['board_type'] == 'medium') & (data['condition'] == 'full')]
    # hard_full = data.loc[(data['board_type'] == 'hard') & (data['condition'] == 'full')]
    #
    # full_boards1 =  data.loc[data['condition'] == 'full']
    # # print bs.bootstrap(full_boards1['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # easy_6 = data.loc[(data['board_size'] == 6) & (data['board_type'] == 'medium')]
    # hard_6 = data.loc[(data['board_size'] == 6) & (data['board_type'] == 'hard')]
    #
    # easy_10 = data.loc[(data['board_size'] == 10) & (data['board_type'] == 'medium')]
    # hard_10 = data.loc[(data['board_size'] == 10) & (data['board_type'] == 'hard')]


    #mcts----
    #
    # mcts_10_easy_full = mctsData.loc[mctsData['board'] == '10_easy_full']
    # mcts_10_easy_pruned = mctsData.loc[mctsData['board'] == '10_easy_pruned']
    #
    # mcts_6_easy_full = mctsData.loc[mctsData['board'] == '6_easy_full']
    # mcts_6_easy_pruned = mctsData.loc[mctsData['board'] == '6_easy_pruned']
    #
    #
    # mcts_6_hard_full = mctsData.loc[mctsData['board'] == '6_hard_full']
    # mcts_6_hard_pruned = mctsData.loc[mctsData['board'] == '6_hard_pruned']
    #
    # mcts_10_hard_full = mctsData.loc[mctsData['board'] == '10_hard_full']
    # mcts_10_hard_pruned = mctsData.loc[mctsData['board'] == '10_hard_pruned']
    #
    # mcts_10_medium_full = mctsData.loc[mctsData['board'] == '10_medium_full']
    # mcts_10_medium_pruned = mctsData.loc[mctsData['board'] == '10_medium_pruned']
    #
    # print bs.bootstrap(mcts_6_easy_full['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_6_easy_pruned['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_10_easy_full['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_10_easy_pruned['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_6_hard_full['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_6_hard_pruned['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_10_hard_full['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_10_hard_pruned['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_10_medium_full['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_10_medium_pruned['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # print bs.bootstrap_ab(mcts_6_easy_full['nodes'].values,mcts_6_easy_full['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)
    # exit()
    #mcts----

    # --- full vs truncated accuracy and num of actions
    # print '6 medium full vs truncated'
    # print bs.bootstrap_ab(easy_pruned_6['solutionAndValidationCorrect'].values,easy_full_6['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_6['solutionAndValidationCorrect'].values, easy_pruned_6['solutionAndValidationCorrect'].values)
    #
    # print '10 medium full vs truncated'
    # print bs.bootstrap_ab(easy_pruned_10['solutionAndValidationCorrect'].values,easy_full_10['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_10['solutionAndValidationCorrect'].values, easy_pruned_10['solutionAndValidationCorrect'].values)
    #
    # print '6 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_6['solutionAndValidationCorrect'].values, hard_full_6['solutionAndValidationCorrect'].values,bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_6['solutionAndValidationCorrect'].values, hard_pruned_6['solutionAndValidationCorrect'].values)
    #
    # print '10 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_10['solutionAndValidationCorrect'].values,hard_full_10['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_10['solutionAndValidationCorrect'].values, hard_pruned_10['solutionAndValidationCorrect'].values)
    #
    # print bs.bootstrap(medium_full_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(medium_pruned_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print '10 CV full vs truncated'
    # print bs.bootstrap_ab( medium_pruned_10['solutionAndValidationCorrect'].values,medium_full_10['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(medium_full_10['solutionAndValidationCorrect'].values, medium_pruned_10['solutionAndValidationCorrect'].values)
    #
    #
    # print '6 medium full vs truncated'
    # print bs.bootstrap_ab( easy_pruned_6['actionsSolution'].values,easy_full_6['actionsSolution'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_6['actionsSolution'].values, easy_pruned_6['actionsSolution'].values)
    #
    # print '10 medium full vs truncated'
    # print bs.bootstrap_ab( easy_pruned_10['actionsSolution'].values,easy_full_10['actionsSolution'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_10['actionsSolution'].values, easy_pruned_10['actionsSolution'].values)
    #
    # print '6 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_6['actionsSolution'].values, hard_full_6['actionsSolution'].values,bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_6['actionsSolution'].values, hard_pruned_6['actionsSolution'].values)
    #
    # print '10 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_10['actionsSolution'].values,hard_full_10['actionsSolution'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_10['actionsSolution'].values, hard_pruned_10['actionsSolution'].values)
    #
    #
    # print bs.bootstrap(medium_full_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(medium_pruned_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print '10 CV full vs truncated'
    # print bs.bootstrap_ab( medium_pruned_10['actionsSolution'].values,medium_full_10['actionsSolution'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(medium_full_10['actionsSolution'].values, medium_pruned_10['actionsSolution'].values)


    # print '6 medium full vs truncated'
    # print bs.bootstrap_ab( easy_pruned_6['timeMinutes'].values,easy_full_6['timeMinutes'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_6['timeMinutes'].values, easy_pruned_6['timeMinutes'].values)
    #
    # print '10 medium full vs truncated'
    # print bs.bootstrap_ab( easy_pruned_10['timeMinutes'].values,easy_full_10['timeMinutes'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_10['timeMinutes'].values, easy_pruned_10['timeMinutes'].values)
    #
    # print '6 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_6['timeMinutes'].values, hard_full_6['timeMinutes'].values,bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_6['timeMinutes'].values, hard_pruned_6['timeMinutes'].values)
    #
    # print '10 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_10['timeMinutes'].values,hard_full_10['timeMinutes'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_10['timeMinutes'].values, hard_pruned_10['timeMinutes'].values)
    #
    #
    # print bs.bootstrap(medium_full_10['timeMinutes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(medium_pruned_10['timeMinutes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print '10 CV full vs truncated'
    # print bs.bootstrap_ab( medium_pruned_10['timeMinutes'].values,medium_full_10['timeMinutes'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(medium_full_10['timeMinutes'].values, medium_pruned_10['timeMinutes'].values)

    # --- full vs truncated accuracy and num of actions end

    # success rates and number of moves participants
    # print bs_compare.difference(wrong.mean(), correct.mean())
    # print bs.bootstrap_ab(wrong.as_matrix(), correct.as_matrix(), bs_stats.mean, bs_compare.difference)
    #
    # bootstrap_t_pvalue(easy_full_6['solutionAndValidationCorrect'].values, easy_pruned_6['solutionAndValidationCorrect'].values)

    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print bs.bootstrap(easy_full_6['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_6['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print bs.bootstrap(easy_pruned_6['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_pruned_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_pruned_6['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_pruned_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)

    # print bs.bootstrap_ab(easy_full_6['solutionAndValidationCorrect'].values, easy_pruned_6['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)

    # print bs.bootstrap(data['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)


    # print bs.bootstrap(hard_full['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_pruned_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_pruned_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_pruned_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_pruned_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)

    # print bs.bootstrap_ab(correct.as_matrix(), wrong.as_matrix(), bs_stats.mean, bs_compare.difference)

    # ax = sns.pointplot(x="size_type", y="actionsSolution",hue="condition", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax.show()

    # print 'boo'