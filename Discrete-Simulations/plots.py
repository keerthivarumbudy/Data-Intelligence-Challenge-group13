# import pandas as pd

# data_frame = pd.read_csv("C:\Users\20181886\Documents\Master DSAI\Q4\Data Intelligence Challenge\Data-Intelligence-Challenge\Discrete-Simulations\text\")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

data_frame = pd.read_csv("text/results_1.txt", delimiter = ";", index_col = False)
print(data_frame)
all_grids = data_frame['grid'].unique()
print(all_grids)

randomness_move = [0.25,0.5,0.75]
for randommove in randomness_move:
    df_plot1 = data_frame.loc[(data_frame["randomness_move"] == randommove)]
    efficiency_avg = []
    efficiency_std = []
    cleanliness_avg = []
    cleanliness_std = []
    grid_names = []
    # for i in len(data_frame['grid'].unique()):
    for grid in data_frame["grid"].unique():
        grid_names.append(grid)
        df_plot1_grid = df_plot1.loc[(df_plot1["grid"] == grid)].reset_index(drop=True)
        # print(grid)
        # print(df_plot1_grid['average_efficiencies'])
        # index = df_plot1_grid.index((df_plot1_grid["randomness_move"] == randommove) & (df_plot1_grid["drain_prob"] == drainprob) & (df_plot1_grid["drain"] == drainvalue) & (df_plot1_grid["vision"] == visionvalue))
        efficiency_avg = efficiency_avg + list(df_plot1_grid['average_efficiencies'])
        efficiency_std = efficiency_std + list(df_plot1_grid['std_efficiencies'])
        cleanliness_avg = cleanliness_avg + list(df_plot1_grid['average_cleaned'])
        cleanliness_std = cleanliness_std + list(df_plot1_grid['std_cleaned'])

    ind = np.arange(len(efficiency_avg))  # the x locations for the groups
    width = 0.35  # the width of the bars
    
    grid_names = [[grid_name]*int(round((len(ind)/len(grid_names)))) for grid_name in grid_names]
    grid_names = [item for sublist in grid_names for item in sublist]
    print(grid_names)

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2,efficiency_avg , width, yerr=efficiency_std,
                    label='Efficiency')
    rects2 = ax.bar(ind + width/2, cleanliness_avg, width, yerr=cleanliness_std,
                    label='Cleanliness')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Percentage of efficiency and cleanliness of different grids with rand_move = ' + str(randommove))# + ',' + ', vision = ' + str(vision))
    ax.set_xticks(ind)
    ax.set_xticklabels(grid_names)
    ax.legend()

    autolabel(rects1, "left")
    autolabel(rects2, "right")

    fig.tight_layout()
    save_dir = os.path.join('plots', f'{randomness_move}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, f'{randomness_move}_plot.png'))
    plt.show()


# plt.errorbar(np.array(df_plot1["average_efficiencies"]), np.array(df_plot1["grid"].unique()), np.array(df_plot1["std_efficiencies"]), linestyle='None', marker='^')

# plt.show()