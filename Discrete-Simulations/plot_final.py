# graph with grouped barplots with percentage y axis and grid name nested in gammma as x axis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def bar_plot():
    # bar plot with grouped barplots with percentage y axis and grid name nested in gammma as x axis
    # load data
    data_frame = pd.read_csv("results_1.txt", delimiter=";", index_col=False)
    # create a dataframe with only the columns we want
    data_frame = data_frame[['grid', 'gamma', 'average_efficiencies', 'average_cleaned', 'randomness_move']]
    all_grids = data_frame['grid'].unique()
    print(all_grids)
    randomness_move = [0.25, 0.5, 0.75]
    gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    colours = [ "blue", "green", "purple", "brown", "black", "red", "orange", "yellow", "pink", "grey"]

    for randommove in randomness_move:
        plt.figure(figsize=(10, 10))
        alpha = 0
        colour_idx = -1
        for grid in all_grids:
            efficiency_avg = []
            cleanliness_avg = []
            grid_names = []
            gamma_list = []
            colour_idx +=1
            grid_names.append(grid)
            df_plot1 = data_frame.loc[(data_frame["randomness_move"] == randommove)& (data_frame["grid"] == grid)]
            print(df_plot1)
            for i in gamma_values:
                df_plot2 = df_plot1.loc[(df_plot1["gamma"] == i) ]
                if df_plot2["average_cleaned"].empty:
                    cleanliness_avg.append(0)
                else:
                    cleanliness_avg.append(df_plot2["average_cleaned"])
                if df_plot2["average_efficiencies"].empty:
                    efficiency_avg.append(0)
                else:
                    efficiency_avg.append(df_plot2["average_efficiencies"])
                gamma_list.append(i)
            # plot the data as line graph
            print(gamma_list, efficiency_avg, grid + "_efficiency", colours[colour_idx], "0.4")
            plt.plot(gamma_list,
                     efficiency_avg,
                     label=grid + "_efficiency",
                     color=colours[colour_idx], alpha=0.4)
            plt.plot(gamma_list, cleanliness_avg, label=grid + "_cleanliness", color=colours[colour_idx], alpha= 1)
            plt.legend()

            plt.xlabel("gamma")
            plt.ylabel("percentage")
        plt.title("p_move: " + str(randommove))
        save_dir = os.path.join('plots_final', f'{randommove}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{randommove}_plot.png'))
        plt.show()



bar_plot()
# First create some toy data:
