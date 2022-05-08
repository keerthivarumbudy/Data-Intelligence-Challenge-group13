# import pandas as pd
# data_frame = pd.read_csv("C:\Users\20181886\Documents\Master DSAI\Q4\Data Intelligence Challenge\Data-Intelligence-Challenge\Discrete-Simulations\text\")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
data_frame = pd.read_csv("text/results.txt", sep = ";")
data_frame['grid'] = data_frame['grid'].astype(int)
print(data_frame)
all_grids = data_frame['grid'].unique()

#plot EFFICIENCY with randomness_move = 0, drain_ prob = 1, drain = 10, vision = 1 
df_plot1 = data_frame.loc[(data_frame["randomness_move"] == 0) & (data_frame["drain_prob"] == 1) & (data_frame["drain"] == 10) & (data_frame["vision"] == 1)]
print(df_plot1)

x = np.array([1, 2, 3, 4, 5])
# y = np.power(len(x)*x, 1) # Effectively y = x**2
y = np.array([1,2,3,4,5])
e = np.array([1.5, 2.6, 3.7, 4.6, 5.5])

plt.errorbar(x, y, e, linestyle='None', marker='^')

plt.show()


# plt.errorbar(np.array(df_plot1["average_efficiencies"]), np.array(df_plot1["grid"].unique()), np.array(df_plot1["std_efficiencies"]), linestyle='None', marker='^')

# plt.show()
 

