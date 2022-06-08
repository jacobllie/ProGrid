import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# grid_MC_data = pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Monte Carlo\\MCresults_GRID.csv")

grid_MC_data = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Monte Carlo\\MCresults_GRID.dat", unpack = True, skiprows = 1, delimiter = ",")
open_mean_dose = 4.98

MC_score = grid_MC_data[0]
position = grid_MC_data[1]
print(MC_score.shape, position.shape)
print(grid_MC_data)

plt.plot(position, MC_score)
plt.show()
