import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Set Seaborn theme
matplotlib.style.use('seaborn')

# data_analysis_functions.py

def read_signals(data, filename):
    with open(filename, "r") as f:
        next(f)  # Skips the first line of the file
        for line in f:
            currentline = line.split(",")
            for i in range(0,len(data)):
                data[i].append(float(currentline[i]))
    return data

def read_gains(filename):
    # Open the file for reading
    with open(filename, "r") as f:
        # Read in the data from the file
        data = []
        next(f)
        for line in f:
            row = [float(val) for val in line.split()]
            data.append(row)
    data = np.array(data).T.tolist()
    return data

def savefig(name):
    plt.savefig(name)
            
