from tkinter.ttk import LabeledScale
import numpy as np
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 1600
def read_content(arr):
    return arr.shape[0]
def count_element(arr,elem):
    count = 0
    temp = arr.flatten()
    total_count = 0
    for val in temp:
        count = count + np.where(val==elem)[0].shape[0]
        total_count = total_count + val.shape[0]
    return count/total_count

grid =  [r".\50x50\55",r".\50x50\60",r".\50x50\65",r".\50x50\70"]
filenames = ["50x50.55.csv","50x50.60.csv","50x50.65.csv","50x50.70.csv"]
labels = ["55% rate","60% rate","65% rate","70% rate"]
for idx,g in enumerate(grid):
    
    cascade_percentage = []
    for root, dirs, files in os.walk(g, topdown=False):
        for name in files:
            fn = os.path.join(root, name)
            single = []
            if "piles.npz" in fn:
                data = np.load(fn,allow_pickle=True)["piles"]
                for iter in data:
                    single.append(count_element(iter,0))
                cascade_percentage.append(single)
    cascade_percentage = np.array(cascade_percentage).T
    for line in range(0,10):
        y = cascade_percentage[:,line]
        plt.plot(np.arange(y.shape[0]),y,label="round: %s-"%str(line)+labels[idx])
    plt.legend()
    plt.savefig(labels[idx]+".png")
    plt.clf()
    pd.DataFrame(cascade_percentage).to_csv(filenames[idx])
