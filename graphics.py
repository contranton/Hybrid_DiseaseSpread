from matplotlib import pyplot as plt
import os
import numpy as np
import pickle

from params import N, N_per_run

# Get file names
files = next(os.walk("data"))[2]

# Get run "a"
for L in "abcdefg":
    A_names = [f for f in files if f[0] == L]
    A_data = []
    if not A_names:
        continue

    for i,fname in enumerate(A_names):
        with open(f"data/{fname}", 'rb') as f:
            print(f"Reading {fname} ({i+1}/{len(A_names)})...", end="")
            A_data.append(pickle.loads(f.read()))
            print("done")

    t_i= 100000
    t_where = 0
    for i, run in enumerate(A_data):
        t_i_ =len(A_data[i]["t"])
        if t_i_ < t_i:
            t_where = i
            t_i = t_i_
    t = A_data[t_where]["t"]

    A_avg = np.zeros((N,len(t)))
    for j, run in enumerate(A_data):
        for i in range(N):
            A_avg[i] += run[f"x{i}"][:t_i]
        A_avg /= N

    #####################################
    print(f"Plotting {L}")

    s = 4
    fig, axs = plt.subplots(s,s, figsize=(10,10))
    for k in range(min(16, len(A_names))):
        if(N_per_run > 1):
            ax = axs.flatten()[k]
        else:
            ax = axs
        for i in range(N):
            try:
                ax.plot(A_data[k]["t"], A_data[k][f"x{i}"], alpha=0.4)
            except:
                break
            #ax.set_xlim((0,1))
            ax.set_ylim((0,100))
    fig.tight_layout()
    #fig.show()
    fig.savefig(f"plots/{L}.pdf", dpi=fig.dpi)