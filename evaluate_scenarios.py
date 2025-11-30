# evaluate_scenarios.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.cm as cm

BASE = pd.read_csv("baseline_5cars_with_emissions.csv")
INF  = pd.read_csv("informed_5cars_with_emissions.csv")

VEH = "NO"
X = "x_m"
Y = "y_m"
EMI = "PM10_total_pred"

def plot_trajectories(df, title, out):
    vehs = sorted(df[VEH].unique())
    colors = list(cm.get_cmap("tab20").colors)

    plt.figure(figsize=(10,8))
    ax = plt.gca()

    for i,vid in enumerate(vehs):
        sub = df[df[VEH] == vid]
        xs = sub[X].values
        ys = sub[Y].values
        em = sub[EMI].values
        em_norm = (em - em.min()) / (em.max() + 1e-9)

        points = np.array([xs, ys]).T.reshape(-1,1,2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)

        r,g,b = colors[i%20]
        cols = [(r*(0.3+0.7*v), g*(0.3+0.7*v), b*(0.3+0.7*v)) for v in em_norm]

        lc = LineCollection(segs, colors=cols, linewidths=2)
        ax.add_collection(lc)
        ax.text(xs[0], ys[0], f"V{vid}", fontsize=7, color=(r,g,b))

    plt.axis("equal")
    plt.title(title)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.savefig(out, dpi=300)
    plt.show()

plot_trajectories(BASE, "Baseline PM10 Emissions", "traj_baseline.png")
plot_trajectories(INF,  "Informed PM10 Emissions", "traj_informed.png")
