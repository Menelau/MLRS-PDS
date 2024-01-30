import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

def plot_winloss(techniques, win, tie, loss, nc, without_tie=False):
    tie = np.array(tie)
    if without_tie:
        win += tie / 2
        loss += tie / 2
        tie = np.zeros_like(tie)
    ind = np.arange(len(techniques))
    width = 0.35

    fig, ax = plt.subplots()
    p1 = ax.bar(ind, win, width, label='Win', color='blue')
    if not without_tie:
        p2 = ax.bar(ind, tie, width, bottom=win, label='Tie', color='yellow')
    p3 = ax.bar(ind, loss, width, bottom=win + tie, label='Loss', color='red')
    # ax.axhline(nc, color='blue', linewidth=1.8)
    ax.set_ylabel('# Datasets')
    ax.set_title('Win-Tie-Loss')
    ax.set_xticks(ind)
    ax.set_xticklabels(techniques)
    ax.legend()
    autolabel(p1, ax)
    autolabel(p3, ax)

    plt.show()

techniques = ['FLT_KNORAE', 'FLT_METADES', 'FLT_KNORAU', 'FLT_DESMI', 'FLT_DESP', 'FLT_MLA', 'FLT_OLA']
win = [256, 245, 241, 256, 248, 262, 266]
loss = [31, 42, 38, 30, 35, 18, 20]
tie = [1, 1, 9, 2, 5, 8, 2]
nc = 220  # Define the non-competitive baseline value

plot_winloss(techniques, win, tie, loss, nc)
