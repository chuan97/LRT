import matplotlib.pyplot as plt


def set_rcParams(*, size, lw, fs):
    plt.rcParams["figure.figsize"] = size
    plt.rcParams["lines.linewidth"] = lw
    plt.rcParams["font.size"] = fs
