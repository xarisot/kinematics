import matplotlib.pyplot as plt
import numpy as np


def plot_raw(r, l, coord):
    if coord not in ['y', 'z']:
        raise ValueError("coord must be either 'y' or 'z'")

    plt.plot(r)
    plt.plot(l)
    plt.legend(['right', 'left'])
    plt.title('raw angular displacement')
    plt.xlabel('data points')
    plt.ylabel('angular displacement (degrees)')
    plt.xlim([2000, 4000])
    plt.show()


def phase_space(xdisp, xvel, ydisp, yvel):
    # Phase space plot section
    plt.plot(xdisp, xvel)
    plt.plot(ydisp, yvel)
    plt.title('Phase space')
    plt.xlabel('displacement')
    plt.ylabel('velocity')
    plt.show()


def plot_crp(relph):
    plt.plot(np.abs(relph))
    plt.title('CRP')
    plt.xlabel('% of arm sway cycle')
    plt.ylabel('φ in degrees')
    plt.ylim([-1, 360])
    plt.show()

