import numpy as np
import scipy.signal


class RelativePhase:

    def __init__(self, ts1, ts2, samp_f):
        self.ts1 = ts1
        self.ts2 = ts2
        self.samp_f = samp_f

    def crp(self):
        # get the displacement and velocity normalised as per Lamp et al. 2014
        ts1, ts1_vel = get_vel(self.ts1)
        ts2, ts2_vel = get_vel(self.ts2)

        # cut signal into cycles using zero_crossing
        zero_crossings = get_cycles(ts1)
        rel_phase = make_ensemble_curves(ts1, ts1_vel, ts2, ts2_vel, zx=zero_crossings)

        self.marp = round(np.abs(rel_phase).mean(), 2)
        self.dph = round(np.std(rel_phase), 2)


def get_vel(x, sampf=100):
    """Calculates the phase angle given x and sample frequency"""

    t = np.arange(0.1, len(x) + .1, 1)

    x_diff = np.diff(x)
    t_diff = np.diff(t)
    x_vel = x_diff/t_diff

    x_norm = normalise(x)
    x_vel_norm = normalise(x_vel)

    return x_norm[1:], x_vel_norm


def normalise(x):
    """Normalise ts based on Lamp et al. 2014 methods"""

    return 2*((x-(np.min(x)))/(np.max(x)-np.min(x)))-1


def get_cycles(ts):

    if ts[0] == 0:
        raise ValueError("The first value of the input signal must be non-zero, "
                         "\n\t\t\totherwise the cycles will not be cut correctly")

    zero_crossings = np.where(np.diff(np.sign(ts)))[0][1::2]  # every two zero crossings to get a full cycle

    return zero_crossings


def segment_crp(ts1, ts1_vel, ts2, ts2_vel):
    """
    Calculates the Continuous Relative Phase in degrees

    Args:
        ts1: displacement of the first time series
        ts1_vel:  velocity of the first time series
        ts2: displacement of the second time series
        ts2_vel: velocity of the second time series

    Returns:
    relative_phase: continuour relative phase of size (len(ts1/ts2...), 1)
    """
    relative_phase = (np.arctan2(ts1_vel, ts1) - np.arctan2(ts2_vel, ts2)) * 180 / np.pi
    relative_phase = correct_quartile(relative_phase)

    return relative_phase


def correct_quartile(x):
    """
    Corrects for changed quartiles when performing CRP

    Args:
        x: the crp time series

    Returns:
        x - corrected
    """

    for i in range(0, len(x)):
        if x[i] > 180:
            x[i] = x[i] - 360
    return x


def make_ensemble_curves(ts1, ts1_vel, ts2, ts2_vel, zx):
    """

    Args:
        ts1: displacement of the first time series
        ts1_vel:  velocity of the first time series
        ts2: displacement of the second time series
        ts2_vel: velocity of the second time series
        zx: zero-crossings (array-like): the index of the zero crossings in ts1

    Returns:
        relph: table which includes the continuous relative phase for each period (cycle)

    * Note: the signals are resampled at 100 to generate segments of equal length (ensemble curves)
    and proceed to marp and dph calculation
    """

    relph = np.zeros((100, len(zx)))
    step = 0

    for i in zx:
        ts1_c = scipy.signal.resample(ts1[i:i+1], 100)
        ts1v_c = scipy.signal.resample(ts1_vel[i:i+1], 100)
        ts2_c = scipy.signal.resample(ts2[i:i+1], 100)
        ts2v_c = scipy.signal.resample(ts2_vel[i:i+1], 100)


        # calculate crp for each segment
        relph[:, step] = segment_crp(ts1_c, ts1v_c, ts2_c, ts2v_c)
        step += 1

    return relph


