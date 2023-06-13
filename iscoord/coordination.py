import numpy as np
import scipy.signal
from kinematic_plots import plot_crp, plot_raw, phase_space
import matplotlib.pyplot as plt


class RelativePhase:

    def __init__(self, samp_f=128, cycles_based_on=None):
        self.samp_f = samp_f
        self.cycles_based_on = cycles_based_on

    def crp(self, ts1, ts2, ts1_vel=None, ts2_vel=None, plots=False):
        """

        Args:
            ts1: displacement of the first time series
            ts1_vel:  velocity of the first time series
            ts2: displacement of the second time series
            ts2_vel: velocity of the second time series
            plots: Boolean (default=False). Whether to show plots

        Attributes:
            marp: Mean Absolute Relative Phase is calculated by averaging the 100 data points of the mean
        ensemble curve
            dph: Deviation Phase is calculated by averaging the standard deviations of the ensemble curve data points

        Returns:
            rel_phase: array n * 100, where n=the number of detected cycles
        """
        if ts1_vel is None and ts2_vel is None:
            ts1_norm, ts1_vel_norm = get_vel(ts1, self.samp_f)
            ts2_norm, ts2_vel_norm = get_vel(ts2, self.samp_f)
            ts1_norm, ts2_norm = ts1_norm[:-1], ts2_norm[:-1]
        else:
            ts1_norm = ts1
            ts1_vel_norm = ts1_vel[:-1]
            ts2_norm = ts2
            ts2_vel_norm = ts2_vel[:-1]

        # cut signal into cycles using zero_crossing
        if self.cycles_based_on is not None:
            zero_crossings = get_cycles(self.cycles_based_on)
        else:
            zero_crossings = get_cycles(ts1_norm)
        rel_phase = make_ensemble_curves(ts1_norm, ts1_vel_norm, ts2_norm, ts2_vel_norm, zx=zero_crossings)

        self.marp = round(np.abs(rel_phase).mean(axis=1).mean(), 2)
        self.dph = round(np.abs(rel_phase).std(axis=1).mean(), 2)

        if plots:
            plot_raw(ts1, ts2, 'z')
            phase_space(ts1_norm, ts1_vel_norm, ts2_norm, ts2_vel_norm)
            plot_crp(rel_phase)

        return rel_phase


def get_vel(x, sampf):
    """Calculates the velocity given the displacement and the sample frequency"""

    tau = 1/sampf
    time = np.arange(tau, len(x) + tau, 1)

    x_diff = np.diff(x)
    t_diff = np.diff(time)
    x_vel = x_diff/t_diff

    # x_norm = normalise(x)
    # x_vel_norm = normalise(x_vel)
    return x, x_vel


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
        * same as crp def

    Returns:
    relative_phase: continuous relative phase of size (len(ts1/ts2...), 1)
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
    x_corrected = x.copy()

    for i in range(0, len(x_corrected)):

        if x_corrected[i] < x_corrected[i-1]-300:
            x_corrected[i] += 360

        if x_corrected[i] > x_corrected[i-1]+300:
            x_corrected[i] -= 360

    return x_corrected


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
    relph = np.zeros((100, len(zx)-1))
    for i in range(0, len(zx)-1):

        ts1_c = scipy.signal.resample(ts1[zx[i]:zx[i+1]], 100)
        ts1v_c = scipy.signal.resample(ts1_vel[zx[i]:zx[i+1]], 100)
        ts2_c = scipy.signal.resample(ts2[zx[i]:zx[i+1]], 100)
        ts2v_c = scipy.signal.resample(ts2_vel[zx[i]:zx[i+1]], 100)
        
        ts1_c = normalise(ts1_c)
        ts1v_c = normalise(ts1v_c)
        ts2_c = normalise(ts2_c)
        ts2v_c = normalise(ts2v_c)
        
        # calculate crp for each segment
        relph[:, i] = segment_crp(ts1_c, ts1v_c, ts2_c, ts2v_c)
    relph = relph[:100, :]
    return relph




