from scipy import signal
from scipy import integrate
import pandas as pd
import numpy as np


def filt(series, samp_f, convert_g, filt_type='bandpass',  cutoff=[.1, 5]):
    """
    Time series filtering

    Args:
        series: the time series to be filtered
        samp_f: type of filter (high, low, band-pass)
        filt_type: the filter type ("bandpass", "low" or "high")
        cutoff: the critical frequency(-ies).
        convert_g: Converts g to m/s^2 (default=True)

    Returns:
        The filtered signal
    """

    # series = series.dropna()

    if convert_g:
        series = series * 9.80665  # convert g to m/s^2

    butter = signal.butter(4, cutoff, filt_type, fs=samp_f, output='sos')
    filtered = signal.sosfilt(butter, series)

    return filtered


def calc_disp_vel(acc_signal, samp_f):
    """ integrate and double integrate acc signal to get velocity and displacement respectively"""

    time_step = 1 / samp_f
    n = len(acc_signal)
    time = np.linspace(0, n*time_step, n)

    velocity = integrate.cumtrapz(y=acc_signal, x=time)
    displacement = integrate.cumtrapz(y=velocity, x=time[:-1])
    return displacement, velocity
