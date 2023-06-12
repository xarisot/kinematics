from scipy import signal, integrate
import pandas as pd
import numpy as np
import checktypes as checks


def center(data, method='first'):
    """Removes the first value, by either removing the first  value or the mean (method = 'first' or 'mean')"""

    checks.check_pd_df(data)

    if method == 'first':
        return data - data.iloc[0]
    elif method == 'mean':
        return data - data.mean()
    else:
        raise ValueError(f"method can be either 'first' or 'mean'")


def downsample(data1, data2=None, n_data_points=None):
    """Downsamples data1 to match the length of data2. Either data2 or n_data_points needs to be input"""

    if data2 is None:
        if n_data_points is None:
            raise ValueError("Both data2 and n_data_points are set to 'None'. At least one needs to be entered")
        else: 
            map(checks.check_pd_df, (data1, data2))
            length = n_data_points
            
    if data2 is not None:
        if n_data_points is not None:
            raise ValueError("Either data2 or n_data_points needs to be None")
        else:
            checks.check_pd_df(data1)
            length = len(data2)
        

    return pd.DataFrame(data=signal.resample(data1, length),
                        columns=data1.columns)


def filt(series, samp_f, convert_g, filt_type='bandpass',  order=4, cutoff=[.1, 5]):
    """
    Time series filtering

    Args:
        series: the time series to be filtered
        samp_f: type of filter (high, low, band-pass)
        filt_type: the filter type ("bandpass", "low" or "high")
        cutoff: the critical frequency(-ies).
        convert_g: Converts g to m/s^2 (default=True)
        order: the order of the filter (default=4)

    Returns:
        The filtered signal
    """

    # series = series.dropna()

    if convert_g:
        series = series * 9.80665  # convert g to m/s^2

    butter = signal.butter(order, cutoff, filt_type, fs=samp_f, output='sos')
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
