import numpy as np
from scipy import signal

from verbio import temporal as tp

def binarize(x, threshold):
    """
    Convert a non-binary array to an array of 0's and 1's based on a threshold using the less than operator.

    :param x: Array
    :param threshold: Threshold, anything below this becomes a 0, anything at or above becomes a 1
    :return: Binarized array
    """
    return np.where(x < threshold, 0, 1).astype(int)

def window(x, win_len, win_stride, win_fn):
    """
    Apply a window function to an array with window size win_len, advancing at win_stride.

    :param x: Array
    :param win_len: Number of samples per window
    :param win_stride: Stride to advance current window
    :param win_fn: Callable window function. Takes the slice of the array as input
    :return: List of windowed values
    """
    n = x.shape[0]
    n_win = ((n - win_len)//win_stride)+1
    x_win = [win_fn(x[i*win_stride:(i*win_stride)+win_len]) for i in range(n_win)]
    return x_win

def window_timed(x, times, win_len, win_stride, win_fn):
    """
    Apply a window function to a timed array with window duration win_len, advancing a duration of win_stride.

    :param x: Array
    :param times: Timestamps for each sample in x
    :param win_len: Duration of window
    :param win_stride: Stride to advance timestamp of current window
    :param win_fn: Callable window function. Takes the slice of the array as input
    :return: List of windowed values
    """
    slices = tp.time_slices(times, win_len, win_stride)
    n_slices = len(slices)
    x_win = [win_fn(x[slices[i][0]:slices[i][1]+1]) for i in range(n_slices)]
    return x_win

def upsample(x, rate):
    """
    Upsample an array by a factor of rate.

    :param x: Array to upsample
    :param rate: Factor to upsample by
    :return: Upsampled array
    """
    return np.repeat(x, rate, axis=0)

def upsample_df(df, rate):
    """
    Upsample a dataframe by a factor of rate

    :param df: Dataframe to upsample
    :param rate: Factor to upsample by
    :return: Upsampled dataframe
    """
    return df.sample(frac=rate)

def downsample(x, rate, method):
    """
    Downsample an array using a supported interpolation method.
    TODO: Implement

    :param x: Array
    :param rate: Downsampling factor (2 == downsample by half)
    :param method: Interpolation method to apply to downsample window
    :return: Downsampled array
    """
    raise NotImplementedError

def downsample_df(df, rate):
    """
    Downsample a dataframe, interpolation not supported.

    :param df:
    :param rate:
    :return:
    """
    pass

def lookback(x, times):
    """

    :param x:
    :param times:
    :return:
    """
    pass

def lookback_df(df, times):
    """

    :param df:
    :param times:
    :return:
    """
    pass
