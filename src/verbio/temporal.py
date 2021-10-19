import math
import numpy as np


def shift(x, n):
    return np.roll(x, n)


def shift_df(df, n):
    return df.shift(n)


def upper_tidx(times, t):
    i = 0
    n_times = times.shape[0]
    while times[i] <= t or math.isclose(times[i], t):
        i += 1
        if i >= n_times:
            break
    return i-1


def lower_tidx(times, t):
    i = times.shape[0] - 1
    while times[i] >= t or math.isclose(times[i], t):
        i -= 1
        if i < 0: break
    return i+1


def time_slices(times, win_len, win_stride):
    """
    Get time slices for windowing. Slices are [t0, tk), so open on the right side
    :param times:
    :param win_len:
    :param win_stride:
    :return:
    """

    slices = []

    ti = times[0]
    ti_idx = 0
    tn = times[-1]

    tk = ti + win_len
    tk_idx = upper_tidx(times, tk)

    while tk <= tn or math.isclose(tk, tn):
        slices.append((ti_idx, tk_idx))

        ti += win_stride
        tk = ti + win_len

        ti_idx = lower_tidx(times, ti)
        tk_idx = upper_tidx(times, tk)

    return slices
