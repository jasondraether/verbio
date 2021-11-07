import numpy as np
import neurokit2 as nk
import pandas as pd
import scipy
import math

from verbio import utils, preprocessing, settings

def bvp_features_sample(signal, sr):
    info, _ = nk.ppg_process(signal, sr)
    hr = info['PPG_Rate'].to_numpy()
    hr_gradient = gradient(hr)
    df = {}
    df['HR'] = np.mean(hr)
    df['HR_Grad'] = np.mean(hr_gradient)
    return df


def bvp_features(signal, times, sr, win_len, win_stride):
    bvp_windows = preprocessing.window_timed(
        signal,
        times,
        win_len,
        win_stride,
        lambda x: np.array(x)
    )
    proc_windows = [bvp_features_sample(x, sr) for x in bvp_windows]
    hr_df = pd.DataFrame(
        {
            'HR': [x['HR'] for x in proc_windows],
            'HR_Grad': [x['HR_Grad'] for x in proc_windows]
        }
    )
    return hr_df


def eda_features_sample(signal, sr, filter_size):
    order = 4
    w0 = 1.5
    w0 = 2 * np.array(w0) / sr

    signal = nk.signal_sanitize(signal)
    b, a = scipy.signal.butter(N=order, Wn=w0, btype='lowpass', analog=False, output='ba')
    filtered = scipy.signal.filtfilt(b, a, signal)

    cleaned = nk.signal_smooth(filtered, method='convolution', kernel='blackman', size=filter_size)

    decomp = nk.eda_phasic(cleaned, sampling_rate=sr)

    peaks, info = nk.eda_peaks(
        decomp['EDA_Phasic'].values,
        sampling_rate=sr,
        method='biosppy',
        amplitude_min=0.1
    )
    df = {}

    df['SCL'] = np.mean(decomp['EDA_Tonic'].to_numpy())
    if math.isnan(df['SCL']): df['SCL'] = 0.0
    
    df['SCR_Onsets'] = np.sum(peaks['SCR_Onsets'].to_numpy())
    if math.isnan(df['SCR_Onsets']): df['SCR_Onsets'] = 0


    df['SCR_Peaks'] = np.sum(peaks['SCR_Peaks'].to_numpy())
    if math.isnan(df['SCR_Peaks']): df['SCR_Peaks'] = 0

    scr_amps = peaks['SCR_Amplitude'].to_numpy()
    df['SCR_Amplitude'] = np.mean(scr_amps[np.nonzero(scr_amps)]) if len(np.nonzero(scr_amps)[0]) > 0 else 0.0
    if math.isnan(df['SCR_Amplitude']): df['SCR_Amplitude'] = 0.0

    return df


def eda_features(signal, times, sr, win_len, win_stride, filter_size):
    eda_windows = preprocessing.window_timed(
        signal,
        times,
        win_len,
        win_stride,
        lambda x: np.array(x)
    )
    proc_windows = [eda_features_sample(x, sr, filter_size) for x in eda_windows]
    hr_df = pd.DataFrame(
        {
            'SCL': [x['SCL'] for x in proc_windows],
            'SCR_Onsets': [x['SCR_Onsets'] for x in proc_windows],
            'SCR_Peaks': [x['SCR_Peaks'] for x in proc_windows],
            'SCR_Amplitude': [x['SCR_Amplitude'] for x in proc_windows]
        }
    )
    return hr_df

def get_audio_features(signal, sr, win_len, win_stride):

    # Times are inferred!
    n_samples = signal.shape[0]

    # Frame length and frame skip in samples
    samples_per_frame = int(sr * frame_length)
    samples_per_skip = int(sr * frame_skip)

    # For functionals: OpenSMILE does the windowing for you
    # For LLD's: OpenSMILE does NOT window for you. It does leave windows, but those are just from the extractor

    feature_set_param = opensmile.FeatureSet.eGeMAPSv02

    #feature_level_param = opensmile.FeatureLevel.LowLevelDescriptors
    feature_level_param = opensmile.FeatureLevel.Functionals

    smile = opensmile.Smile(feature_set=feature_set_param, feature_level=feature_level_param)

    windowed_dfs = preprocessing.window_array(
        signal,
        samples_per_frame,
        samples_per_skip,
        lambda x: smile.process_signal(x, sr),
    )

    n_windows = len(windowed_dfs)  # sketchy...
    start_times = np.arange(0.0, (frame_skip * n_windows), frame_skip)
    end_times = np.arange(frame_length, (frame_skip * n_windows) + frame_length, frame_skip)

    df = pd.concat(windowed_dfs, axis=0)

    df['t0'] = start_times
    df['tn'] = end_times

    # Just to be safe..
    df = df.sort_values(by=['t0']).reset_index(drop=True)

    return df

def get_HRV_features(signal, sr, frame_length, frame_skip, times):
    """Extract HRV time-series features using BVP (PPG) or ECG data.
    Extraction is done in a similar way as ComParE16.
    # TODO: We could also just use IBI instead of finding peaks?

    Parameters
    ----------
    signal : ndarray
        Array of BVP (PPG) or ECG data
    sr : int
        Sampling rate of BVP or ECG data
    times : ndarray
        Timestamps of each BVP/ECG sample TODO: Allow this to be inferred from sr
    frame_length : float
        Windowing length for data in seconds
    frame_skip : float
           Window stride for data in seconds
    time_key : str, optional
        Optional time key to include for a time axis in the new dataframe.
        Default to 'Time (s)'. The time is assumed to start at 0 and
        is inferred from the sampling rate
    """

    # Unfortunately, we can't get good enough time series data unless
    # BVP is at least 4 seconds in duration
    assert frame_length >= 4.0 or math.isclose(frame_length, 4.0)

    time_slices = preprocessing.get_window_slices(times, frame_length, frame_skip)
    n_slices = len(time_slices)
    feature_dfs = [None for _ in range(n_slices)]

    for i in range(n_slices):
        frame = signal[time_slices[i][0]:time_slices[i][1] + 1]
        frame_clean = nk.ppg_clean(frame, sampling_rate=sr)
        info = nk.ppg_findpeaks(frame_clean, sampling_rate=sr)
        if frame_length >= 30.0 or math.isclose(frame_length,
                                                30.0):  # Minimum required window for accurate freq + nonlinear features
            feature_df = nk.hrv(info['PPG_Peaks'], sampling_rate=sr)
        else:
            feature_df = nk.hrv_time(info['PPG_Peaks'], sampling_rate=sr)
        feature_df['t0'] = [i * frame_skip]
        feature_df['tn'] = [(i * frame_skip) + frame_length]
        feature_dfs[i] = feature_df

    df = pd.concat(feature_dfs, axis=0)
    df = df.sort_values(by=['t0']).reset_index(drop=True)

    return df


def gradient(x):
    if isinstance(x, pd.DataFrame):
        return _gradient_df(x)
    elif isinstance(x, np.ndarray) or isinstance(x, list):
        return _gradient(x)
    else:
        raise TypeError(f'Unsupported type {type(x)}')

def _gradient(x):
    """
    Given an array, take the gradient of the array across the 0'th axis.

    :param x: Array of numbers
    :return: Gradient of input array as floats
    """
    return np.gradient(x, axis=0)

def _gradient_df(df):
    """
    Given a Pandas dataframe, take the gradient of every column
    in that dataframe.

    :param df: Pandas dataframe
    :return: Input dataframe with gradient of all columns each appended with '_grad'
    """
    df_keys = utils.get_df_keys(df)
    for key in df_keys:
        gradient_key = key + '_grad'
        df[gradient_key] = gradient(df[key].to_numpy())
    return df
