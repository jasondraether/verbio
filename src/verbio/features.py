"""Feature extraction code for the VerBIO project
"""
import pandas as pd
import numpy as np 
from scipy import stats
import opensmile
from scipy.io import wavfile
import preprocessing
import neurokit2 as nk
import scipy
import math

def get_df_gradient(df, feature_keys):
	"""Given a list of keys for a dataframe, takes the gradient of those features and adds it to a new
	column with '_grad' appended to the original key name.
	
	Parameters
	----------
	df : Pandas dataframe
	    Dataframe that has columns in feature_keys
	feature_keys : list[str]
	    Keys in the dataframe we want to take the gradient of
	
	Returns
	-------
	df : Pandas dataframe
		Modified Dataframe with new gradient keys
	grad_keys : list[str]
		New keys added with '_grad' appended to it
	"""
	grad_keys = []
	for key in feature_keys:
		new_key = key+'_grad'
		df[new_key] = np.gradient(df[key].to_numpy, axis=0, dtype='float64')
		grad_keys.append(new_key)
	return df, grad_keys


def format_extracted_features(df, target_keys=[], time_key='', repair_fns={}, shift_fn=None, lookback_fn=None, sampling_fn=None):
	"""Summary
	
	Parameters
	----------
	df : Pandas dataframe
	    Dataframe that holds our features, does NOT contain the outcome (i.e., only 'X', not 'y')
	target_keys : list[str], optional
	    Keep only 'target_keys' and drop the rest. If empty (or not specified), then keep all columns
	time_key : str, optional
	    If there is a time key in the dataframe that needs to be dropped, then specify it. Otherwise
	    we assume there is no time key in the dataframe
	repair_fns : list, optional
	    A dictionary of lambda functions, where the key to the function is the key 
	    in the dataframe that we repair. By default, every key is eventually repaired
	    with interpolation
	shift_fn : None, optional
	    An optional lambda function to shift the data back or forward in time
	sampling_fn : None, optional
	    An optional lambda function to upsample or downsample the data
	
	Returns
	-------
	df : Pandas dataframe
	    The prepared dataframe for training
	"""
	if len(target_keys) > 0:
		kept_keys = set()
		kept_keys.update(target_keys)
		if time_key != '': kept_keys.add(time_key)
		for key in df.columns:
			if key not in kept_keys: df.drop(columns=key, inplace=True)

	if len(repair_fns) > 0:
		for key in repair_fns.keys():
			df[key] = repair_fns[key](df[key])

	# Regardless of repair functions, every column needs to be repaired just in case		
	df = preprocessing.repair_dataframe(df, 'inter')	

	# Shift, remove time key, then resample (this is correct, see on paper)
	# TODO: Support multiple shift functions
	if shift_fn != None: df = shift_fn(df) 

	if time_key != None and time_key in df.columns: df = df.drop(columns=time_key)

	# Lookback happens here
	if lookback_fn != None: df = lookback_fn(df)

	# TODO: Support multiple sampling functions
	if sampling_fn != None: df = sampling_fn(df)

	return df

def format_annotation(df, window_size=1, stride=1, window_fn=lambda x: np.mean(x, axis=0), threshold=None, time_key='', target_keys=[]):
 	"""Prepare the annotation features to be used for training.
 	
 	Parameters
 	----------
 	df : Pandas dataframe
 	    Dataframe containing annotations of anxiety levels
 	window_size : float
 	    Length of the window in seconds to apply to the annotations 
 	stride : float
 	    Stride of the window in seconds to apply to the annotations
 	window_fn : function, optional
 	    Optional window function to be apply to the annotations. Default to mean
 	threshold : int, optional
 	    Threshold to binarize the data. If annotation < threshold, 0, otherwise 1
 	time_key : str, optional
 	    If there is a time key in the dataframe that needs to be dropped, then specify it. Otherwise
	    we assume there is no time key in the dataframe
 	target_keys : list, optional
 	    Keep only 'target_keys' and drop the rest. If empty (or not specified), then keep all columns
 	
 	Returns
 	-------
 	df : Pandas dataframe
 	    The prepared dataframe for training
 	"""
 	# TODO: Allow to combine annotators
 	if target_keys != None:
	 	kept_keys = set()
	 	kept_keys.update(target_keys)
	 	if time_key != None:
	 		kept_keys.add(time_key)
	 	for key in df.columns:
	 		if key not in kept_keys: df.drop(columns=key, inplace=True)
 	
 	df = preprocessing.repair_dataframe(df, 'inter')
 	df = preprocessing.window_dataframe(df, time_key, window_size, stride, window_fn)
 	if threshold != None: df = preprocessing.binarize_dataframe(df, threshold, target_keys)
 	
 	if time_key != '' and time_key in df.columns: df = df.drop(columns=time_key)
 	
 	return df

def get_audio_features(signal, sr, frame_length, frame_skip, feature_set='eGeMAPSv02', feature_level='LLDs'):
	"""Extract ComParE16 features using the OpenSMILE toolkit
	
	Parameters
	----------
	signal : ndarray
	    Array of signal data from audio file
	sr : int
	   	Sampling rate of audio
	frame_length : float
	    Time in seconds of window during extraction
	frame_skip : float
	    Stride in seconds of window during windowing
	times : ndarray, optional
		Used to make this broadcastable (unused since times are inferred)
	time_key : str, optional
	    Optional time key to include for a time axis in the new dataframe.
		Default to 'Time (s)'. The time is assumed to start at 0 and
		is inferred from the sampling rate

	Returns
	-------
	df : Pandas dataframe
	    Dataframe with the ComParE16 features with a time axis specified by time_key
	"""
	# Times are inferred!
	n_samples = signal.shape[0]

	# Frame length and frame skip in samples
	samples_per_frame = int(sr*frame_length)
	samples_per_skip = int(sr*frame_skip)


	# For functionals: OpenSMILE does the windowing for you
	# For LLD's: OpenSMILE does NOT window for you. It does leave windows, but those are just from the extractor

	if feature_set == 'eGeMAPSv02': feature_set_param = opensmile.FeatureSet.eGeMAPSv02
	elif feature_set == 'ComParE16': feature_set_param = opensmile.FeatureSet.ComParE_2016
	else: raise ValueError(f'Unrecognized feature_set {feature_set}') 

	if feature_level == 'LLDs': feature_level_param = opensmile.FeatureLevel.LowLevelDescriptors
	elif feature_level == 'Functionals': feature_level_param = opensmile.FeatureLevel.Functionals
	else: raise ValueError(f'Unrecognized feature_level {feature_level}')

	smile = opensmile.Smile(feature_set=feature_set_param, feature_level=feature_level_param)

	windowed_dfs = preprocessing.window_array(
		signal, 
		samples_per_frame, 
		samples_per_skip,
		lambda x: smile.process_signal(x, sr),
	)

	if feature_level == 'LLDs':
		# Since OpenSmile doesn't window for us, we just do it here by taking the mean
		for i, df in enumerate(windowed_dfs):
			df = df.reset_index(drop=True).astype('float64')
			windowed_dfs[i] = df.mean(axis=0).to_frame().T 

	n_windows = len(windowed_dfs) # sketchy...
	start_times = np.arange(0.0, (frame_skip*n_windows), frame_skip)
	end_times = np.arange(frame_length, (frame_skip*n_windows)+frame_length, frame_skip)

	df = pd.concat(windowed_dfs, axis=0)

	df['t0'] = start_times
	df['tn'] = end_times

	# Just to be safe..
	df = df.sort_values(by=['t0']).reset_index(drop=True)

	return df


def get_EDA_features(signal, sr, frame_length, frame_skip, times):
	"""Summary
	
	Parameters
	----------
	signal : ndarray
	    Array of EDA data
	sr : int
	    Sampling rate of EDA data
	times : ndarray
	    Timestamps of each EDA sample TODO: Allow this to be inferred from sr 
	frame_length : float
	    Windowing length for data in seconds
	frame_skip : float
	   	Window stride for data in seconds
	time_key : str, optional
	    Optional time key to include for a time axis in the new dataframe.
		Default to 'Time (s)'. The time is assumed to start at 0 and
		is inferred from the sampling rate
	
	Returns
	-------
	df : Pandas dataframe
	    Windowed EDA features
	"""
	# TODO: Not sure if we should window the samples, then extract
	# or extract, then window samples. My guess is it doesn't matter!
	order = 4
	w0 = 1.5 # Cutoff frequency for Butterworth (should I remove?)
	w0 = 2 * np.array(w0) / sr 

	signal = nk.signal_sanitize(signal)
	b, a = scipy.signal.butter(N=order, Wn=w0, btype='lowpass', analog=False, output='ba')
	filtered = scipy.signal.filtfilt(b, a, signal)

	signal_clean = nk.signal_smooth(filtered, method='convolution', kernel='blackman', size=48)

	signal_decomp = nk.eda_phasic(signal_clean, sampling_rate=sr)

	signal_peak, info = nk.eda_peaks(
			signal_decomp['EDA_Phasic'].values,
			sampling_rate=sr,
			method='biosppy',
			amplitude_min=0.1
		)



	# Only window nonzero amplitudes
	df = pd.DataFrame({
		'SCL': preprocessing.window_timed_array(times, signal_decomp['EDA_Tonic'].to_numpy(), frame_length, frame_skip),
		'SCR_Amplitude': preprocessing.window_timed_array(times, signal_peak['SCR_Amplitude'].to_numpy(), frame_length, frame_skip, lambda x: np.mean(x[np.nonzero(x)]) if len(np.nonzero(x)[0]) > 0 else 0),
		'SCR_Onsets': preprocessing.window_timed_array(times, signal_peak['SCR_Onsets'].to_numpy(), frame_length, frame_skip, lambda x: np.sum(x)),
		'SCR_Peaks': preprocessing.window_timed_array(times, signal_peak['SCR_Peaks'].to_numpy(), frame_length, frame_skip, lambda x: np.sum(x)),
		}) # Meh, recoverytime isn't really useful

	
	start_times = np.arange(0.0, (frame_skip*(len(df.index))), frame_skip)
	end_times = np.arange(frame_length, (frame_skip*(len(df.index)))+frame_length, frame_skip)

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
		frame = signal[time_slices[i][0]:time_slices[i][1]+1]
		frame_clean = nk.ppg_clean(frame, sampling_rate=sr)
		info = nk.ppg_findpeaks(frame_clean, sampling_rate=sr)
		if frame_length >= 30.0 or math.isclose(frame_length, 30.0): # Minimum required window for accurate freq + nonlinear features
			feature_df = nk.hrv(info['PPG_Peaks'], sampling_rate=sr) 
		else:
			feature_df = nk.hrv_time(info['PPG_Peaks'], sampling_rate=sr)
		feature_df['t0'] = [i*frame_skip]
		feature_df['tn'] = [(i*frame_skip)+frame_length]
		feature_dfs[i] = feature_df

	df = pd.concat(feature_dfs, axis=0)
	df = df.sort_values(by=['t0']).reset_index(drop=True)

	return df


