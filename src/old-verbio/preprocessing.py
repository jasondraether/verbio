import numpy as np
import pandas as pd 

from scipy import signal

import math

# Returns the index of the time if it matches exactly OR
# the index of the time RIGHT BEFORE upper_time
def get_upper_time_index(times, upper_time):
	k = 0
	n_times = times.shape[0]
	while times[k] <= upper_time or math.isclose(times[k], upper_time): 
		k += 1
		if k >= n_times: return k-1
	return k-1

# Returns the index of the time if it matches exactly OR
# the index of the time RIGHT AFTER lower_time
def get_lower_time_index(times, lower_time):
	k = times.shape[0]-1
	while times[k] >= lower_time or math.isclose(times[k], lower_time):
		k -= 1
		if k < 0: return k+1
	return k+1

def window_array(data, window_size, stride, fn=lambda x: np.mean(x, axis=0)):
	
	n_datum = data.shape[0]
	n_windows = ((n_datum-window_size)//stride)+1
	windowed_data = [None for _ in range(n_windows)]

	for i in range(n_windows):
		windowed_data[i] = fn(data[i*stride:(i*stride)+window_size])

	return windowed_data

def get_window_slices(times, duration, stride):
	
	time_slices = []
	n_times = times.shape[0]
	
	ti = times[0]
	ti_index = 0
	tn = times[-1]
	
	tk = ti + duration 
	tk_index = get_upper_time_index(times[ti_index:], tk)

	while tk <= tn or math.isclose(tk, tn):
		time_slices.append((ti_index, tk_index))

		ti += stride 
		tk = ti + duration

		ti_index = get_lower_time_index(times, ti)
		tk_index = get_upper_time_index(times, tk)

	return time_slices


def window_timed_array(times, data, duration, stride, fn=lambda x: np.mean(x, axis=0)):
	
	assert times.shape[0] == data.shape[0]

	time_slices = get_window_slices(times, duration, stride)
	n_slices = len(time_slices)

	windowed_data = [None for _ in range(n_slices)]

	for i in range(n_slices):
		windowed_data[i] = fn(data[time_slices[i][0]:time_slices[i][1]+1])

	return windowed_data


# TODO: Support multiple window functions by taking input of dict
def window_dataframe(df, time_key, duration, stride, fn=lambda x: np.mean(x, axis=0)):	
	times = df[time_key].to_numpy()

	tn = times[-1]

	time_slices = get_window_slices(times, duration, stride)
	n_slices = len(time_slices)

	start_times = np.arange(0.0, (stride*n_slices), stride)
	end_times = np.arange(duration, (stride*n_slices)+duration, stride)
	new_df_dict = {'t0': start_times, 'tn': end_times}

	assert n_slices == start_times.shape[0]

	for feature_key in df.drop(time_key, axis=1).columns:
		feature_data = df[feature_key]
		windowed_feature_data = [None for _ in range(n_slices)]
		for i in range(n_slices):
			windowed_feature_data[i] = fn(feature_data[time_slices[i][0]:time_slices[i][1]+1])

		new_df_dict[feature_key] = windowed_feature_data

	return pd.DataFrame(new_df_dict)

def binarize_array(data, threshold):
	return np.where(data < threshold, 0, 1).astype(int)

def binarize_dataframe(df, threshold, keys):
	new_data = {}
	for key in keys:
		data = df[key].to_numpy()
		new_data[key] = binarize_array(data, threshold)
	df.update(new_data)
	df = df.astype({key: int for key in keys})
	return df 

def get_shift_indices(times, delay):
	t0 = times[0]
	tn = times[-1]
	if delay > 0:
		tk = t0 + delay
		shift_index = get_upper_time_index(times, tk)
	else:
		tk = t0 - delay # If delay is negative, we cut out time t0 - (-delay)
		shift_index = get_lower_time_index(times, tk)
	return shift_index


def shift_timed_array(times, data, delay, pad=True):
	# Data at time 't' is now at time 't+delay',
	# where 'delay' can be positive (shift forward in time)
	# or negative (shift backward in time)
	# Data that goes past the boundary is deleted
	n_times = times.shape[0]
	n_data = data.shape[0]
	assert n_times == n_data 

	t0 = times[0]
	tn = times[-1]

	shift_index = get_shift_indices(times, delay)

	if delay > 0:
		if pad:
			shifted_data = np.zeros(data.shape)
			shifted_data[shift_index:] = data[0:-shift_index]
			shifted_data[:shift_index] = 0
			shifted_times = times 
		else:
			shifted_data = data[0:-shift_index]
			shifted_times = times[0:-shift_index]

	else:
		if pad:
			shifted_data = np.zeros(data.shape)
			shifted_data[:shift_index] = data[shift_index:]
			shifted_data[shift_index:] = 0
			shifted_times = times 
		else:
			shifted_data = data[shift_index:]
			shifted_times = times[shift_index]

	return (shifted_times, shifted_data)

def shift_dataframe(df, delay, pad=True):
	if pad: df = df.shift(delay, fill_value=0)
	else: df = df.shift(delay)
	return df

def shift_timed_dataframe(df, time_key, delay, pad=True):
	
	shift_index = get_shift_indices(df[time_key].to_numpy(), delay)
	
	if delay > 0:
		if pad:
			for feature_key in df.drop(time_key, axis=1).columns:
				df[feature_key] = df[feature_key].shift(shift_index, fill_value=0)
		else:
			df.loc[:, df.columns != time_key] = df[:, df.columns != time_key].shift(shift_index)
	else:
		if pad:
			for feature_key in df.drop(time_key, axis=1).columns:
				df[feature_key] = df[feature_key].shift(-shift_index, fill_value=0)
		else:
			df.loc[:, df.columns != time_key] = df[:, df.columns != time_key].shift(-shift_index)

def divide_periods(period_num, period_den):
	if not math.isclose(float(period_num % period_den), 0.0):
		raise ValueError('Period {0} is not a multiple of Period {1}'.format(period_den, period_num))
	return int(period_num/period_den)

def upsample_array(data, n_repeats):
	return np.repeat(data, n_repeats, axis=0)

def upsample_dataframe(df, old_period, new_period):
	# Basically repeat entries in dataframe
	# I.e., say we have a sample from a 10 second period
	# and want to upsample to a 1 second period
	# This just repeats the 10 second sample 10 times
	# ASSUMES the data has already been windowed into samples,
	# and the old period was known, and that the time axis
	# has been removed. This will NOT 
	# format the time axis for the dataframe

	n_repeats = divide_periods(old_period, new_period)

	return pd.DataFrame(upsample_array(df.values, n_repeats), columns=df.columns)

def downsample_array(data, n_interpolates, method):
	# Currently only supports 1D
	if method == 'mean':
		window = np.ones(n_interpolates)/n_interpolates
	elif method == 'last': 
		window = np.zeros(n_interpolates)
		window[-1] = 1
	elif method == 'first':
		window = np.zeros(n_interpolates)
		window[0] = 1
	elif method == 'mid':
		window = np.zeros(n_interpolates)
		if n_interpolates % 2 == 0:
			window[(n_interpolates//2)-1:(n_interpolates//2)+1] = 1/2
		else:
			window[(n_interpolates//2)] = 1
	elif method == 'gauss':
		window = signal.windows.gaussian(n_interpolates, 1.0)
		window /= np.sum(window)
	else:
		raise ValueError('Downsampling method {0} not recognized'.format(method))

	n_splits = data.shape[0]//n_interpolates
	n_data = data.shape[0]
	cutoff_point = n_data - (n_data%n_interpolates) # lol shady
	return np.dot(np.split(data[:cutoff_point], n_splits, axis=0),window) 

def downsample_dataframe(df, n_interpolates, method):
	new_df_dict = {}

	for key in df.columns:
		new_df_dict[key] = downsample_array(df[key].to_numpy(), n_interpolates, method)

	return pd.DataFrame(new_df_dict)

def combine_dataframes(dfs):
	min_len = np.inf
	for df in dfs:
		min_len = min(min_len, len(df.index))

	truncated_dfs = [df.reset_index(drop=True).truncate(after=min_len-1) for df in dfs]
	return pd.concat(truncated_dfs, axis=1)

def repair_series(df, policy):
	# This method should be explored in the future for better methods
	if not df.isna().values.any(): return df # If there are no NaNs, we don't need to repair

	if df.isna().all(): df.fillna(0, inplace=True)

	if policy == 'zero':
		return df.fillna(0)
	elif policy == 'mean':
		return df.fillna(df.mean())
	elif policy == 'inter':
		return df.interpolate(method='linear').ffill().bfill() # Kinda sketch
	else:
		raise ValueError('Policy {0} not recognized.'.format(policy))

def repair_dataframe(df, policy):
	# This method should be explored in the future for better methods
	if not df.isna().values.any(): return df # If there are no NaNs, we don't need to repair

	for feature_key in df.columns:
		if df[feature_key].isna().all():
			df[feature_key].fillna(0, inplace=True) # Note: This should raise a warning later on

	if policy == 'zero':
		return df.fillna(0)
	elif policy == 'mean':
		return df.fillna(df.mean())
	elif policy == 'inter':
		return df.interpolate(method='linear').ffill().bfill() # Kinda sketch
	else:
		raise ValueError('Policy {0} not recognized.'.format(policy))

def generate_lookback_dataframe(df, n_lookback, target_keys=[]):
	n_data = len(df.index)

	if len(target_keys) == 0: target_keys = df.columns

	for key in target_keys:
		for lookback_num in range(1,n_lookback+1): # So if n_lookback = 2, we add new keys key_lookback_1 and key_lookback_2
			added_key = key+'_lookback_'+str(lookback_num)
			generated_column_data = np.zeros(n_data)
			
			for i in range(lookback_num, n_data):
				generated_column_data[i] = df[key].iloc[i-lookback_num]

			df[added_key] = generated_column_data

	return df

def generate_lookback(data, n_lookback):
	n_data = data.shape[0]

	padding = np.zeros((n_lookback, *data.shape[1:]))
	padded_data = np.concatenate([padding, data])

	# TODO: This is a good technique to use! Use it for other parts of the code
	generated_data = np.zeros((n_data, n_lookback, *data.shape[1:]))

	for i in range(n_data):
		generated_data[i] = padded_data[i:i+1+n_lookback]

	return generated_data

def shuffle_dataframe(df):
	return df.sample(frac=1).reset_index(drop=True)

def normalize_dataframe(df, target_keys=[], df_mean=None, df_std=None):
	if len(target_keys) == 0: target_keys = df.columns

	if df_mean == None or df_std == None:
		df[target_keys] = df[target_keys].sub(df[target_keys].mean(axis=0), axis=1)
		df[target_keys] = df[target_keys].div(df[target_keys].std(axis=0), axis=1)
	else:
		df[target_keys] = df[target_keys].sub(df_mean[target_keys], axis=1)
		df[target_keys] = df[target_keys].div(df_std[target_keys], axis=1)


	return df
