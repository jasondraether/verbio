import pandas as pd
import numpy as np

import os 
import configparser
import glob 
from collections import defaultdict
import logging
from typing import List
from scipy.io import wavfile

def get_pt_data(root_dir: str, feature_pairs, pid: int, sessions: List[str]) -> (dict, dict):
	"""Used to extract a single participant's data for a list of features and sessions. Currently supports excel (.xlsx) files and wavfiles (.wav)
	For wavfiles, the data will be in the format (sampling rate, signal). For excel, it will be a Pandas dataframe
	
	Args:
		root_dir (str): Base directory of data
		features (List[str]): Features desired for extraction, choose from: E4_{2sec,5sec,10sec}, IS09_{2sec,5sec,10sec}, IS10_{2sec,5sec,10sec}, annotation
		pid (int): Participant ID, e.g. 5 for P005
		sessions (List[str]): Sessions to pull from, choose from: PRE, POST, TEST01-TEST08
		exp_type (str): Expected file type for data extraction, choose from: wav, excel
	
	Returns:
		data_table (dict): Dictionary for looking up data for feature and session, usage -> dict[(feature, session)] = data
		missing (dict): Dictionary for looking up missing sessions for a feature, usage -> dict[feature] = [sessions]
	
	"""

	features = [x[0] for x in feature_pairs]

	assert len(features) > 0
	assert len(sessions) > 0

	p_string = 'P{:03d}'.format(pid)
	data_table = defaultdict(None)
	missing = defaultdict(list)

	for feature, exp_type in feature_pairs:
		for session in sessions:

			if exp_type == 'excel':
				data_path = os.path.join(root_dir, p_string, session, feature+'.xlsx')
				if os.path.exists(data_path): data_table[(feature, session)] = pd.read_excel(data_path, engine='openpyxl')
				else: missing[feature].append(session)
			elif exp_type == 'wav':
				data_path = os.path.join(root_dir, p_string, session, feature+'.wav')
				if os.path.exists(data_path): 
					sr, signal = wavfile.read(data_path)

					n_signal = signal.shape[0]
					# Generate times for each sample and convert to Pandas dataframe 
					times = np.linspace(0.0, n_signal/sr, n_signal, dtype='float32')

					if len(signal.shape) > 1: signal = np.mean(signal, axis=1)

					df = pd.DataFrame({'Time (s)': times, 'Audio': signal}) # In case it is stereo

					df.sr = sr # ONLY exists for audio
					data_table[(feature, session)] = df
				else: missing[feature].append(session)


	return data_table, missing

def get_pts_data(root_dir: str, feature_pairs, pids: List[int], sessions: List[str]) -> (dict, dict):
	"""A wrapper for get_pt_data(...) which extracts multiple participants' data
	
	Args:
		root_dir (str): Base directory of data
		features (List[str]): Features desired for extraction, choose from: E4_{2sec,5sec,10sec}, IS09_{2sec,5sec,10sec}, IS10_{2sec,5sec,10sec}, annotation
		pids (List[int]): List of participant IDs, e.g. [1,3,5] for P001, P003, P005
		sessions (List[str]): Sessions to pull from, choose from: PRE, POST, TEST01-TEST08
		exp_type (str): Expected file type for data extraction, choose from: wav, excel
	
	Returns:
		built_data_table (dict): Dictionary for looking up participant data for feature and session, usage -> dict[pid][(feature, session)] = data
		built_missing (dict): Dictionary for looking up participant missing sessions for a feature, usage -> dict[pid][feature] = [sessions]
	"""
	# Looks hacky, but we're actually just matching the defaultdict types
	# from get_pt_data(...)
	built_data_table = defaultdict(lambda:defaultdict(None))
	built_missing = defaultdict(lambda:defaultdict(list))

	for pid in pids:
		pt_data_table, pt_missing = get_pt_data(root_dir, feature_pairs, pid, sessions)
		built_data_table[pid] = pt_data_table
		# Only add missing if the missing dict isn't empty
		if len(pt_missing) != 0: built_missing[pid] = pt_missing

	return built_data_table, built_missing
