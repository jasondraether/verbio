import pandas as pd 
from scipy import stats
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np
import preprocessing

def select_by_LASSO(df, target_key, feature_keys=[]):
	
	# scikit-learn recommends using LinearSVC, but this can be experimented on
	dropped_keys = []
	kept_keys = []

	if len(feature_keys) == 0: feature_keys = [key for key in df.columns if key != target_key]

	# Don't worry about things being out of order with feature_keys, you can trust Pandas
	X = df[feature_keys].to_numpy()
	y = df[target_key].to_numpy()

	print('Fitting linear classifier for LASSO')
	lsvc = LinearSVC(C=0.01, penalty='l1', dual=False, max_iter=100000).fit(X, y)
	selector = SelectFromModel(lsvc, prefit=True)

	supports = selector.get_support()

	for i, support in enumerate(supports):
		if support == True: kept_keys.append(feature_keys[i])
		else: dropped_keys.append(feature_keys[i])

	return dropped_keys, kept_keys

def select_by_ridge():
	pass # TODO: Implement (probably with some sort of threshold)

def drop_target_correlations(df, target_key, lower_bound, upper_bound, feature_keys=[], preserve_keys=[]):
	"""Correlates the given target data with the remaining data in a Pandas dataframe. 
	Drops those that don't meet the correlation criterion, which would be
	when the correlation value lies BETWEEN the lower_bound and upper_bound.
	
	Parameters
	----------
	df : Pandas dataframe
	    Dataframe that has the target_key column and any other desired features for the correlation
	target_key : str
	    The column we are treating as the outcome data in calculating the Spearman's test
	lower_bound : float
	    Lower bound of correlation (i.e., the feature is 'safe' if it lands below this)
	upper_bound : float
	    Upper bound of correlation (i.e., the feature is 'safe' if it lands above this)
	feature_keys : list[str], optional
	    Instead of testing against all features, only evaluate the features described by this list.
	    If this is None, then it evaluates all keys - target_key
	preserve_keys : list[str], optional
	    If a key is going to be dropped, but is in this list of keys, then it is instead not dropped.
	
	Returns
	-------
	df : Pandas dataframe
	    A copy of the original dataframe with the keys dropped that didn't meet our criterion.
	    preserve_keys are not dropped.
	dropped_keys : list[str]
		The keys that didn't meet the criterion, regardless of if they were in preserve_keys.
	kept_keys : list[str]
		The keys that did meet the criterion, regardless of if they were in preserve_keys.
	"""
	target_feature = df[target_key].to_numpy()
	dropped_keys = []
	kept_keys = []
	
	if len(feature_keys) == 0: feature_keys = [key for key in df.columns if key != target_key]

	if len(preserve_keys) > 0: preserve_keys = set(preserve_keys)
	
	for key in feature_keys:
		base_feature = df[key].to_numpy()
		correlation, _ = stats.spearmanr(base_feature, target_feature)
		if (correlation < upper_bound and correlation > lower_bound): dropped_keys.append(key)
		else: kept_keys.append(key)

	true_dropped_keys = [key for key in dropped_keys if key not in preserve_keys]
	return df.drop(columns=true_dropped_keys), dropped_keys, kept_keys

def drop_within_correlations(df, lower_bound, upper_bound, feature_keys=[]):
	"""Correlates the given target data with the remaining data in a Pandas dataframe. 
	Drops those that don't meet the correlation criterion, which would be
	when the correlation value lies OUTSIDE the lower_bound and upper_bound. 
	These bounds should be REALLY large, like -0.9 and 0.9.
	
	Parameters
	----------
	df : Pandas dataframe
	    Dataframe that has the target_key column and any other desired features for the correlation
	lower_bound : float
	    Lower bound of correlation (i.e., the feature is 'safe' if it lands below this)
	upper_bound : float
	    Upper bound of correlation (i.e., the feature is 'safe' if it lands above this)
	feature_keys : list[str], optional
	    Instead of testing against all features, only evaluate the features described by this list.
	    If this is None, then it evaluates all keys - target_key
	preserve_keys : list[str], optional
	    If a key is going to be dropped, but is in this list of keys, then it is instead not dropped.
	
	Returns
	-------
	df : Pandas dataframe
	    A copy of the original dataframe with the keys dropped that didn't meet our criterion.
	    preserve_keys are not dropped.
	dropped_keys : list[str]
		The keys that didn't meet the criterion, regardless of if they were in preserve_keys.
	kept_keys : list[str]
		The keys that did meet the criterion, regardless of if they were in preserve_keys.
	"""
	
	# TODO: Make this more sophisticated by dropping 
	# the feature with the most correlation issues (since right now we're
	# only choose the target_key, the choice should really be biased to the more problematic key)

	dropped_keys = []
	kept_keys = []
	
	if len(feature_keys) == 0: feature_keys = [key for key in df.columns]
	
	for key in feature_keys:
		base_feature = df[key].to_numpy()
		for target_key in feature_keys:
			if key != target_key:
				target_feature = df[target_key].to_numpy()
				print(key, target_key)
				display(base_feature)
				display(target_feature)
				correlation, _ = stats.spearmanr(base_feature, target_feature)
				if (correlation > upper_bound or correlation < lower_bound): dropped_keys.append(target_key)
				else: kept_keys.append(key)

	return df.drop(columns=dropped_keys), dropped_keys, kept_keys

