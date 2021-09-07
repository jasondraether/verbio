from sklearn.utils import resample
import pandas as pd
import preprocessing
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

def get_pt_dfs(features_data, target_data, valid_pts_sessions, target_feature, target_function, features_to_extract, extracted_dir):
	
	pt_dfs = {}
	
	for pt, valid_sessions in valid_pts_sessions:
	    print(f'Valid sessions for Participant {pt}: {valid_sessions}')
	    pt_df_list = []
	    for session in valid_sessions:
	        session_dfs = []
	        target_df = target_data[pt][(target_feature, session)]
	        target_df = target_function(target_df)
	        session_dfs.append(target_df)
	        for raw_data_name, file_format, feature_name, extraction_fn, format_fn, write, use_existing in features_to_extract:
	            
	            p_string = 'P{:03d}'.format(pt)
	            cache_path = os.path.join(extracted_dir,p_string,session,feature_name+'.xlsx')
	            
	            if use_existing and os.path.exists(cache_path):
	                #print(f'Using cache data at {cache_path}')
	                feature_df = pd.read_excel(cache_path, engine='openpyxl')
	            else:
	                feature_df = features_data[pt][(raw_data_name, session)]
	                feature_df = extraction_fn(feature_df)
	                feature_df = feature_df.add_suffix(feature_name)
	                if write: # Nested in here (no point in writing back the file we just read!)
	                    #print(f'Writing to cache at {cache_path}')
	                    feature_df.to_excel(cache_path, engine='openpyxl')

	            # All extracted df's must be formatted!
	            feature_df = format_fn(feature_df)
	            session_dfs.append(feature_df)
	            
	        session_df = preprocessing.combine_dataframes(session_dfs)
	        pt_df_list.append(session_df)
	        
	    pt_dfs[pt] = pt_df_list

	return pt_dfs


def get_valid_pts_sessions(pts, missing_dicts, sessions, features):
	valid_pts_sessions = []
	for pt in pts:
		valid_sessions = get_valid_pt_sessions(pt, missing_dicts, sessions, features)
		if len(valid_sessions) > 0:
			valid_pts_sessions.append((pt, valid_sessions))
	return valid_pts_sessions


def get_valid_pt_sessions(pt, missing_dicts, sessions, features):
	invalid_sessions = set()
	
	for missing_dict in missing_dicts:
		for f in features:
			if f in missing_dict[pt]:
				invalid_sessions.update(missing_dict[pt][f])

	return [s for s in sessions if s not in invalid_sessions]

def eq_class_dist(df, key, targets, method='under'):
	
	key_classes = []
	
	for target in targets:
		key_class = df[df[key] == target]
		key_classes.append(key_class)

	if method == 'under': # Resample to the minority class
		min_count = np.inf 
		for d in key_classes:
			min_count = min(min_count, len(d.index))
		new_count = min_count 
	elif method == 'average': # Resample to average across classes
		total_count = 0
		for d in key_classes:
			total_count += len(d.index)
		new_count = total_count // len(key_classes)
	else:
		raise ValueError('Sampling method {0} not recognized'.format(method))

	resampled_dfs = []
	for d in key_classes:
		d_resampled = resample(d, replace=True, n_samples=new_count)
		resampled_dfs.append(d_resampled)

	df = pd.concat(resampled_dfs, axis=0)
	df = preprocessing.shuffle_dataframe(df)
	return df

def data_to_train_test(train_data, test_data, target_key):
	y_train = train_data[target_key].to_numpy()
	X_train = train_data.drop(target_key, axis=1).to_numpy()

	y_test = test_data[target_key].to_numpy()
	X_test = test_data.drop(target_key, axis=1).to_numpy()

	return X_train, y_train, X_test, y_test

def LOOCV_subject(pts, pt_dfs, target_key, targets, model, resample_method='', shuffle=True, show_confusion=True):
	
	# If you resample, a shuffling is forced, sorry.
	# Assumes that all keys in the DF besides the target_key will be used as features!
	metrics = defaultdict(list)
	for pt_removed in pts:
		test_data = pt_dfs[pt_removed]
		
		train_data = [pt_dfs[pt_included] for pt_included in pts if pt_included != pt_removed]
		train_data = pd.concat(train_data, axis=0, ignore_index=True)

		if resample_method != '': train_data = eq_class_dist(train_data, target_key, targets, method=resample_method)

		if shuffle: train_data = preprocessing.shuffle_dataframe(train_data)

		print(f'Training fold on {len(train_data.index)} samples.')

		X_train, y_train, X_test, y_test = data_to_train_test(train_data, test_data, target_key)

		fold_metrics = train_test_sk_model(X_train, y_train, X_test, y_test, model, show_confusion)

		for key, value in fold_metrics.items():
			metrics[key].append(value)

	return metrics

def LOOCV_session():
	pass # TODO: Implement

def LOOCV_sample():
	pass # TODO: Implement

def k_fold_CV(pts, pt_dfs, target_key, targets, model, n_folds=5, resample_method='', shuffle=True, show_confusion=True):

	# TODO: If n_folds == -1, do LO sample O

	metrics = defaultdict(list)

	data = [pt_dfs[pt] for pt in pts]
	data = pd.concat(data, axis=0, ignore_index=True)

	feature_keys = [key for key in data.columns if key != target_key]

	n_data = len(data.index)

	data_per_fold = n_data//n_folds

	for k in range(n_folds):
		train_data = pd.concat([data.iloc[:k*data_per_fold], data.iloc[(k+1)*data_per_fold:]], axis=0, ignore_index=True)
		test_data = data.iloc[(k*data_per_fold):(k+1)*data_per_fold]

		if resample_method != '': train_data = eq_class_dist(train_data, target_key, targets, method=resample_method)

		if shuffle: train_data = preprocessing.shuffle_dataframe(train_data)

		print(f'Training fold on {len(train_data.index)} samples.')

		X_train, y_train, X_test, y_test = data_to_train_test(train_data, test_data, target_key)

		fold_metrics = train_test_sk_model(X_train, y_train, X_test, y_test, model, show_confusion)

		for key, value in fold_metrics.items():
			metrics[key].append(value)

	return metrics


def train_test_sk_model(X_train, y_train, X_test, y_test, model, show_confusion=True):

	# Sklearn models (TODO: Fix this to be adaptable to anything)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	ret_metrics = {}

	tp = 0
	fp = 0
	tn = 0
	fn = 0

	for i in range(y_pred.shape[0]):
		if y_pred[i] == y_test[i] == 1: tp += 1
		if y_pred[i] == 1 and y_test[i] != y_pred[i]: fp += 1
		if y_pred[i] == y_test[i] == 0: tn += 1
		if y_pred[i] == 0 and y_test[i] != y_pred[i]: fn += 1

	if show_confusion:
		cm = confusion_matrix(y_test, y_pred, normalize=None,labels=[0,1])
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm, vmin=0)
		plt.title('Confusion matrix')
		fig.colorbar(cax)
		ax.set_xticklabels([''] + ['negative','positive'])
		ax.set_yticklabels([''] + ['negative','positive'])
		plt.xlabel('Predicted')
		plt.ylabel('Ground Truth')
		plt.show()

	try:
		ret_metrics['tp'] = tp
		ret_metrics['fp'] = fp
		ret_metrics['tn'] = tn
		ret_metrics['fn'] = fn
		ret_metrics['accuracy'] = (tp + tn) / (tp + fp + tn + fn)
		ret_metrics['recall'] = tp / (tp + fn)
		ret_metrics['precision'] = tp / (tp + fp)
		ret_metrics['f1'] = (2*ret_metrics['precision']*ret_metrics['recall']) / (ret_metrics['precision'] + ret_metrics['recall'])
	except ZeroDivisionError:
		print(f'Divide by zero error encountered! Returning empty metrics. Had ==> tp: {tp} | fp {fp} | tn {tn} | fn {fn}.')
		ret_metrics = {} # Clear metrics if invalid 

	return ret_metrics

def train_test_tf_model(X_train, y_train, X_test, y_test, model, train_func, test_func):
	pass # Tensorflow models