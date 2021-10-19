# Global libs 
import logging 
from datetime import datetime 

# ML/DS libs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# Local libs 
import file_parser
import feature_extraction

# TODO: Set up logfile and config file 
# log_filename = str(datetime.now().strftime('%Y-%m-%d--%H:%M:%S'))
# logging.basicConfig(filename='logs/verbio.log', level=logging.INFO)

annotation_shift = 5 # seconds (how far back we look at the data)
win_len = 5 # seconds (frame length of annotations)
win_stride = 5 # seconds (overlap length of annotations)
annotation_threshold = 3 # 0 if < threshold, 1 if >= threshold
target_annotator = 'R1'
all_sessions = ['PRE','TEST01','TEST02','TEST03','TEST04','TEST05','TEST06','TEST07','TEST08','POST']
pt_range = range(1,74,1) # P001-0073 (full: range(1,74,1))

# Sessions and features for analysis
sessions = all_sessions

data_features = ['IS10_5sec', 'E4_10sec']
data_periods = [5, 10]

target_feature = 'annotation'
target_column_name = 'R1'

# Usage:
# data[pid] = fs_dict
# fs_dict[(feature, session)] = data
data, data_missing = file_parser.get_pts_data('data/', data_features, pt_range, sessions, 'excel')
target, target_missing = file_parser.get_pts_data('data/', [target_feature], pt_range, sessions, 'excel')

# Take set difference of all pts and missing pts to get valid pts 
all_pts_set = set(pt_range)
missing_pts_data = set(data_missing.keys())
missing_pts_target = set(target_missing.keys())
valid_pts = all_pts_set - missing_pts_data - missing_pts_target
print('Valid:',sorted(valid_pts))
print('Missing data:',sorted(missing_pts_data))
print('Missing target:',sorted(missing_pts_target))

# Window annotations by 10 seconds
pt_dfs = []
for pt in pt_range:
	
	missing_pt_sessions = set()
	for s in target_missing[pt][target_feature]: missing_pt_sessions.add(s)
	
	for data_feature in data_features:
		for s in data_missing[pt][data_feature]: missing_pt_sessions.add(s)
		
	valid_pt_sessions = [s for s in all_sessions if s not in missing_pt_sessions]
	
	for session in valid_pt_sessions:
			
		target_df = target[pt][(target_feature,session)]

		full_df = feature_extraction.prepare_annotation(target_df,
													   annotation_threshold,
													   [target_annotator],
													   annotation_shift,
													   win_len,
													   win_stride,
													   'mean',
													   'start')


		for data_feature, data_period in zip(data_features, data_periods):
			data_df = data[pt][(data_feature, session)]
			data_df = feature_extraction.clean_dataframe(data_df, 'interpolate')
			if data_period > win_len:
				data_df = feature_extraction.upsample_dataframe(data_df, data_period, win_len)
			elif data_period < win_len:
				data_df = feature_extraction.downsample_dataframe(data_df, data_period, win_len, method='last')
			full_df = feature_extraction.combine_dataframes(full_df, data_df)


		
		pt_dfs.append(full_df)

# NOTE: For now, we just shuffle the entire dataset. This is fine because we aren't
# using any 'temporal awareness' just yet. In the future, it is still fine to shuffle,
# but the participant dataframes list should be shuffled, since we don't need any temporal 
# awareness for the session or participant, but the data for a given
# participant sessions should NOT be shuffled


# Concatenate data along session and participant axis
# We don't need to worry much about indices at this point (unless we shuffle later)
final_df = pd.concat(pt_dfs, axis=0, ignore_index=True)
data_column_names = set(final_df.columns) - set(target_annotator)
final_df, dropped_features, kept_features = feature_extraction.drop_correlations(final_df, 
												target_annotator, 
												list(data_column_names),
												-0.05, 0.05, 0.05)


print('DROPPED:')
for feature in dropped_features: print(feature)
print('=====')
print('KEPT:')
for feature in kept_features: print(feature)
print('=====')

false_class = final_df[final_df[target_annotator] == 0]
true_class = final_df[final_df[target_annotator] == 1]

n_false = len(false_class.index)
n_true = len(true_class.index)

min_count = min(n_true, n_false)

# Undersampling (I.e. matching length of abundance data with length of sparse data)

false_class_sampled = resample(false_class, replace=False, n_samples=min_count, random_state=123)
true_class_sampled = resample(true_class, replace=False, n_samples=min_count, random_state=123)

final_df = pd.concat([false_class_sampled, true_class_sampled], axis=0)

y = final_df[target_annotator]
X = final_df.drop(columns=target_annotator)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = RandomForestClassifier(n_estimators=30, max_depth=30, random_state=500)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print('Accuracy: ', acc, 'Samples in each class: ',min_count)

# Example on shuffling
#final_df_shuffled = final_df.sample(frac=1).reset_index(drop=True)
