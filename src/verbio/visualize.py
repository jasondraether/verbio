from typing import List
from matplotlib import pyplot as plt 
from matplotlib import animation
import numpy as np 
from pydub import AudioSegment
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Input X is of shape (N, D) where N is the number of points and D is the dimension
# Input y is of shape (N, len(classes)), one-hot vector for classes
# Input labels is of shape (len(classes)), and contains the label indexed with argmax of y
# Output a DxD matrix of graphs
# TODO: Fix to work with this
def plot_matrix(df, target_key, feature_keys, is_binary=True):
	plot_keys = feature_keys.copy()
	plot_keys.append(target_key)
	sns.pairplot(df[plot_keys], hue=target_key)
	plt.show()

def run_pt_stream(data_map: dict, features: List[str], session: str, feature_column_names: List[str], time_column_name: str, speed: float=1.0, smoothing_factor: float=0.1, title: str=''):
	"""Displays an animation of features synchronized across time. Helpful for comparing feature changes over time compared to annotations
	
	Args:
	    data_map (dict): Dictionary indexed with key (feature, session) to get Pandas dataframe
	    features (List[str]): Features as supported by the data_map
	    session (str): Target session to plot the data along (TODO: Support multiple sessions)
	    feature_column_names (List[str]): 1:1 Correspondence with features, the column label for the expected feature in the dataframe
	    time_column_name (str): Label for the time column in the dataframe
	    speed (float, optional): Rate at which to play the animation (default to 1.0)
	    smoothing_factor (float, optional): 'How smooth' to play the animation. Between [1.0,0.0). Lower will look smoother but impact performance and may be out of sync
	    title (str, optional): Title of the plot 
	"""

	n_features = len(features)

	fig = plt.figure(1)

	axes = [None]*n_features
	lines = [None]*n_features
	feature_data = [None]*n_features
	times = [None]*n_features

	max_time = 0.0

	for feature_id in range(n_features):

		feature = features[feature_id]
		feature_column_name = feature_column_names[feature_id]

		feature_df = data_map[(feature, session)]
		
		times[feature_id] = feature_df[time_column_name].to_numpy()

		max_time = max(max_time, times[feature_id][-1])

		feature_data[feature_id] = feature_df[feature_column_name].to_numpy()

		# N_ROWS x N_COLUMNS x SUB_PLOT
		# This basically just sets it so each new graph is stacked 
		plot_layout = int('{0}1{1}'.format(n_features,feature_id+1))
		axes[feature_id] = fig.add_subplot(plot_layout,xlim=(0,times[feature_id][-1]), ylim=(np.min(feature_data[feature_id]),np.max(feature_data[feature_id])))
		lines[feature_id],= axes[feature_id].plot([],[],lw=2,color='red', clip_on=False)
		axes[feature_id].set_ylabel(feature)
		axes[feature_id].grid()

	# Set xticks to new range 
	for feature_id in range(n_features): axes[feature_id].set_xticks(range(0, int(max_time), 5))

	# Only provide time axis label for bottom subplot 
	axes[-1].set_xlabel('Time (s)')

	# Only provide title for top subplot 
	axes[0].set_title(title)

	# Required for matplotlib animation
	def init():
		return tuple(lines)

	# Required for matplotlib animation
	def animate(frame_no):
		for feature_id in range(n_features):
			upper_index = 0
			n_times = times[feature_id].shape[0]
			while times[feature_id][upper_index] < frame_no*smoothing_factor:
				upper_index += 1
				if upper_index >= n_times-1: break 
			lines[feature_id].set_data(times[feature_id][:upper_index+1], feature_data[feature_id][:upper_index+1])
		return tuple(lines)

	ani = animation.FuncAnimation(fig, animate, frames=int(max_time//(smoothing_factor))+1, init_func=init, interval=int(smoothing_factor*1000//speed), blit=True, repeat=False, cache_frame_data=True)


	plt.show()

