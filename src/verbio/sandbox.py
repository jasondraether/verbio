import reader
import visualize 
import features

import scipy
import neurokit2 as nk
import pandas as pd 
import opensmile
import numpy as np
import matplotlib.pyplot as plt 
import os
from scipy.io import wavfile 
from scipy.ndimage import median_filter

from random import shuffle


pts = [i for i in range(1, 74, 1)]
shuffle(pts)

for pt in pts:

	base_path = f'data/raw_data/P{pt:03d}/'

	hr_pre_relax_path = os.path.join(base_path, 'PRE/E4_HR_RELAX.xlsx')
	hr_pre_prep_path = os.path.join(base_path, 'PRE/E4_HR_PREP.xlsx')
	hr_post_relax_path = os.path.join(base_path, 'POST/E4_HR_RELAX.xlsx')
	hr_post_prep_path = os.path.join(base_path, 'POST/E4_HR_PREP.xlsx')

	eda_pre_relax_path = os.path.join(base_path, 'PRE/E4_EDA_RELAX.xlsx')
	eda_pre_prep_path = os.path.join(base_path, 'PRE/E4_EDA_PREP.xlsx')
	eda_post_relax_path = os.path.join(base_path, 'POST/E4_EDA_RELAX.xlsx')
	eda_post_prep_path = os.path.join(base_path, 'POST/E4_EDA_PREP.xlsx')

	# temp_pre_relax_path = os.path.join(base_path, 'PRE/E4_TEMP_RELAX.xlsx')
	# temp_pre_prep_path = os.path.join(base_path, 'PRE/E4_TEMP_PREP.xlsx')
	# temp_post_relax_path = os.path.join(base_path, 'POST/E4_TEMP_RELAX.xlsx')
	# temp_post_prep_path = os.path.join(base_path, 'POST/E4_TEMP_PREP.xlsx')

	try:

		hr_pre_relax_df = pd.read_excel(hr_pre_relax_path, engine='openpyxl')
		hr_pre_prep_df = pd.read_excel(hr_pre_prep_path, engine='openpyxl')
		hr_post_relax_df = pd.read_excel(hr_post_relax_path, engine='openpyxl')
		hr_post_prep_df = pd.read_excel(hr_post_prep_path, engine='openpyxl')

		hr_pre_relax_avg = hr_pre_relax_df['HR'].mean()
		hr_pre_prep_avg = hr_pre_prep_df['HR'].mean()
		hr_post_relax_avg = hr_post_relax_df['HR'].mean()
		hr_post_prep_avg = hr_post_prep_df['HR'].mean()

		eda_pre_relax_df = pd.read_excel(eda_pre_relax_path, engine='openpyxl')
		eda_pre_prep_df = pd.read_excel(eda_pre_prep_path, engine='openpyxl')
		eda_post_relax_df = pd.read_excel(eda_post_relax_path, engine='openpyxl')
		eda_post_prep_df = pd.read_excel(eda_post_prep_path, engine='openpyxl')

		eda_pre_relax_avg = eda_pre_relax_df['EDA'].mean()
		eda_pre_prep_avg = eda_pre_prep_df['EDA'].mean()
		eda_post_relax_avg = eda_post_relax_df['EDA'].mean()
		eda_post_prep_avg = eda_post_prep_df['EDA'].mean()

		# temp_pre_relax_df = pd.read_excel(temp_pre_relax_path, engine='openpyxl')
		# temp_pre_prep_df = pd.read_excel(temp_pre_prep_path, engine='openpyxl')
		# temp_post_relax_df = pd.read_excel(temp_post_relax_path, engine='openpyxl')
		# temp_post_prep_df = pd.read_excel(temp_post_prep_path, engine='openpyxl')

		# temp_pre_relax_avg = temp_pre_relax_df['TEMP'].mean()
		# temp_pre_prep_avg = temp_pre_prep_df['TEMP'].mean()
		# temp_post_relax_avg = temp_post_relax_df['TEMP'].mean()
		# temp_post_prep_avg = temp_post_prep_df['TEMP'].mean()


	except IOError:
		continue

	for session in ['PRE', 'POST', 'TEST01', 'TEST02', 'TEST03', 'TEST04', 'TEST05', 'TEST06', 'TEST07', 'TEST08']:

		session_path = os.path.join(base_path, session)
		hr_path = os.path.join(session_path, 'E4_HR_PPT.xlsx')
		eda_path = os.path.join(session_path, 'E4_EDA_PPT.xlsx')
		#temp_path = os.path.join(session_path, 'E4_TEMP_PPT.xlsx')
		annotation_path = os.path.join(session_path, 'annotation.xlsx')

		try:
			annotation_df = pd.read_excel(annotation_path, engine='openpyxl')

			annotations_r1 = annotation_df['R1'].to_numpy()
			annotations_r2 = annotation_df['R2'].to_numpy()
			
			annotations_r1 = median_filter(annotations_r1, size=10)
			annotations_r2 = median_filter(annotations_r2, size=10)

			annotation_times = annotation_df['Time (s)'].to_numpy()
		except IOError:
			continue

		try:

			hr_df = pd.read_excel(hr_path, engine='openpyxl')			

			hr_times = hr_df['Time (s)'].to_numpy()
			hrs = hr_df['HR'].to_numpy()

			fig = plt.figure()
			fig.clf()

			ax1 = fig.add_subplot(4,1,1)
			ax1.set_ylim([55, 115])
			ax1.grid()
			ax1.axhline(y=hr_pre_relax_avg, color='b', linestyle='-.', label=f'PRE_RELAX ({int(hr_pre_relax_avg)})')
			ax1.axhline(y=hr_pre_prep_avg, color='y', linestyle='-.', label=f'PRE_PREP ({int(hr_pre_prep_avg)})')
			ax1.axhline(y=hr_post_relax_avg, color='g', linestyle=':', label=f'POST_RELAX ({int(hr_post_relax_avg)})')
			ax1.axhline(y=hr_post_prep_avg, color='r', linestyle=':', label=f'POST_PREP ({int(hr_post_prep_avg)})')
			ax1.plot(hr_times, hrs, lw=2, color='k', label='PPT_HR', clip_on=False)
			ax1.set_ylabel('Heart Rate (HR)')
			ax1.set_title(f'Session {session}: Heart Rate Levels and Annotations for P{pt:03d}')

			plt.legend(loc='upper right', bbox_to_anchor=(1.125, 1))
			plt.locator_params(axis='x', nbins=24)
			ax1.set_yticks(np.arange(60, 120, 10))

			ax2 = fig.add_subplot(4,1,2)
			ax2.set_ylim([-1.0,1.0])
			ax2.grid()
			ax2.plot(hr_times, np.gradient(hrs), lw=2, color='k', label='HR-grad')
			ax2.set_ylabel('Gradient of Heart Rate')

			plt.legend(loc='upper right', labels=['HR-grad'], bbox_to_anchor=(1.1, 1))
			plt.locator_params(axis='x', nbins=24)

			ax3 = fig.add_subplot(4,1,3)
			ax3.set_ylim([1,5])
			ax3.set_ylabel('Annotation')
			ax3.grid()
			ax3.plot(annotation_times, annotations_r1, lw=2, color='c', label='R1')
			ax3.plot(annotation_times, annotations_r2, lw=2, linestyle='--',color='m', label='R2')
			
			plt.legend(loc='upper right', labels=['R1', 'R2'], bbox_to_anchor=(1.075, 1))
			plt.locator_params(axis='x', nbins=24)

			ax4 = fig.add_subplot(4,1,4)
			ax4.set_ylim([-1,1])
			ax4.set_xlabel('Time (s)')
			ax4.set_ylabel('Gradient of Annotation')
			ax4.grid()
			ax4.plot(annotation_times, np.gradient(annotations_r1), lw=2, color='c', label='R1-grad')
			ax4.plot(annotation_times, np.gradient(annotations_r2), lw=2, linestyle='--',color='m', label='R2-grad')

			plt.legend(loc='upper right', labels=['R1-grad','R2-grad'], bbox_to_anchor=(1.1, 1))
			plt.locator_params(axis='x', nbins=24)

			mng = plt.get_current_fig_manager()
			mng.window.showMaximized()

			plt.show()

		except IOError:
			continue

		try:
			eda_df = pd.read_excel(eda_path, engine='openpyxl')			

			eda_times = eda_df['Time (s)'].to_numpy()
			edas = eda_df['EDA'].to_numpy()


			# Filter EDA with NK
			sr = 4
			order = 4
			w0 = 1.5 # Cutoff frequency 
			w0 = 2 * np.array(w0) / sr 

			edas = nk.signal_sanitize(edas)
			b, a = scipy.signal.butter(N=order, Wn=w0, btype='lowpass', analog=False, output='ba')
			edas = scipy.signal.filtfilt(b, a, edas)
			edas = nk.signal_smooth(edas, method='convolution', kernel='blackman', size=48)
			# scipy.ndimage.median_filter for median filtering


			fig = plt.figure()
			fig.clf()

			ax1 = fig.add_subplot(4,1,1)
			#ax1.set_ylim([0, 1])
			ax1.grid()
			ax1.axhline(y=eda_pre_relax_avg, color='b', linestyle='-.', label=f'PRE_RELAX ({eda_pre_relax_avg:.2f})')
			ax1.axhline(y=eda_pre_prep_avg, color='y', linestyle='-.', label=f'PRE_PREP ({eda_pre_prep_avg:.2f})')
			ax1.axhline(y=eda_post_relax_avg, color='g', linestyle=':', label=f'POST_RELAX ({eda_post_relax_avg:.2f})')
			ax1.axhline(y=eda_post_prep_avg, color='r', linestyle=':', label=f'POST_PREP ({eda_post_prep_avg:.2f})')
			ax1.plot(eda_times, edas, lw=2, color='k', label='PPT_EDA', clip_on=False)
			ax1.set_ylabel('EDA')
			ax1.set_title(f'Session {session}: EDA Levels and Annotations for P{pt:03d}')

			plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
			plt.locator_params(axis='x', nbins=24)

			ax2 = fig.add_subplot(4,1,2)
			#ax2.set_ylim([-0.1,0.1])
			ax2.grid()
			ax2.plot(eda_times, np.gradient(edas), lw=2, color='k', label='EDA-grad')
			ax2.set_ylabel('Gradient of EDA')

			plt.legend(loc='upper right', labels=['EDA-grad'], bbox_to_anchor=(1.1, 1))
			plt.locator_params(axis='x', nbins=24)

			ax3 = fig.add_subplot(4,1,3)
			ax3.set_ylim([1,5])
			ax3.set_ylabel('Annotation')
			ax3.grid()
			ax3.plot(annotation_times, annotations_r1, lw=2, color='c', label='R1')
			ax3.plot(annotation_times, annotations_r2, lw=2, linestyle='--',color='m', label='R2')
			
			plt.legend(loc='upper right', labels=['R1', 'R2'], bbox_to_anchor=(1.075, 1))
			plt.locator_params(axis='x', nbins=24)

			ax4 = fig.add_subplot(4,1,4)
			ax4.set_ylim([-1,1])
			ax4.set_xlabel('Time (s)')
			ax4.set_ylabel('Gradient of Annotation')
			ax4.grid()
			ax4.plot(annotation_times, np.gradient(annotations_r1), lw=2, color='c', label='R1-grad')
			ax4.plot(annotation_times, np.gradient(annotations_r2), lw=2, linestyle='--',color='m', label='R2-grad')

			plt.legend(loc='upper right', labels=['R1-grad','R2-grad'], bbox_to_anchor=(1.1, 1))
			plt.locator_params(axis='x', nbins=24)

			mng = plt.get_current_fig_manager()
			mng.window.showMaximized()

			plt.show()

		except IOError:
			continue

		# try:
		# 	temp_df = pd.read_excel(temp_path, engine='openpyxl')			

		# 	temp_times = temp_df['Time (s)'].to_numpy()
		# 	temps = temp_df['TEMP'].to_numpy()

		# 	fig = plt.figure()
		# 	fig.clf()

		# 	ax1 = fig.add_subplot(4,1,1)
		# 	#ax1.set_ylim([55, 115])
		# 	ax1.grid()
		# 	ax1.axhline(y=temp_pre_relax_avg, color='b', linestyle='-.', label=f'PRE_RELAX ({temp_pre_relax_avg:.2f})')
		# 	ax1.axhline(y=temp_pre_prep_avg, color='y', linestyle='-.', label=f'PRE_PREP ({temp_pre_prep_avg:.2f})')
		# 	ax1.axhline(y=temp_post_relax_avg, color='g', linestyle=':', label=f'POST_RELAX ({temp_post_relax_avg:.2f})')
		# 	ax1.axhline(y=temp_post_prep_avg, color='r', linestyle=':', label=f'POST_PREP ({temp_post_prep_avg:.2f})')
		# 	ax1.plot(temp_times, temps, lw=2, color='k', label='PPT_TEMP', clip_on=False)
		# 	ax1.set_ylabel('Temperature (TEMP)')
		# 	ax1.set_title(f'Session {session}: Temperature Levels and Annotations for P{pt:03d}')

		# 	plt.legend(loc='upper right', bbox_to_anchor=(1.125, 1))
		# 	plt.locator_params(axis='x', nbins=24)
		# 	#ax1.set_yticks(np.arange(60, 120, 10))

		# 	ax2 = fig.add_subplot(4,1,2)
		# 	#ax2.set_ylim([-1.0,1.0])
		# 	ax2.grid()
		# 	ax2.plot(temp_times, np.gradient(temps), lw=2, color='k', label='TEMP-grad')
		# 	ax2.set_ylabel('Gradient of Temperature')

		# 	plt.legend(loc='upper right', labels=['TEMP-grad'], bbox_to_anchor=(1.1, 1))
		# 	plt.locator_params(axis='x', nbins=24)

		# 	ax3 = fig.add_subplot(4,1,3)
		# 	ax3.set_ylim([1,5])
		# 	ax3.set_ylabel('Annotation')
		# 	ax3.grid()
		# 	ax3.plot(annotation_times, annotations_r1, lw=2, color='c', label='R1')
		# 	ax3.plot(annotation_times, annotations_r2, lw=2, linestyle='--',color='m', label='R2')
			
		# 	plt.legend(loc='upper right', labels=['R1', 'R2'], bbox_to_anchor=(1.075, 1))
		# 	plt.locator_params(axis='x', nbins=24)

		# 	ax4 = fig.add_subplot(4,1,4)
		# 	ax4.set_ylim([-1,1])
		# 	ax4.set_xlabel('Time (s)')
		# 	ax4.set_ylabel('Gradient of Annotation')
		# 	ax4.grid()
		# 	ax4.plot(annotation_times, np.gradient(annotations_r1), lw=2, color='c', label='R1-grad')
		# 	ax4.plot(annotation_times, np.gradient(annotations_r2), lw=2, linestyle='--',color='m', label='R2-grad')

		# 	plt.legend(loc='upper right', labels=['R1-grad','R2-grad'], bbox_to_anchor=(1.1, 1))
		# 	plt.locator_params(axis='x', nbins=24)

		# 	mng = plt.get_current_fig_manager()
		# 	mng.window.showMaximized()

		# 	plt.show()

		# except IOError:
		# 	continue