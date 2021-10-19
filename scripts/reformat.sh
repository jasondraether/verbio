#!/bin/bash

#if [ $# -eq 0 ]; then
#    echo "No arguments provided"
#    exit 1
#fi

# NOTE: The directories for each participant and each session must
# exist before running these scripts. To make, run something like
# $ mkdir P{001..073}/{PRE,POST,TEST01,TEST02,TEST03,TEST04,TEST05,TEST06,TEST07,TEST08}
# DON'T RUN IT WITH EXISTING DATA. IT IS BEST TO RUN IT IN A NEW FRESH FOLDER,
# THEN WORK WITH THAT FRESH FOLDER. IF YOU REMAKE ALL THE DIRECTORIES
# WITH EXISTING DATA, YOU MIGHT WIPE THAT DATA
shopt -s dotglob
# For each participant
for p in {001..073}
do
	# For each session
	for s in {PRE,POST,TEST01,TEST02,TEST03,TEST04,TEST05,TEST06,TEST07,TEST08}
	do

	  cp "/home/jason/hubbs/project_verbio/data/new_annotations/${s}/Annotation/${s}_P${p}_annotation.xlsx" "/home/jason/hubbs/project_verbio/data/raw_data/P${p}/${s}/MANUAL_ANNOTATION_PPT.xlsx"

    #mv "$1/P${i}/${p}/MANUAL_ANNOTATION_PPT.wav" "$1/P${i}/${p}/MANUAL_ANNOTATION_PPT.xlsx"

		# for t in {2sec,5sec,10sec}
		# do
		# 	mv "../data/extracted_data/P${i}/${p}/E4_${t}.xlsx" "../data/extracted_data_old/P${i}/${p}/" 
		# 	mv "../data/extracted_data/P${i}/${p}/IS09_${t}.xlsx" "../data/extracted_data_old/P${i}/${p}/" 
		# 	mv "../data/extracted_data/P${i}/${p}/IS10_${t}.xlsx" "../data/extracted_data_old/P${i}/${p}/" 
		# done
		#mv "../data/extracted_data/P${i}/${p}/annotation.xlsx" "../data/extracted_data_old/P${i}/${p}/" 

		# # Copying raw Actiwave data
		# cp -a "../../Akansha/raw_data/${p}/Actiwave/P${i}/." "../raw_data/P${i}/${p}/"
		
		# # Copying raw annotations
		# cp "../../Akansha/raw_data/${p}/Annotation/${p}_P${i}_annotation.xlsx" "../raw_data/P${i}/${p}/annotation.xlsx"
		
		# # Copying raw audio
		# cp "../../Akansha/raw_data/${p}/Audio/${p}_P${i}_AUDIO_PPT.wav" "../raw_data/P${i}/${p}/audio.wav"
		
		# # Rename Actiwave data to keep separate from E4
		# for v in {ECG_PPT,ECG_PREP,ECG_RELAX,HR_PPT,HR_PREP,HR_RELAX}
		# do
		# 	mv "../raw_data/P${i}/${p}/${v}.xlsx" "../raw_data/P${i}/${p}/Actiwave_${v}.xlsx"
		# done 
		
		# # Rename E4 data to keep it separate from Actiwave
		# cp -a "../../Akansha/raw_data/${p}/E4/P${i}/." "../raw_data/P${i}/${p}/"
		# for f in {ACC,BVP,EDA,HR,IBI,TEMP}
		# do
		# 	for v in {PPT,PREP,RELAX}
		#  	do
		#  		mv "../raw_data/P${i}/${p}/${f}_${v}.xlsx" "../raw_data/P${i}/${p}/E4_${f}_${v}.xlsx"
		#  	done 
		#  done 

		# # Copy Annotations
		# cp "Akansha/extracted_data/Annotation/${p}/${p}_P${i}_annotation.xlsx" "layout/P${i}/${p}/annotation.xlsx"

		# # Iterate over windowed data 
		# for t in {2sec,5sec,10sec}
		# do
		
		# # Copy audio (IS09 and IS10)
		# cp "Akansha/extracted_data/audio/${p}/${t}/${p}_P${i}_IS09.xlsx" "layout/P${i}/${p}/IS09_${t}.xlsx"
		# cp "Akansha/extracted_data/audio/${p}/${t}/${p}_P${i}_IS10.xlsx" "layout/P${i}/${p}/IS10_${t}.xlsx"
		
		# # Copy E4
		# cp "Akansha/extracted_data/E4/${p}/${t}/P${i}_E4_feature_PPT.xlsx" "layout/P${i}/${p}/E4_${t}.xlsx"
		
	done
done