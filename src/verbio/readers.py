import os
import pandas as pd
from scipy.io import wavfile
import numpy as np

from verbio import settings

class DataReader(object):
    def __init__(self):
        self.excel_exts = ['.xlsx']
        self.audio_exts = ['.wav']

    def read_excel(self, file_path: str):
        """
        Read an excel file into a DataFrame using Pandas.

        :param file_path: Absolute or relative path to excel file
        :return: Dataframe of excel file
        """
        df = pd.read_excel(file_path, engine='openpyxl')
        return df


    def read_wav(self, file_path: str):
        """
        Read a wav file into a DataFrame using Scipy. Synthesize timestamps by
        using the sampling rate

        :param file_path: Path to wav file
        :return: Dataframe of wavfile data with sampling rate attached
        """
        sr, signal = wavfile.read(file_path)
        signal_len = signal.shape[0]
        # Synthesize times from sampling rate
        times = np.linspace(0.0, signal_len/sr, signal_len, dtype='float32')
        # Check if stereo or mono. If stereo, compress to mono
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)
        # Create df
        df = pd.DataFrame({settings.time_key: times, settings.audio_key: signal})
        # Attach sampling rate (for now...)
        df.sr = sr
        return df


    def read_file(self, file_path: str):
        """
        Given a vb file path, read the file and return
        a Pandas dataframe of the data within it.

        :param file_path: Path to the file
        :return: Pandas dataframe of the data in that file
        """
        _, ext = os.path.splitext(file_path)
        if ext in self.audio_exts:
            df = self.read_wav(file_path)
        elif ext in self.excel_exts:
            df = self.read_excel(file_path)
        else:
            raise ValueError(f'Extension {ext} not supported.')
        return df


    def read_dir(self, dir_path: str):
        """
        Given a vb dir path, read all files in the dir and return
        a list of Pandas dataframes of the data within all.

        :param dir_path: Path to the directory
        :return: List of Pandas dataframes of the data in that directory
        """
        file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        dfs = [self.read_file(file_path) for file_path in file_paths]
        return dfs
