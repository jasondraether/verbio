import os
from typing import List
from verbio import settings, readers

class Dataset(object):
    def __init__(self, base_directory: str):
        """
        Dataset object for handling interactions with the VerBIO dataset.

        :param base_directory: Directory of VerBIO data. Inside this directory should be the subdirectories of participants
        """
        # Check and setup user parameters
        if not os.path.exists(base_directory):
            raise ValueError(f'Initialized Dataset object with directory {base_directory}, but directory does not exist.')
        self.base_directory = base_directory

        # Setup internal attributes
        # Participants range between 1 and 73
        self.all_pids = range(1, 74, 1)
        # Participant format is like P001, P005, P023, etc.
        self.pid_formatter = 'P{pid:03d}'
        # Data piece is like E4_EDA_PPT.xlsx, MIC_SPEECH_PPT.wav, etc.
        self.device_data_formatter = '{device}_{data_type}_{stage}'
        # Extensions for wav and excel files, along with the data types that go with them
        self.wav_extension = '.wav'
        self.wav_types = ['SPEECH']
        self.excel_extension = '.xlsx'
        self.excel_types = ['EDA', 'BVP', 'HR', 'TEMP', 'ACC', 'IBI', 'ECG', 'ANNOTATION']
        self.data_types = self.excel_types + self.wav_types
        # The four devices we use
        self.devices = ['E4', 'ACTIWAVE', 'MIC', 'MANUAL']
        # Stages of the public speaking task -- RELAX and PREP only available for real sessions, non-speech
        self.stages = ['RELAX', 'PREP', 'PPT']
        # Real presentation sessions in front of people
        self.real_sessions = ['PRE', 'POST']
        # VR headset presentations
        self.vr_sessions = ['TEST01', 'TEST02', 'TEST03', 'TEST04', 'TEST05', 'TEST06', 'TEST07', 'TEST08']
        # All sessions combined
        self.all_sessions = self.real_sessions + self.vr_sessions
        # Default device to use for each data type
        self.default_device_data = {
            'EDA': 'E4',
            'BVP': 'E4',
            'HR': 'E4',
            'TEMP': 'E4',
            'ACC': 'E4',
            'IBI': 'E4',
            'ECG': 'ACTIWAVE',
            'SPEECH': 'MIC',
            'ANNOTATION': 'MANUAL'
        }

    def format_pid(self, pid: int):
        """
        Format a participant ID for use with VerBIO file formats.

        :param pid: Participant ID (between 1 and 73)
        :return: Formatted participant ID string
        """
        return self.pid_formatter.format(pid=pid)

    def format_device_data(self, device: str, data_type: str, stage: str):
        """
        Format device, data type, and stage parameters for use with VerBIO file formats.

        :param device: Device source the data is coming from
        :param data_type: Type of data from that device
        :param stage: Stage of the device data (RELAX, PREP, PPT)
        :return: Formatted device data string
        """
        return self.device_data_formatter.format(device=device.upper(), data_type=data_type.upper(), stage=stage.upper())

    def format_data_path(self, pid: int, session: str, device: str, data_type: str, stage: str):
        """
        Format participant ID, session, device, data type, and stage parameters for use with VerBIO file formats.
        Get the relevant extension by checking against internal list. This file path is what gets passed to file readers.

        :param pid: Participant ID
        :param session: Session (PRE, POST, TEST01, etc.)
        :param device: Device source the data is coming from
        :param data_type: Type of data from that device
        :param stage: Stage of the device data
        :return: Formatted data path string
        """
        extension = self._get_data_extension(data_type)
        return os.path.join(self.format_pid(pid), session, self.format_device_data(device, data_type, stage) + extension)

    def get_pt_data(self, pid: int, session: str, data_type: str, stage: str = 'PPT', device: str = ''):
        """
        For a given participant, session, date type, stage, and device source, retrieve the dataframe for
        that piece of data in the VerBIO dataset

        :param pid: Participant ID
        :param session: Session
        :param data_type: Type of data from the given device
        :param stage: Stage of the device data
        :param device: Device source the data is coming from (leave blank for defaults)
        :return: Dataframe of the requested piece of data
        """
        if data_type not in self.data_types:
            raise ValueError(f'Data type {data_type} not recognized. Supported data types: {self.data_types}.')
        if pid not in self.all_pids:
            raise ValueError(f'PID {pid} out of range. Participant IDs range between [1, 73].')
        if session not in self.all_sessions:
            raise ValueError(f'Requested invalid session {session}. Sessions must be any from {self.all_sessions}.')

        if device == '':
            device = self.device_map[data_type]

        self._check_request(pid, session, data_type, stage, device)

        file_path = os.path.join(self.base_directory, self.format_data_path(pid, session, device, data_type, stage))
        data_reader = self._get_data_reader(data_type)

        return data_reader(file_path)

    def _get_data_reader(self, data_type: str):
        """
        Retrieve the associated file reader from the readers.py module for a given data type.

        :param data_type: Type of data
        :return: Callable file reader appropriate for that data
        """
        if data_type in self.wav_types:
            return readers.read_wav
        else:
            return readers.read_excel

    def _get_data_extension(self, data_type: str):
        """
        Retrieve the associated file extension for a given data type.

        :param data_type: Type of data
        :return: Extension appropriate for that data
        """
        if data_type in self.wav_types:
            return self.wav_extension
        else:
            return self.excel_extension

    def _check_request(self, pid, session, device, data_type, stage):
        """
        Perform validation and sanity checks on the request. Raises errors and warnings -- returns nothing.
        TODO: Check if session is missing

        :param pid: Participant ID
        :param session: Session
        :param device: Device source the data is coming from
        :param data_type: Type of data from that device
        :param stage: Stage of the device data
        :return: None
        """
        if data_type == 'SPEECH' and stage != 'PPT':
            raise ValueError(f'Stage {stage} was passed, but SPEECH data is only available for stage PPT.')

        file_path = os.path.join(self.base_directory, self.format_data_path(pid, session, device, data_type, stage))
        if not os.path.exists(file_path):
            raise ValueError(f'Path {file_path} does not exist.')
