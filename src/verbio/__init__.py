"""
Notes:
    'vb' is an alias for 'VerBIO'

    't' is a reserved key for dataframes that holds timestamps

    Shifting by times is unsupported at this time
"""
# Need this to initialize submodules in verbio namespace
from verbio import dataset, features, preprocessing, readers, temporal, utils, settings