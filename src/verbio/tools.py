from typing import List

def combine_dicts(data_map1: dict, data_map2: dict) -> dict:
	return {**data_map1, **data_map2}

def is_pt_valid(data_map: dict, exp_features: List[str], exp_sessions: List[str]) -> bool:
	"""Searches for (feature, session) pairs in data map for participant
	
	Args:
		data_map (dict): Dictionary indexed with keys (feature, session)
		exp_features (List[str]): Expected features to search for
		exp_sessions (List[str]): Expected sessions to search for
	
	Returns:
		bool: True if participant has all expected feature session pairs, False otherwise
	"""
	for feature in exp_features:
		for session in exp_sessions:
			if (feature, session) not in data_map: return False
	return True 

