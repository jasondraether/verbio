import numpy as np 

class Tempra(object):
	def __init__(self, times=None, data=None, **kwargs):
		self.times = times 
		self.data = data 

	def generate_times(self, t0, tn, interval):
		pass