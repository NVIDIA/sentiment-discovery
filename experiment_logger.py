import os
import pickle as pkl

def check_and_create_dir(dirpath):
	"""checks if directory exists, creates it if not"""
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)

class Logger(object):
	"""
	Handles creating directories for logging experiments + provides few basic logging functionalities.
	"""
	def __init__(self, experiment_name, experiment_dir='./experiments', sub_experiment=None):
		super(Logger, self).__init__()
		self.name = experiment_name
		self.root_dir = experiment_dir
		self.experiment_path = os.path.join(self.root_dir, self.name)
		self.sub_experiment = sub_experiment

	def set_sub_experiment(self, exp):
		self.sub_experiment = exp

	def get_log_dir(self, key, create_dir=True):
		"""gets directory key for the root from the root experiment directory"""
		log_path = self.experiment_path
		if self.sub_experiment is not None:
			log_path = os.path.join(log_path, self.sub_experiment)
		log_path = os.path.join(log_path, key)
		if create_dir:
			check_and_create_dir(log_path)
		return log_path

	def log_pkl(self, obj, key, basepath=None, write_mode='w'):
		"""gets key directory and saves `obj` in pkl file `basepath`"""
		dirpath = self.get_log_dir(key)
		if basepath is None:
			basepath = 'data.pkl'
		path = os.path.join(dirpath, basepath)
		pkl.dump(obj, open(path, write_mode))

	def log_txt(self, text, key, basepath=None, write_mode='w'):
		""""gets key directory and writes `text` in txt file `basepath`"""
		if basepath is None:
			basepath = 'log'
		dirpath = self.get_log_dir(key)
		with open(os.path.join(dirpath, basepath), write_mode) as f:
			f.write(text)
