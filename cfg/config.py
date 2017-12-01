import argparse

from .configure_system import configure_system
from .configure_model import configure_model
from .configure_devices import configure_devices
from .configure_data import configure_data

class Config(object):
	"""Holds commandline arguments and configured objects as instance attributes."""
	def __init__(self):
		super(Config, self).__init__()

	def __call__(self, name, script_config=None):
		"""Configures cmdline arg parsing and parses. Stores parsed args in `opt` attribute"""
		self.parser = argparse.ArgumentParser(description=name)
		self.parser.add_argument('-config', default='',
								help='pyaml/json file to load config from')
		self.system_config = configure_system(self.parser)
		self.model_config = configure_model(self.parser)
		self.device_config = configure_devices(self.parser)
		self.data_config = configure_data(self.parser)
		self.script_config = None
		if script_config is not None:
			self.script_config = script_config(self.parser)
		self.opt = self.parser.parse_args()
		return self

	def configure(self):
		"""
		Configures objects/data/models necessary to run program.
		Passes self to config to assign values as instance attributes.
		"""
		opt = self.opt
		self.system_config.apply(self, opt)
		self.model_config.apply(self, opt)
		self.device_config.apply(self, opt)
		self.data_config.apply(self, opt)
		if self.script_config is not None:
			self.script_config.apply(self, opt)
