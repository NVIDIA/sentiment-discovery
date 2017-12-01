from .config import Config

cfg = Config()

def parse_args(script_name):
	"""parse standard arguments as well as experiment-specific args"""
	script_config = None
	if script_name == 'text_reconstruction.py':
		from .configure_text_reconstruction import script_config
	elif script_name == 'sentiment_transfer.py':
		from .configure_sentiment_transfer import script_config
	elif script_name == 'visualize.py':
		from .configure_visualize import script_config
	cfg(script_name, script_config)

def configure_usage(script_name):
	"""parse cmdline args and configure experiment"""
	parse_args(script_name)
	cfg.configure()
