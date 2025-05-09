import os
import torch
import logging

env_level = os.environ.get('QFACTORY_LOG_LEVEL')
if env_level is None:
    level = logging.INFO
elif env_level == 'DEBUG':
    level = logging.DEBUG
elif env_level == 'INFO':
    level = logging.INFO
elif env_level == 'WARNING':
    level = logging.WARNING
elif env_level == 'ERROR':
    level = logging.ERROR
elif env_level == 'CRITICAL':
    level = logging.CRITICAL
else:
    assert False, f"Unknown log level {env_level}"

logging.basicConfig(
    format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    level=level
)

from .core import QMatmul
from .qlinear import QLinear
