import typing as tp
import logging

from ovm.debug_level import DEBUG_LEVEL

# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def start_logging():
    logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=logFormatter, level=logging.DEBUG)


def console_log(logger, msgs: tp.List[str], level: int = DEBUG_LEVEL):
    if logger.getEffectiveLevel() > level:
        return

    for msg in msgs:
        logger.debug(msg)
