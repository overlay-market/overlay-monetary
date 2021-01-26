import logging

DEBUG_LEVEL = 10
INFO_LEVEL = 20

PERFORM_DEBUG_LOGGING = logging.root.level <= DEBUG_LEVEL
PERFORM_INFO_LOGGING = logging.root.level <= INFO_LEVEL
