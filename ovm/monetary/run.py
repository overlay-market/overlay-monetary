from logs import start_logging
from server import server  # noqa

# Logs
start_logging()

# Server
server.launch()
