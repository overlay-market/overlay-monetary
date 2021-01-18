import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter,
                    level=logging.DEBUG
                    )

# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

from ovm.monetary.server import server  # noqa

server.launch()
