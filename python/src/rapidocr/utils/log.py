"""Logger name matches rapidocr 3.x (``RapidOCR``)."""

import logging

logger = logging.getLogger("RapidOCR")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s %(filename)s:%(lineno)d: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
