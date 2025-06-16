import logging
import os
from datetime import datetime
from config import LOGS_DIR


class CustomLogger:
    """Singleton logger writing both to console and to a daily log file."""

    _logger_instance = None

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._logger_instance:
            return cls._logger_instance

        os.makedirs(LOGS_DIR, exist_ok=True)
        log_file = datetime.now().strftime(
            os.path.join(LOGS_DIR, "app_%Y-%m-%d.log")
        )

        logger = logging.getLogger("RAGAppLogger")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s"
            )
            # File handler
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            # Stream handler
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.addHandler(sh)

        cls._logger_instance = logger
        return logger
