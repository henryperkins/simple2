import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

class LoggerSetup:

    @staticmethod
    def get_logger(module_name: str, console_logging: bool=False) -> logging.Logger:
        """
        Get a logger for a specific module with optional console logging.

        Args:
            module_name (str): The name of the module for which to set up the logger.
            console_logging (bool): If True, also log to console.

        Returns:
            logging.Logger: Configured logger for the module.
        """
        logger = logging.getLogger(module_name)
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            LoggerSetup._add_file_handler(logger, module_name)
            if console_logging:
                LoggerSetup._add_console_handler(logger)
        return logger

    @staticmethod
    def _add_file_handler(logger: logging.Logger, module_name: str) -> None:
        """
        Add a file handler with a timestamped filename to the logger.
        """
        log_dir = os.path.join('logs', module_name)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f'{module_name}_{timestamp}.log')
        handler = RotatingFileHandler(log_filename, maxBytes=10 ** 6, backupCount=5)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    @staticmethod
    def _add_console_handler(logger: logging.Logger) -> None:
        """
        Add a console handler to the logger.
        """
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)