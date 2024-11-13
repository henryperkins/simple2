import logging

def configure_logger(name='docstring_workflow', log_file='workflow.log', level=logging.DEBUG):
    """Configure and return a logger for the application."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)

        # Create handlers
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()

        # Set level for handlers
        file_handler.setLevel(level)
        console_handler.setLevel(logging.INFO)

        # Create formatter and add it to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Create and configure a global logger instance for the entire workflow
logger = configure_logger()

def log_info(message):
    """Log an informational message."""
    logger.info(message)

def log_error(message):
    """Log an error message."""
    logger.error(message)

def log_debug(message):
    """Log a debug message for detailed tracing."""
    logger.debug(message)
