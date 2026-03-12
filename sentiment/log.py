import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for scripts and notebooks.

    Should be called once at the entry point of any script or notebook.
    Library code must never call this function.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
