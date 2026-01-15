import logging
from pathlib import Path
VERBOSE_LEVEL = 15
logging.VERBOSE = VERBOSE_LEVEL                 # optional convenience
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

def verbose(self, msg, *args, **kwargs):
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, msg, args, **kwargs)

# add the method to all Logger instances
logging.Logger.verbose = verbose
def normalize_level(level):
    if isinstance(level, int):
        return level
    name = str(level).upper()
    num = logging.getLevelName(name)
    if isinstance(num, int):    # works for VERBOSE after registration
        return num
    return int(level)           # allow numeric strings

def configure_logging(pkg_name, *, log_file=None, level="INFO",
                      console=True, overwrite=True, rotating=False):
    # Pre-create directory if logging to file
    handlers = {}
    logger_handlers = []

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "stream": "ext://sys.stdout",
        }
        logger_handlers.append("console")

    if log_file:
        p = Path(log_file).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        if rotating:
            handlers["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(p),
                "mode": "w" if overwrite else "a",
                "maxBytes": 10_000_000,
                "backupCount": 3,
                "encoding": "utf-8",
                "formatter": "std",
            }
        else:
            handlers["file"] = {
                "class": "logging.FileHandler",
                "filename": str(p),
                "mode": "w" if overwrite else "a",
                "encoding": "utf-8",
                "formatter": "std",
            }
        logger_handlers.append("file")

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"std": {"format": fmt, "datefmt": datefmt}},
        "handlers": handlers,
        "loggers": {
            pkg_name: {
                "level": normalize_level(level),  # children inherit
                "handlers": logger_handlers,
                "propagate": False,               # avoid duplicates via root
            }
        }
    })