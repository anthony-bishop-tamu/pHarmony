import logging

VERBOSE_LEVEL = 15
logging.VERBOSE = VERBOSE_LEVEL                 # optional convenience
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

def verbose(self, msg, *args, **kwargs):
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, msg, args, **kwargs)

# add the method to all Logger instances
logging.Logger.verbose = verbose