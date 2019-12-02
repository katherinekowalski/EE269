import logging

ch = logging.StreamHandler()
formatter = logging.Formatter(fmt='%(levelname)s [%(asctime)s.%(msecs)d]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)

logger_ = logging.getLogger('nfc')
logger_.addHandler(ch)
logger_.setLevel(logging.DEBUG)  # This toggles all the logging in your app

def debug(*msg):
    logger_.debug(" ".join([str(m) for m in msg]))
def error(*msg):
    logger_.error(" ".join([str(m) for m in msg]))
def warn(*msg):
    logger_.warning(" ".join([str(m) for m in msg]))
def info(*msg):
    logger_.info(" ".join([str(m) for m in msg]))

if __name__ == "__main__":
    debug("test")
    error("test")
    warn("test")
    info("test")
