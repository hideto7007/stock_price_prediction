import logging

from datetime import datetime, timedelta, timezone


class ISOTimeFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt=None):
        tz_jst = timezone(timedelta(hours=+9), 'JST')
        ct = datetime.fromtimestamp(record.created, tz=tz_jst)
        s = ct.isoformat(timespec="microseconds")

        return s

logger = logging.getLogger()
fmt = ISOTimeFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') # logging.Formatterの代わりに自作のクラスを使う

sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)

logger.setLevel(logging.INFO)


class Logger:
    @classmethod
    def info(cls, data):
        return logger.info(data)

    @classmethod
    def debug(cls, data):
        return logger.debug(data)

    @classmethod
    def warning(cls, data):
        return logger.warning(data)

    @classmethod
    def error(cls, data):
        return logger.error(data)
