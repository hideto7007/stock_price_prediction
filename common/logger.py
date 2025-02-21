import logging
import traceback
from typing import Any
from fastapi import Request
import json

from datetime import datetime, timedelta, timezone
import os


class ISOTimeFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt=None):
        tz_jst = timezone(timedelta(hours=+9), 'JST')
        ct = datetime.fromtimestamp(record.created, tz=tz_jst)
        s = ct.isoformat(timespec="microseconds")

        return s


class LoggerInitialize:
    _instance = None
    __file_name = './logger.log'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerInitialize, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """ロガーの初期化処理"""
        self.logger = logging.getLogger("app_logger")
        self.logger.setLevel(logging.DEBUG)
        if self.logger.hasHandlers():
            return

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # ログファイル存在チェック
        if not os.path.isfile(LoggerInitialize.__file_name):
            with open(LoggerInitialize.__file_name, "w"):
                pass
        file_handler = logging.FileHandler(
            LoggerInitialize.__file_name, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        fmt = ISOTimeFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger


class Logger:
    """ロギングヘルパークラス"""
    @staticmethod
    def error(
        req: Request,
        request_body: Any | str,
        result: Any
    ):
        """ERROR レベルのログを出力"""
        logger = LoggerInitialize().get_logger()

        error_traceback = traceback.format_exc()

        log_data = {
            "request_id": req.state.request_id,
            "method": req.method,
            "header": {
                "authorization": req.headers.get("authorization"),
                "cookie": req.headers.get("cookie"),
            },
            "params": dict(req.query_params),
            "request_body": request_body,
            "url": str(req.url),
            "error": result if isinstance(result, str) else str(result)
        }

        if error_traceback and error_traceback.strip() != "NoneType: None":
            log_data["traceback"] = error_traceback
            print(error_traceback)

        logger.error(json.dumps(log_data, ensure_ascii=False))

    @staticmethod
    def info(
        req: Request,
        request_body: Any | str,
        result: Any
    ):
        """INFO レベルのログを出力"""
        logger = LoggerInitialize().get_logger()

        log_data = {
            "request_id": req.state.request_id,
            "method": req.method,
            "header": {
                "authorization": req.headers.get("authorization"),
                "cookie": req.headers.get("cookie"),
            },
            "params": dict(req.query_params),
            "request_body": request_body,
            "url": str(req.url),
            "result": result
        }

        logger.info(json.dumps(log_data, ensure_ascii=False))

    @staticmethod
    def debug(
        req: Request,
        request_body: Any | str,
        result: Any
    ):
        """DEBUG レベルのログを出力"""
        logger = LoggerInitialize().get_logger()

        log_data = {
            "request_id": req.state.request_id,
            "method": req.method,
            "header": {
                "authorization": req.headers.get("authorization"),
                "cookie": req.headers.get("cookie"),
            },
            "params": dict(req.query_params),
            "request_body": request_body,
            "url": str(req.url),
            "result": result
        }

        logger.debug(json.dumps(log_data, ensure_ascii=False))
