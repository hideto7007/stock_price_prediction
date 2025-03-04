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

    def __new__(cls, file_name):
        if cls._instance is None:
            cls._instance = super(LoggerInitialize, cls).__new__(cls)
            cls._instance._initialize(file_name)
        return cls._instance

    def _initialize(self, file_name):
        """ロガーの初期化処理"""
        self.logger = logging.getLogger("app_logger")
        self.logger.setLevel(logging.DEBUG)

        # ログファイル存在チェック
        if not os.path.isfile(file_name):
            with open(file_name, "w"):
                pass
        file_handler = logging.FileHandler(
            file_name, mode='a', encoding='utf-8')
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

    def __init__(self, file_name):
        self.file_name = file_name
        self.logger = LoggerInitialize(file_name).get_logger()

    def error(
        self,
        req: Request,
        request_body: Any | str,
        result: Any
    ):
        """ERROR レベルのログを出力"""
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

        self.logger.error(json.dumps(log_data, ensure_ascii=False))

    def info(
        self,
        req: Request,
        request_body: Any | str,
        result: Any
    ):
        """INFO レベルのログを出力"""

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

        self.logger.info(json.dumps(log_data, ensure_ascii=False))

    def debug(
        self,
        req: Request,
        request_body: Any | str,
        result: Any
    ):
        """DEBUG レベルのログを出力"""

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

        self.logger.debug(json.dumps(log_data, ensure_ascii=False))
