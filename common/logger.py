import logging
from fastapi import Request
import json

from datetime import datetime, timedelta, timezone
import os
import traceback


class ISOTimeFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt=None):
        tz_jst = timezone(timedelta(hours=+9), 'JST')
        ct = datetime.fromtimestamp(record.created, tz=tz_jst)
        s = ct.isoformat(timespec="microseconds")

        return s


file_name = './logger.log'


class SingletonLogger:
    _instance = None

    def __new__(cls, log_file=file_name):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance._initialize(log_file)
        return cls._instance

    def _initialize(self, log_file):
        """ロガーの初期化処理"""
        self.logger = logging.getLogger("app_logger")
        if self.logger.hasHandlers():
            return  # すでにハンドラが設定されていたら何もしない

        self.logger.setLevel(logging.DEBUG)

        # ログファイル存在チェック
        if not os.path.isfile(log_file):
            with open(log_file, "w"):
                pass

        # ファイルハンドラ
        file_handler = logging.FileHandler(
            log_file, mode='a', encoding='utf-8')
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
    async def error(req: Request, exc: Exception):
        """ERROR レベルのログを出力"""
        logger = SingletonLogger().get_logger()

        try:
            body = await req.json()
        except Exception:
            body = await req.body()

        log_data = {
            "request_id": req.state.request_id,
            "method": req.method,
            "params": dict(req.query_params),
            "body": body,
            "url": str(req.url),
            "error": str(exc)
        }

        logger.error(json.dumps(log_data, ensure_ascii=False))  # JSON形式でログを出力
        traceback.print_exc()  # 例外の詳細なトレースバックを出力

    @staticmethod
    async def info(req: Request, message: str):
        """INFO レベルのログを出力"""
        logger = SingletonLogger().get_logger()

        log_data = {
            "request_id": req.state.request_id,
            "method": req.method,
            "params": dict(req.query_params),
            "url": str(req.url),
            "message": message
        }

        logger.info(json.dumps(log_data, ensure_ascii=False))

    @staticmethod
    async def debug(req: Request, message: str):
        """DEBUG レベルのログを出力"""
        logger = SingletonLogger().get_logger()

        log_data = {
            "request_id": req.state.request_id,
            "method": req.method,
            "params": dict(req.query_params),
            "url": str(req.url),
            "message": message
        }

        logger.debug(json.dumps(log_data, ensure_ascii=False))
