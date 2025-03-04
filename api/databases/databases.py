import os
from typing import Any, Callable, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from common.env import Env


class DataBase:
    """
        データベースエンジン
    """
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance._initialize()
        return cls.__instance

    def _initialize(self) -> None:
        """
        コンストラクタ（初回の1回のみ実行）
        """
        env = Env.get_instance()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.database_url = f"sqlite:///{os.path.join(
            self.base_dir,
            env.database_path
        )}"
        self.engine = create_engine(
            self.database_url,
            connect_args={"check_same_thread": False}
        )
        self.session_local = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    @classmethod
    def get_db(cls) -> Callable[[], Generator[Session, Any, None]]:
        """
            内部でジェネレーターを返す
        """
        return cls()._get_db

    @staticmethod
    def _get_db():
        """
            テスト用のデータベースセッションの取得
            - データベースのセッション取得からクローズまで管理

            ジェネレーター
            - セッション情報を返す
        """
        db = DataBase().session_local()
        try:
            yield db
        finally:
            db.close()
