import os
from typing import Any, Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from common.env import Env

env = Env.get_instance()


class TestDataBase:
    """
        テスト用のデータベースエンジン
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
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_database_url = f"sqlite:///{os.path.join(
            self.base_dir,
            env.test_database_path
        )}"
        self.engine = create_engine(
            self.test_database_url,
            connect_args={"check_same_thread": False}
        )
        self.session_local = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        self.create_path = os.path.join(self.base_dir, env.test_create_path)
        self.insert_path = os.path.join(self.base_dir, env.test_insert_path)
        self.delete_path = os.path.join(self.base_dir, env.test_delete_path)

    @classmethod
    def get_test_db(cls) -> Generator[Session, Any, Any]:
        """
            テスト用のデータベースセッションの取得
            - データベースのセッション取得からクローズまで管理

            ジェネレーター
            - セッション情報を返す
        """
        db = cls().session_local()
        try:
            yield db
        finally:
            db.close()

    @classmethod
    def execute_sql_file(cls, file_path: str) -> None:
        """
            sql実行
            - ddlに記載されているsqlを実行しテスト用のデータベースに反映

            引数:
                file_path (str): sqlファイルパス

            戻り値:
                なし
        """
        with cls().engine.begin() as connection:
            with open(file_path, 'r') as f:
                sql_statements = f.read().split(';')
                for statement in sql_statements:
                    statement = statement.strip()
                    if statement:
                        connection.execute(text(statement + ";"))

    @classmethod
    def init_db(cls):
        """
            テスト用のテーブル初期化
            - テストクラスのユニットテスト開始時に実行される
        """
        cls.execute_sql_file(cls().create_path)
        cls.execute_sql_file(cls().insert_path)

    # TODO:CI実行時になぜか先に実行されてテーブルが削除されてしまう
    # そもそも、テスト終了時にデータベースごと削除するので必要ない気がする
    # @classmethod
    # def drop_db(cls):
    #     """
    #         テスト用のテーブル削除
    #         - テストクラスのユニットテスト終了時に実行される
    #     """
    #     cls.execute_sql_file(cls().delete_path)
