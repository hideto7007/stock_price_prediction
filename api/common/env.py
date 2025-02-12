

import os
from typing import Any, Optional

from pydantic import BaseModel


class BaseEnvModel(BaseModel):
    ################
    # 環境変数の定義 #
    ################

    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    database_path: str
    test_database_path: str
    test_create_path: str
    test_delete_path: str


class Env:
    # シングルトンインスタンス
    _instance: Any = None

    def __init__(self):
        """インスタンス"""
        if self._instance is None:
            self._instance = self
            self._initialize()
        else:
            raise ValueError(
                "get_instanceメソッドを経由して作成してください。"
            )

    def _initialize(self):
        """環境変数の値を取得"""
        object.__setattr__(self, "SECRET_KEY", os.getenv("SECRET_KEY"))
        object.__setattr__(self, "ALGORITHM", os.getenv("ALGORITHM"))
        object.__setattr__(
            self,
            "ACCESS_TOKEN_EXPIRE_MINUTES",
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
        )
        object.__setattr__(self, "DATABASE_PATH", os.getenv("DATABASE_PATH"))
        object.__setattr__(self, "TEST_DATABASE_PATH",
                           os.getenv("TEST_DATABASE_PATH"))
        object.__setattr__(self, "TEST_CREATE_PATH",
                           os.getenv("TEST_CREATE_PATH"))
        object.__setattr__(self, "TEST_DELETE_PATH",
                           os.getenv("TEST_DELETE_PATH"))

    @classmethod
    def get_instance(cls) -> BaseEnvModel:
        """シングルトンインスタンス取得"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # --- @property を使って環境変数を取得する ---

    @property
    def secret_key(self) -> Optional[str]:
        # シークレットーキー
        return getattr(self, "SECRET_KEY")

    @property
    def algorithm(self) -> Optional[str]:
        # アルゴリズム
        return getattr(self, "ALGORITHM")

    @property
    def access_token_expire_minutes(self) -> Optional[int]:
        # アクセストークン有効時間
        return getattr(self, "ACCESS_TOKEN_EXPIRE_MINUTES")

    @property
    def database_path(self) -> Optional[str]:
        # データベースパス
        return getattr(self, "DATABASE_PATH")

    @property
    def test_database_path(self) -> Optional[str]:
        # テストデータベースパス
        return getattr(self, "TEST_DATABASE_PATH")

    @property
    def test_create_path(self) -> Optional[str]:
        # テストddl作成パス
        return getattr(self, "TEST_CREATE_PATH")

    @property
    def test_delete_path(self) -> Optional[str]:
        # テストddl削除パス
        return getattr(self, "TEST_DELETE_PATH")
