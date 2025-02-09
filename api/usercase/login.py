
from datetime import datetime, timedelta, timezone
from api.common.authentication import Authentication
from typing import Any, Optional
from api.common.env import Env
from api.common.exceptions import ExpiredSignatureException
from api.models.models import UserModel
from sqlalchemy.orm import Session
from jose import jwt

from api.schemas.login import UserCreateRequest


env = Env.get_instance()


class Login:
    """
    ログインクラス
    """

    @staticmethod
    def get_payload(
        token: str,
    ) -> dict[str, Any] | None:
        """
            認証トークンから情報取得

            引数:
                token (str): トークン
            戻り値:
                dict[str, Any]: payload情報
        """
        try:
            payload = jwt.decode(
                token,
                env.secret_key,
                algorithms=[env.algorithm]
            )
            return payload  # 正常な場合はデコードされたペイロードを返す
        except ExpiredSignatureException as e:
            raise e
        except Exception:
            return None

    def get_user_name(
        self,
        db: Session,
        user_name: str
    ) -> UserModel | None:
        """
            登録済みのユーザー情報取得

            引数:
                db (Session): dbインスタンス
                user_name (str): ユーザー名

            戻り値:
                str: ハッシュ化したパスワード
        """

        return db.query(UserModel) \
                 .filter(UserModel.user_name == user_name) \
                 .first()

    def create_user(
        self,
        db: Session,
        user: UserCreateRequest
    ) -> UserModel:
        """
            ユーザー情報登録

            引数:
                db (Session): dbインスタンス
                user (UserCreateRequest): ユーザーモデル

            戻り値:
                UserModel: ユーザーモデル
        """
        try:
            hashed_password = Authentication.hash_password(user.user_password)
            db_user = UserModel(
                user_name=user.user_name,
                user_email=user.user_email,
                user_password=hashed_password
            )
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            return db_user
        except Exception as e:
            db.rollback()
            raise e

    def authenticate_user(
        self,
        db: Session,
        user_name: str,
        user_password: str
    ) -> Optional[UserModel] | None:
        """
            認証チェック

            引数:
                db (Session): dbインスタンス
                user_name (str): ユーザー名
                user_password (str): ユーザーパスワード

            戻り値:
                UserCreate: ユーザーモデル
        """
        user = self.get_user_name(db, user_name)
        if not user:
            return None
        if not Authentication.verify_password(
            user_password,
            str(user.user_password)
        ):
            return None
        return user

    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
            認証トークン作成

            引数:
                data (dict): ユーザー名が含まれた辞書
                user_name (str): ユーザー名
                expires_delta (Optional[timedelta]): 有効期限

            戻り値:
                UserCreate: ユーザーモデル
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode,
            env.secret_key,
            algorithm=env.algorithm
        )
        return encoded_jwt

    def get_valid_user_name(
        self,
        token: str,
    ) -> str | None:
        """
            認証トークンからログイン中の有効なユーザー名取得

            引数:
                token (str): トークン
            戻り値:
                UserCreate: ユーザーモデル
        """
        payload = self.get_payload(token)
        if payload is None:
            return None
        user_name = payload.get("sub")
        return user_name
