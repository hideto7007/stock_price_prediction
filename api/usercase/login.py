
from datetime import datetime, timedelta
import os
from api.common.authentication import Authentication
from dotenv import load_dotenv  # type: ignore
from typing import Optional
from api.models.models import UserModel
from sqlalchemy.orm import Session  # type: ignore
from jose import jwt  # type: ignore

from api.schemas.schemas import (
    UserCreate
)


load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")


class Login:
    """
    ログインクラス
    """

    def get_user_name(
        self,
        db: Session,
        user_name: str
    ) -> Optional[UserModel]:
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
        user: UserCreate
    ) -> Optional[UserModel]:
        """
            ユーザー情報登録

            引数:
                db (Session): dbインスタンス
                user (UserCreate): ユーザーモデル

            戻り値:
                UserCreate: ユーザーモデル
        """
        hashed_password = Authentication.hash_password(user.user_password)
        db_user = UserModel(
            user_name=user.user_name,
            user_email=user.user_email,
            hashed_password=hashed_password
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    def authenticate_user(
        self,
        db: Session,
        user_name: str,
        user_password: str
    ) -> Optional[UserModel] | bool:
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
            return False
        if not Authentication.verify_password(
            user_password,
            user.user_password
        ):
            return False
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
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
