
from datetime import datetime, timedelta, timezone
from sqlalchemy.exc import IntegrityError
from api.common.authentication import Authentication
from typing import Any, Optional
from api.common.env import Env
from api.common.exceptions import (
    ConflictException, ExpiredSignatureException,
    NotFoundException, SqlException
)
from api.models.models import UserModel
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, false
from jose import jwt

from api.schemas.login import (
    CreateUserRequest, RestoreUserRequest, UpdateUserRequest
)
from utils.utils import Utils


env = Env.get_instance()


class LoginService:
    """
    ログインサービスクラス
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

    def get_user_info(
        self,
        db: Session,
        user_name: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> UserModel | None:
        """
            登録済みのユーザー情報取得

            引数:
                db (Session): dbインスタンス
                user_name (Optional[str]): ユーザー名 デフォルト None
                user_id (Optional[int]): ユーザーid デフォルト None

            戻り値:
                UserModel | None: userモデル (存在しない場合`None`で返す)
        """
        filters = []
        if user_name:
            filters.append(UserModel.user_name == user_name)
        if user_id:
            filters.append(UserModel.user_id == user_id)

        return db.query(UserModel).filter(
            and_(
                or_(*filters) if filters else false(),
                UserModel.is_valid == 1
            )
        ).first()

    def get_delete_user(
        self,
        db: Session,
        user_name: str
    ) -> UserModel | None:
        """
            削除済みのユーザー情報取得

            引数:
                db (Session): dbインスタンス
                user_name (str): ユーザー名

            戻り値:
                UserModel | None: userモデル (存在しない場合`None`で返す)
        """

        return db.query(UserModel).filter(
            UserModel.user_name == user_name,
            UserModel.is_valid == 0
        ).first()

    def _user_password_update(
        self,
        plain_password: str,
        hash_password: str,
        new_password: str
    ) -> str:
        """
            現在のパスワード存在チェック後新規ハッシュパスワード発行

            引数:
                plain_password (str): 登録済みのパスワード
                hash_password (str): 登録済みのハッシュパスワード
                new_password (str): 新規登録するパスワード

            戻り値:
                str: new_password
        """

        verify_password = Authentication.verify_password(
            plain_password,
            hash_password
        )

        # パスワードの存在チェック
        if verify_password:
            return Authentication.hash_password(
                new_password
            )
        raise NotFoundException("存在しないパスワードです。")

    def delete_make_recode(
        self,
        val: str,
    ) -> str:
        """
            削除時に登録するレコード作成

            引数:
                val (str): レコード

            戻り値:
                Column[str]: 削除レコード
        """

        return f'{val}_deleted'

    def save_user(
        self,
        db: Session,
        data: CreateUserRequest
    ) -> UserModel:
        """
            新規登録 or 削除済みのユーザー情報復活

            引数:
                db (Session): dbインスタンス
                data (CreateUserRequest): リクエストデータ

            戻り値:
                UserModel: ユーザーモデル
        """
        # 削除ずみのユーザー存在チェック
        user = self.get_delete_user(
            db,
            self.delete_make_recode(data.user_name)
        )

        if user:
            restore_data = RestoreUserRequest(
                user_name=data.user_name,
                user_email=data.user_email,
                user_password=data.user_password
            )
            # 削除済みユーザーを復活
            new_user = self.restore_user(db, user, restore_data)
        else:
            # 完全新規登録
            new_user = self.create_user(db, data)

        return new_user

    def create_user(
        self,
        db: Session,
        user: CreateUserRequest
    ) -> UserModel:
        """
            ユーザー情報登録

            引数:
                db (Session): dbインスタンス
                user (CreateUserRequest): ユーザーモデル

            戻り値:
                UserModel: ユーザーモデル
        """
        try:
            hashed_password = Authentication.hash_password(
                user.user_password
            )
            db_user = UserModel(
                user_name=user.user_name,
                user_email=user.user_email,
                user_password=hashed_password,
                create_at=Utils.today(),
                create_by=user.user_name,
                update_at=Utils.today(),
                update_by=user.user_name,
                is_valid=True
            )
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            return db_user
        except IntegrityError:
            db.rollback()
            raise ConflictException("ユーザー名かメールアドレスが既に存在しています。")
        except SqlException as e:
            db.rollback()
            raise e

    def restore_user(
        self,
        db: Session,
        user: UserModel,
        restore_data: RestoreUserRequest
    ) -> UserModel:
        """
            新規登録時に既に削除済みのユーザーが存在していたらそのレコードに対して更新
            引数:
                db (Session): dbインスタンス
                user (UserModel): ユーザーモデル
                restore_data (RestoreUserRequest): 更新するユーザー情報を含むリクエストデータ

            戻り値:
                UserModel: ユーザーモデル
        """
        try:
            user.user_name = Utils.column_str(restore_data.user_name)
            user.user_email = Utils.column_str(restore_data.user_email)
            user.user_password = Utils.column_str(
                Authentication.hash_password(
                    restore_data.user_password
                )
            )
            user.update_at = Utils.column_datetime(Utils.today())
            user.update_by = Utils.column_str(restore_data.user_name)
            user.is_valid = Utils.column_bool(True)
            db.commit()
            db.refresh(user)
            return user
        except SqlException as e:
            db.rollback()
            raise e

    def update_user(
        self,
        db: Session,
        user_id: int,
        update_data: UpdateUserRequest
    ) -> None:
        """
            ユーザー情報更新

            引数:
                db (Session): dbインスタンス
                user_id (int): ユーザーid
                update_data (UpdateUserRequest): 更新するユーザー情報を含むリクエストデータ

            戻り値:
                None
        """
        try:
            user = self.get_user_info(db, None, user_id)
            if user is None:
                raise NotFoundException("存在しないユーザーです。")

            if update_data.user_name is not None:
                user.user_name = Utils.column_str(update_data.user_name)
                user.update_by = Utils.column_str(update_data.user_name)

            if update_data.user_email is not None:
                user.user_email = Utils.column_str(update_data.user_email)

            if update_data.user_password is not None and \
                    update_data.user_confirmation_password is not None:
                user.user_password = Utils.column_str(
                    self._user_password_update(
                        update_data.user_confirmation_password,
                        str(user.user_password),
                        update_data.user_password
                    )
                )
            user.update_at = Utils.column_datetime(Utils.today())
            db.commit()
            db.refresh(user)
            return None
        except SqlException as e:
            db.rollback()
            raise e

    def delete_user(
        self,
        db: Session,
        user_id: int,
    ) -> None:
        """
            ユーザー情報論理削除

            引数:
                db (Session): dbインスタンス
                user_id (int): ユーザーid

            戻り値:
                None
        """
        try:
            user = self.get_user_info(db, None, user_id)
            if user is None:
                raise NotFoundException("存在しないユーザーです。")

            user_name = self.delete_make_recode(str(user.user_name))
            user_email = self.delete_make_recode(
                str(user.user_email))

            user.user_name = Utils.column_str(user_name)
            user.user_email = Utils.column_str(user_email)
            user.update_at = Utils.column_datetime(Utils.today())
            user.update_by = Utils.column_str(user_name)
            user.is_valid = Utils.column_bool(False)

            db.commit()
            db.refresh(user)
            return None
        except SqlException as e:
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
                Optional[UserModel] | None: ユーザーモデル (存在しない場合は`None`で返す)
        """
        user = self.get_user_info(db, user_name)
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
                expires_delta (Optional[timedelta]): 有効期限

            戻り値:
                str: 認証トークン
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
                str | None: ユーザー名 (存在しない場合は、`None`で返す)
        """
        payload = self.get_payload(token)
        if payload is None:
            return None
        user_name = payload.get("sub")
        return user_name
