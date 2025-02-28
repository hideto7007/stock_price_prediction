from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api.common.authentication import Authentication
from api.models.models import UserModel
from api.schemas.login import (
    CreateUserRequest, LoginUserModel, LoginUserRequest,
    LoginUserResponse, UpdateUserRequest, UserResponseModel
)
from api.schemas.response import Content
from api.schemas.validation import ValidatonModel
from api.usercase.login import LoginService
from const.const import HttpStatusCode

from tests.test_case import TestBaseAPI
from utils.utils import Utils

# モック
GET_OAUTH2_SCHEME = 'api.middleware.OAuth2Middleware.get_oauth2_scheme'
IS_VALID_TOKEN = 'api.middleware.OAuth2Middleware.is_valid_token'


class TestEndpointBase(TestBaseAPI):

    def setUp(self):
        """各テスト実行前に認証処理をモック"""
        super().setUp()

        self.get_oauth2_scheme_patch = patch(
            GET_OAUTH2_SCHEME,
            new_callable=AsyncMock
        )
        self.is_valid_token_patch = patch(IS_VALID_TOKEN)

        self.mock_get_oauth2_scheme = self.get_oauth2_scheme_patch.start()
        self.mock_is_valid_token = self.is_valid_token_patch.start()

        self.mock_get_oauth2_scheme.return_value = "token"
        self.mock_is_valid_token.return_value = {"auth": "token"}

    def tearDown(self):
        """各テスト実行後にモックを解除"""
        super().tearDown()
        self.get_oauth2_scheme_patch.stop()
        self.is_valid_token_patch.stop()

    def _insert_user(self, user: UserModel) -> None:
        """
        テスト用のユーザーデータ登録

        引数:
            user (UserModel): ユーザーモデル

        戻り値:
            None
        """
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

    def _delete_user_by_id(self, user_id: int) -> None:
        """
        指定した user_id のユーザーデータを物理削除

        引数:
            user_id (int): 削除対象のユーザーid

        戻り値:
            None
        """
        self.db.query(UserModel).filter(
            UserModel.user_id == user_id).delete()
        self.db.commit()


class TestRegisterUser(TestEndpointBase):

    def get_path(self) -> str:
        """
        ログイン情報登録APIのパス

        引数:
            なし
        戻り値:
            str: urlパス
        """
        return self.get_login_path(
            'register_user'
        )

    def test_register_user_valid_error_01(self):
        """
        正常系: バリデーションエラー
        - 以下項目の須チェック
            - user_name
            - user_email
            - user_password
        """
        # データセット
        data = CreateUserRequest(
            user_name="",
            user_email="",
            user_password=""
        ).model_dump()
        # API実行
        response = self.post(self.get_path(), data)
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="user_name",
                    message="user_nameは必須です。"
                ),
                ValidatonModel(
                    field="user_email",
                    message="user_emailは必須です。"
                ),
                ValidatonModel(
                    field="user_password",
                    message="user_passwordは必須です。"
                ),
            ]
        )
        self.response_body_check(
            response.json(), expected_response.model_dump())

    def test_register_user_valid_error_02(self):
        """
        正常系: バリデーションエラー
        - user_email形式チェック
        """
        # データセット
        data_list = [
            CreateUserRequest(
                user_name="test",
                user_email="testexample.com",
                user_password="Test12345!"
            ).model_dump(),
            CreateUserRequest(
                user_name="test",
                user_email="test@examplecom",
                user_password="Test12345!"
            ).model_dump(),
        ]
        # API実行
        for data in data_list:
            response = self.post(self.get_path(), data)
            self.assertEqual(
                response.status_code,
                HttpStatusCode.BADREQUEST.value
            )
            expected_response = Content[list[ValidatonModel]](
                result=[
                    ValidatonModel(
                        field="user_email",
                        message="user_emailの形式が間違っています。"
                    )
                ]
            )
            self.assertEqual(response.json(), expected_response.model_dump())

    def test_register_user_valid_error_03(self):
        """
        正常系: バリデーションエラー
        - user_password形式チェック
        """
        # データセット
        data_list = [
            CreateUserRequest(
                user_name="test",
                user_email="test@example.com",
                user_password="Test12345"
            ).model_dump(),
            CreateUserRequest(
                user_name="test",
                user_email="test@example.com",
                user_password="test12345!"
            ).model_dump(),
            CreateUserRequest(
                user_name="test",
                user_email="test@example.com",
                user_password="Test12!"
            ).model_dump(),
        ]
        # API実行
        for data in data_list:
            response = self.post(self.get_path(), data)
            self.assertEqual(
                response.status_code,
                HttpStatusCode.BADREQUEST.value
            )
            expected_response = Content[list[ValidatonModel]](
                result=[
                    ValidatonModel(
                        field="user_password",
                        message=(
                            ("user_passwordは8文字以上24文字以下、大文字、記号(ビックリマーク(!)、"
                             "ピリオド(.)、スラッシュ(/)、クエスチョンマーク(?)、ハイフン(-))を含めてください")
                        )
                    )
                ]
            )
            self.assertEqual(response.json(), expected_response.model_dump())

    def test_register_user_error_01(self):
        """
        異常系: 既に登録済みのユーザー
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="既に登録済みのユーザーです"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.CONFLICT.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_register_user_error_02(
        self,
        _get_user_info
    ):
        """
        異常系: 予期せぬエラー
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    @patch.object(LoginService, "save_user")
    def test_register_user_success_01(
        self,
        _save_user,
        _get_user_info
    ):
        """
        正常系: ログイン情報登録
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        user = UserModel(
            user_id=1,
            user_name="テスト",
            user_email="test@example.com",
            user_password="hashed_password_123",
        )
        _get_user_info.return_value = None
        _save_user.return_value = user
        # API実行
        response = self.post(self.get_path(), data)

        expected_response = Content[UserResponseModel](
            result=UserResponseModel(
                user_id=Utils.int(user.user_id),
                user_email=str(user.user_email),
                user_name=str(user.user_name),
                user_password=str(user.user_password)
            )
        )
        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())


class TestLoginUser(TestEndpointBase):

    def get_path(self) -> str:
        """
        ログイン情報取得APIのパス

        引数:
            なし
        戻り値:
            str: urlパス
        """
        return self.get_login_path(
            'login_user'
        )

    def test_login_user_valid_error_01(self):
        """
        正常系: バリデーションエラー
        - 以下項目の須チェック
            - user_name
            - user_password
        """
        # データセット
        data = LoginUserRequest(
            user_name="",
            user_password=""
        ).model_dump()
        # API実行
        response = self.post(self.get_path(), data)
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="user_name",
                    message="user_nameは必須です。"
                ),
                ValidatonModel(
                    field="user_password",
                    message="user_passwordは必須です。"
                ),
            ]
        )
        self.response_body_check(
            response.json(), expected_response.model_dump())

    def test_login_user_valid_error_02(self):
        """
        正常系: バリデーションエラー
        - user_password形式チェック
        """
        # データセット
        data_list = [
            LoginUserRequest(
                user_name="test",
                user_password="Test12345"
            ).model_dump(),
            LoginUserRequest(
                user_name="test",
                user_password="test12345!"
            ).model_dump(),
            LoginUserRequest(
                user_name="test",
                user_password="Test12!"
            ).model_dump(),
        ]
        # API実行
        for data in data_list:
            response = self.post(self.get_path(), data)
            self.assertEqual(
                response.status_code,
                HttpStatusCode.BADREQUEST.value
            )
            expected_response = Content[list[ValidatonModel]](
                result=[
                    ValidatonModel(
                        field="user_password",
                        message=(
                            ("user_passwordは8文字以上24文字以下、大文字、記号(ビックリマーク(!)、"
                             "ピリオド(.)、スラッシュ(/)、クエスチョンマーク(?)、ハイフン(-))を含めてください")
                        )
                    )
                ]
            )
            self.assertEqual(response.json(), expected_response.model_dump())

    def test_login_user_error_01(self):
        """
        異常系: ユーザーIDまたはパスワードが異なる場合
        """
        # テスト用のサンプルデータ登録
        user_id = 3
        user_name = "サンプル"
        user_password = "Test12345!"
        self._insert_user(
            UserModel(
                user_id=user_id,
                user_name=user_name,
                user_email="sample@example.com",
                user_password=Authentication.hash_password(
                    user_password
                ),
                create_at=Utils.today(),
                create_by=user_name,
                update_at=Utils.today(),
                update_by=user_name,
                is_valid=True
            )
        )
        # データセット
        data_list = [
            LoginUserRequest(
                user_name="test1",
                user_password=user_password
            ).model_dump(),
            LoginUserRequest(
                user_name=user_name,
                user_password="Test1234567!"
            ).model_dump(),
        ]

        for data in data_list:
            response = self.post(self.get_path(), data)
            expected_response = Content[str](
                result="ユーザーIDまたはパスワードが間違っています。"
            )

            self.assertEqual(
                response.status_code,
                HttpStatusCode.UNAUTHORIZED.value
            )
            self.assertEqual(response.json(), expected_response.model_dump())
        # 最後に削除
        self._delete_user_by_id(user_id)

    @patch.object(LoginService, "authenticate_user")
    def test_login_user_error_02(
        self,
        _authenticate_user
    ):
        """
        異常系: 予期せぬエラー
        """
        # データセット
        data = LoginUserRequest(
            user_name="test",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _authenticate_user.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "authenticate_user")
    @patch.object(LoginService, "create_access_token")
    def test_login_user_success_01(
        self,
        _create_access_token,
        _authenticate_user
    ):
        """
        正常系: ログイン情報取得
        """
        # データセット
        data = LoginUserRequest(
            user_name="test",
            user_password="Test12345!"
        ).model_dump()
        user = UserModel(
            user_id=1,
            user_name="テスト",
            user_email="test@example.com",
            user_password="hashed_password_123",
        )
        access_token = "mock_token"
        _authenticate_user.return_value = user
        _create_access_token.return_value = access_token
        # API実行
        response = self.post(self.get_path(), data)

        expected_response = Content[LoginUserResponse](
            result=LoginUserResponse(
                user_info=LoginUserModel(
                    user_id=Utils.int(user.user_id),
                    user_name=str(user.user_name),
                    user_email=str(user.user_email),
                    access_token=access_token
                )
            )
        )
        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())
        self.assertEqual(
            response.cookies.get("auth_stock_price_token"),
            access_token
        )


class TestReadUsersMe(TestEndpointBase):

    def get_path(self, token: str) -> str:
        """
        ユーザー情報取得APIのパス

        引数:
            token (str): トークン
        戻り値:
            str: urlパス
        """
        return self.get_login_path(
            f'read_users_me/{token}'
        )

    @patch.object(LoginService, "get_valid_user_name")
    def test_read_users_me_error_01(
        self,
        _get_valid_user_name
    ):
        """
        異常系: 認証情報の有効期限が切れている
        """
        # mock
        _get_valid_user_name.return_value = None
        # API実行
        response = self.get(self.get_path("mock_token"))

        expected_response = Content[str](
            result="認証情報の有効期限が切れています。"
        )
        self.assertEqual(
            response.status_code,
            HttpStatusCode.UNAUTHORIZED.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    @patch.object(LoginService, "get_valid_user_name")
    def test_read_users_me_error_02(
        self,
        _get_valid_user_name,
        _get_user_info
    ):
        """
        異常系: 対象のユーザー情報がない
        """
        # mock
        _get_valid_user_name.return_value = "test"
        _get_user_info.return_value = None
        # API実行
        response = self.get(self.get_path("mock_token"))

        expected_response = Content[str](
            result="対象のユーザー情報がありません。"
        )
        self.assertEqual(
            response.status_code,
            HttpStatusCode.UNAUTHORIZED.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_valid_user_name")
    def test_login_user_error_02(
        self,
        _get_valid_user_name
    ):
        """
        異常系: 予期せぬエラー
        """
        # エラーモック
        _get_valid_user_name.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.get(self.get_path("mock_token"))
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    @patch.object(LoginService, "get_valid_user_name")
    def test_read_users_me_success_01(
        self,
        _get_valid_user_name,
        _get_user_info
    ):
        """
        正常系: ユーザー情報取得
        """
        # データセット
        user = UserModel(
            user_id=1,
            user_name="テスト",
            user_email="test@example.com",
            user_password="hashed_password_123",
        )
        # mock
        _get_valid_user_name.return_value = "test"
        _get_user_info.return_value = user
        # API実行
        response = self.get(self.get_path("mock_token"))

        expected_response = Content[UserResponseModel](
            result=UserResponseModel(
                user_id=Utils.int(user.user_id),
                user_name=str(user.user_name),
                user_email=str(user.user_email)
            )
        )
        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())


class TestUserInfoMe(TestEndpointBase):

    def get_path(self, user_id: int) -> str:
        """
        ユーザー情報取得APIのパス

        引数:
            user_id (int): 対象のユーザーid
        戻り値:
            str: urlパス
        """
        return self.get_login_path(
            f'user_info/{user_id}'
        )

    def test_login_user_valid_error_01(self):
        """
        正常系: バリデーションエラー
        - 以下項目の須チェック
            - user_id
        """
        # API実行
        response = self.get(self.get_path(0))
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="user_id",
                    message="user_idは必須です。"
                )
            ]
        )
        self.response_body_check(
            response.json(), expected_response.model_dump())

    def test_user_info_error_01(self):
        """
        異常系: 認証情報の有効期限が切れている
        """
        # API実行
        response = self.get(self.get_path(999999))

        expected_response = Content[str](
            result="対象のユーザー情報がありません。"
        )
        self.assertEqual(
            response.status_code,
            HttpStatusCode.UNAUTHORIZED.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_user_info_error_02(
        self,
        _get_user_info
    ):
        """
        異常系: 予期せぬエラー
        """
        # エラーモック
        _get_user_info.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.get(self.get_path(1))
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_user_info_success_01(
        self,
        _get_user_info
    ):
        """
        正常系: ユーザー情報取得
        """
        # データセット
        user = UserModel(
            user_id=1,
            user_name="テスト",
            user_email="test@example.com",
            user_password="hashed_password_123",
        )
        # mock
        _get_user_info.return_value = user
        # API実行
        response = self.get(self.get_path(1))

        expected_response = Content[UserResponseModel](
            result=UserResponseModel(
                user_id=Utils.int(user.user_id),
                user_name=str(user.user_name),
                user_email=str(user.user_email)
            )
        )
        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())


class TestUpdateUserPrice(TestEndpointBase):

    def get_path(self, user_id: int) -> str:
        """
        ユーザー情報更新APIのパス

        引数:
            user_id (int): ユーザーid

        戻り値:
            str: urlパス
        """
        return self.get_login_path(
            f'update_user/{user_id}'
        )

    def test_update_user_valid_error_01(self):
        """
        正常系: バリデーションエラー
        - 以下項目の須チェック
            - user_id
        """
        # データセット
        data = UpdateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_confirmation_password="Test12345!",
            user_password="Test12345!"
        ).model_dump()
        # API実行
        response = self.put(self.get_path(0), data)
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="user_id",
                    message="user_idは必須です。"
                )
            ]
        )
        self.response_body_check(
            response.json(),
            expected_response.model_dump()
        )

    @patch.object(LoginService, "update")
    def test_update_user_error_01(self, _update):
        """
        異常系: 予期せぬエラー
        """
        # データセット
        data = UpdateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_confirmation_password="Test12345!",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _update.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.put(self.get_path(1), data)
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "update")
    def test_update_usersuccess_01(self, _update):
        """
        正常系: 更新成功
        """
        # データセット
        data = UpdateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_confirmation_password="Test12345!",
            user_password="Test12345!"
        ).model_dump()
        _update.return_value = None
        # API実行
        response = self.put(self.get_path(1), data)

        expected_response = Content[str](
            result="ユーザー情報の更新成功"
        )

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())


class TestDeleteUser(TestEndpointBase):

    def get_path(self, user_id: int) -> str:
        """
        予測データ削除APIのパス

        引数:
            user_id (int): ユーザーid

        戻り値:
            str: urlパス
        """
        return self.get_login_path(
            f'delete_user/{user_id}'
        )

    def test_delete_user_valid_error_01(self):
        """
        正常系: バリデーションエラー
        - 以下項目の須チェック
            - user_id
        """
        # API実行
        response = self.delete(self.get_path(0))
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="user_id",
                    message="user_idは必須です。"
                )
            ]
        )
        self.response_body_check(
            response.json(),
            expected_response.model_dump()
        )

    @patch.object(
        LoginService,
        "delete"
    )
    def test_delete_user_error_01(self, _delete):
        """
        異常系: 予期せぬエラー
        """
        # エラーモック
        _delete.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.delete(self.get_path(1))
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(
        LoginService,
        "delete"
    )
    def test_delete_user_success_01(
        self,
        _delete
    ):
        """
        正常系: 削除成功
        """
        _delete.return_value = None
        # API実行
        response = self.delete(self.get_path(1))

        expected_response = Content[str](
            result="ユーザー情報の削除成功"
        )

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())
