
import copy
from datetime import timedelta
from unittest.mock import MagicMock, patch

from jose import jwt

from api.common.authentication import Authentication
from api.common.env import Env
from api.common.exceptions import (
    ConflictException, ExpiredSignatureException,
    NotFoundException, SqlException
)
from api.models.models import UserModel
from api.schemas.login import (
    CreateUserRequest, RestoreUserRequest,
    UpdateUserRequest
)
from api.usercase.login import LoginService
from tests.test_case import TestBaseAPI
from utils.utils import Utils

# モック
JWT_DECODE = 'jose.jwt.decode'
REQUEST = 'fastapi.Request'
TRAIN_MAIN = 'prediction.train.train.PredictionTrain.main'
TEST_MAIN = 'prediction.test.test.PredictionTest.main'
STOCK_PRICE_BASE_PREDICTION = (
    'api.endpoints.stock_price.StockPriceBase.prediction'
)

env = Env.get_instance()


class TestLoginService(TestBaseAPI):

    def setUp(self):
        """テストごとにデータベースを初期化"""
        super().setUp()
        self.mock_db = MagicMock()
        self.db.begin()  # トランザクション開始

    def tearDown(self):
        """テストごとにデータベースをリセット"""
        super().tearDown()
        self.db.rollback()  # 変更をロールバック
        self.db.close()

    def _insert_user(self, user_id: int) -> None:
        """
        指定 ユーザーID でユーザーデータ登録

        引数:
            user_id (int): 登録ユーザーID

        戻り値:
            None
        """
        user_name = "hoge"
        db_user = UserModel(
            user_id=user_id,
            user_name=user_name,
            user_email="hoge@example.com",
            user_password="hoge_hashed_password",
            create_at=Utils.today(),
            create_by=user_name,
            update_at=Utils.today(),
            update_by=user_name,
            is_valid=True
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)

    def _delete_user_by_id(self, user_id: int) -> None:
        """
        指定した user_id のユーザーデータを物理削除

        引数:
            user_id (int): 削除対象のユーザーID

        戻り値:
            None
        """
        self.db.query(UserModel).filter(UserModel.user_id == user_id).delete()
        self.db.commit()

    @patch.object(jwt, "decode")
    def test_get_payload_error_01(
        self, _decode
    ):
        """
        異常系： `ExpiredSignatureException`が発生すること
        """
        mock_token = "mock_token"
        _decode.side_effect = ExpiredSignatureException("署名の期限切れ")
        with self.assertRaisesRegex(
            ExpiredSignatureException,
            "署名の期限切れ"
        ):
            LoginService.get_payload(mock_token)

    @patch.object(jwt, "decode")
    def test_get_payload_error_02(
        self, _decode
    ):
        """
        異常系： `Exception`が発生すること
        """
        mock_token = "mock_token"
        _decode.side_effect = Exception("予期せぬエラー")
        payload = LoginService.get_payload(mock_token)
        self.assertIsNone(payload)

    @patch.object(jwt, "decode")
    def test_get_payload_success_01(
        self, _decode
    ):
        """
        正常系： 認証トークンから情報取得できること
        """
        mock_token = "mock_token"
        _decode.return_value = {
            "sub": "test_user",
            "exp": 1740408437
        }
        payload = LoginService.get_payload(mock_token)

        self.assertIsNotNone(payload)
        # Pylance(reportOptionalSubscript)を無視してしまうと
        # 実処理のところで警告が出なくなってしまうため、
        # ここでは`type: ignore`で対応
        self.assertEqual(
            payload["sub"],  # type: ignore
            "test_user"
        )

    def test_get_user_info_success_01(self):
        """
        正常系： 登録済みのユーザー情報取得できなくNoneになること
        """
        params_list = [
            (None, "hoge"),
            (0, None),
            (None, None),
        ]
        for user_id, user_name in params_list:
            get_user_info = LoginService().get_user_info(
                self.db,
                user_name,
                user_id,
            )
            self.assertIsNone(get_user_info)

    def test_get_user_info_success_02(self):
        """
        正常系： 登録済みのユーザー情報取得ができること
        """
        params_list = [
            (None, "テスト"),
            (1, None),
            (1, "テスト"),
        ]
        for user_id, user_name in params_list:
            get_user_info = LoginService().get_user_info(
                self.db,
                user_name,
                user_id,
            )
            self.assertEqual(get_user_info.user_id, 1)
            self.assertEqual(get_user_info.user_email, "test@example.com")
            self.assertEqual(get_user_info.user_name, "テスト")

    def test_get_delete_user_success_01(self):
        """
        正常系： 削除済みのユーザー情報取得ができないこと
        """
        get_delete_user = LoginService().get_delete_user(
            self.db,
            "テスト"
        )
        self.assertIsNone(get_delete_user)

    def test_get_delete_user_success_02(self):
        """
        正常系： 削除済みのユーザー情報取得ができること
        """
        get_delete_user = LoginService().get_delete_user(
            self.db,
            "テスト1"
        )
        self.assertEqual(get_delete_user.user_id, 2)
        self.assertEqual(get_delete_user.user_email, "test1@example.com")
        self.assertEqual(get_delete_user.user_name, "テスト1")

    @patch.object(Authentication, "verify_password")
    def test_user_password_update_error_01(self, _verify_password):
        """
        異常系： 現在のパスワードが存在しない
        """
        _verify_password.return_value = False
        with self.assertRaisesRegex(
            NotFoundException,
            "存在しないパスワードです。"
        ):
            LoginService()._user_password_update(
                "plain_password",
                "hash_password",
                "new_password",
            )

    @patch.object(Authentication, "verify_password")
    @patch.object(Authentication, "hash_password")
    def test_user_password_update_success_01(
        self,
        _hash_password,
        _verify_password
    ):
        """
        正常系： 現在のパスワードが存在して新規パスワードが発行できる
        """
        _verify_password.return_value = True
        _hash_password.return_value = "new_password"
        user_password_update = LoginService()._user_password_update(
            "plain_password",
            "hash_password",
            "new_password",
        )
        self.assertEqual(user_password_update, "new_password")

    def test_delete_make_recode_success_01(self):
        """
        正常系： 削除時に登録するレコード作成
        """
        delete_make_recode = LoginService().delete_make_recode(
            "val",
        )
        self.assertEqual(delete_make_recode, "val_deleted")

    def test_save_user_success_01(self):
        """
        正常系： 新規登録から削除済みのユーザー情報復活まで検証
        """
        # 新規登録から検証
        login = LoginService()
        data = CreateUserRequest(
            user_name="test2",
            user_email="test2@example.com",
            user_password="Test12345!"
        )
        create_user = login.save_user(
            self.db,
            data
        )
        self.assertEqual(create_user.user_name, "test2")
        self.assertEqual(create_user.user_email, "test2@example.com")

        # 論理削除データ作成
        get_user_info = login.get_user_info(
            self.db,
            Utils.str(create_user.user_name),
        )

        delete = login.delete(
            self.db,
            Utils.int(get_user_info.user_id)
        )
        self.assertIsNone(delete)

        # 論理削除済みデータが存在するか確認
        get_delete_user = login.get_delete_user(
            self.db,
            login.delete_make_recode(data.user_name)
        )
        self.assertEqual(get_delete_user.user_name, "test2_deleted")
        self.assertEqual(get_delete_user.is_valid, 0)

        # 削除済みデータ復活検証
        update = login.save_user(
            self.db,
            data
        )
        self.assertEqual(update.user_name, "test2")
        self.assertEqual(update.is_valid, 1)

        # 最後に登録したデータ削除
        self._delete_user_by_id(Utils.int(update.user_id))

    def test_create_user_error_01(self):
        """
        異常系： ユーザー情報登録を行う際に既に登録済みのユーザーが存在
        - 正常系は`test_save_user_success_01`で検証済み
        """
        login = LoginService()
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        )
        with self.assertRaisesRegex(
            ConflictException,
            "ユーザー名かメールアドレスが既に存在しています。"
        ):
            login.create_user(
                self.db,
                data
            )

    def test_create_user_error_02(self):
        """
        異常系： 予期せぬエラー
        - 正常系は`test_save_user_success_01`で検証済み
        """
        login = LoginService()
        self.mock_db.commit.side_effect = SqlException("予期せぬエラー")
        data = CreateUserRequest(
            user_name="test2",
            user_email="test2@example.com",
            user_password="Test12345!"
        )
        with self.assertRaisesRegex(
            SqlException,
            "予期せぬエラー"
        ):
            login.create_user(
                self.mock_db,
                data
            )
        self.mock_db.rollback.assert_called_once()

    def test_restore_user_error_01(self):
        """
        異常系： 予期せぬエラー
        - 正常系は`test_save_user_success_01`で検証済み
        """
        login = LoginService()
        self.mock_db.commit.side_effect = SqlException("予期せぬエラー")
        mock_user = UserModel(
            user_id=1,
            user_name="deleted_user",
            user_email="deleted@example.com",
            user_password="hashed_password",
            is_valid=True
        )
        data = RestoreUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        )
        with self.assertRaisesRegex(
            SqlException,
            "予期せぬエラー"
        ):
            login.restore_user(
                self.mock_db,
                mock_user,
                data
            )
        self.mock_db.rollback.assert_called_once()

    def test_update_user_error_01(self):
        """
        異常系： 存在しないユーザー
        """
        login = LoginService()
        with self.assertRaisesRegex(
            NotFoundException,
            "存在しないユーザーです。"
        ):
            login.update(
                self.db,
                2,
                UpdateUserRequest()
            )

    @patch.object(LoginService, "_user_password_update")
    def test_update_user_error_02(
        self,
        _user_password_update
    ):
        """
        異常系： 予期せぬエラー
        """
        # 新規登録から検証
        login = LoginService()
        self.mock_db.commit.side_effect = SqlException("予期せぬエラー")
        _user_password_update.return_value = "new_password"
        data = UpdateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_confirmation_password="Test12345!",
            user_password="Test12345!"
        )
        with self.assertRaisesRegex(
            SqlException,
            "予期せぬエラー"
        ):
            login.update(
                self.mock_db,
                1,
                data
            )
        self.mock_db.rollback.assert_called_once()

    @patch.object(LoginService, "_user_password_update")
    def test_update_user_success_01(
        self,
        _user_password_update
    ):
        """
        正常系： ユーザー情報更新
        """
        login = LoginService()
        # データセット
        expected_user_name = "test_update"
        expected_user_email = "test_update@example.com"
        expected_password = "new_password"
        user_id = 3
        _user_password_update.return_value = expected_password
        data_1 = UpdateUserRequest(
            user_name="test_update",
            user_email=None,
            user_confirmation_password=None,
            user_password=None
        )
        data_2 = UpdateUserRequest(
            user_name=None,
            user_email="test_update@example.com",
            user_confirmation_password=None,
            user_password=None
        )
        data_3 = UpdateUserRequest(
            user_name=None,
            user_email=None,
            user_confirmation_password="Test12345!",
            user_password="Test12345!"
        )
        data_list = [
            ("test_update", data_1),
            ("test_update@example.com", data_2),
            (expected_password, data_3),
        ]
        # テスト用のデータ作成
        self._insert_user(user_id)
        for expected_result, data in data_list:

            # 更新日時確認用
            # before を deepcopy してオブジェクトの状態を固定
            before = copy.deepcopy(login.get_user_info(self.db, None, user_id))

            update = login.update(
                self.db,
                user_id,
                data
            )
            self.assertIsNone(update)

            # 更新データ取得し確認
            after = login.get_user_info(self.db, None, user_id)

            if expected_result == expected_user_name:
                self.assertEqual(after.user_name, expected_result)
            elif expected_result == expected_user_email:
                self.assertEqual(after.user_email, expected_result)
            elif expected_result == expected_password:
                self.assertEqual(after.user_password, expected_result)
            self.assertNotEqual(
                before.update_at,
                after.update_at,
                "更新時刻に変化なし"
            )

        # テストデータは削除
        self._delete_user_by_id(user_id)

    def test_delete_user_error_01(self):
        """
        異常系： 存在しないユーザー
        - 正常系は`test_save_user_success_01`で検証済み
        """
        login = LoginService()
        with self.assertRaisesRegex(
            NotFoundException,
            "存在しないユーザーです。"
        ):
            login.delete(
                self.db,
                999
            )

    def test_delete_user_error_02(self):
        """
        異常系： 予期せぬエラー
        """
        self.mock_db.commit.side_effect = SqlException("予期せぬエラー")
        login = LoginService()
        with self.assertRaisesRegex(
            SqlException,
            "予期せぬエラー"
        ):
            login.delete(
                self.mock_db,
                1
            )
        self.mock_db.rollback.assert_called_once()

    def test_authenticate_user_success_01(self):
        """
        正常系： 存在しないユーザーで`None`になる
        """
        login = LoginService()
        self.assertIsNone(
            login.authenticate_user(
                self.db,
                "hogehoge",
                "hashed_password"
            )
        )

    @patch.object(Authentication, "verify_password")
    def test_authenticate_user_success_02(
        self,
        _verify_password
    ):
        """
        正常系： 平文パスワード検証で存在していなく`None`になる
        """
        # データセット
        _verify_password.return_value = False
        login = LoginService()
        self.assertIsNone(
            login.authenticate_user(
                self.db,
                "テスト",
                "hashed_password"
            )
        )

    @patch.object(Authentication, "verify_password")
    def test_authenticate_user_success_03(
        self,
        _verify_password
    ):
        """
        正常系： 平文パスワード検証で一致してユーザーモデルが返されること
        """
        # データセット
        _verify_password.return_value = True
        login = LoginService()
        user = login.authenticate_user(
            self.db,
            "テスト",
            "hashed_password"
        )
        self.assertEqual(user.user_id, 1)
        self.assertEqual(user.user_name, "テスト")
        self.assertEqual(user.user_email, "test@example.com")

    @patch.object(jwt, "encode")
    def test_create_access_token_success_01(
        self,
        _encode
    ):
        """
        正常系： 平文パスワード検証で一致してユーザーモデルが返されること
        """
        # データセット
        token = "new_token"
        _encode.return_value = token
        data_list = [
            None,
            timedelta(
                minutes=int(
                    env.access_token_expire_minutes
                )
            )
        ]
        login = LoginService()
        for data in data_list:
            token = login.create_access_token(
                {"sub": "テスト"},
                data
            )
            self.assertEqual(token, token)

    @patch.object(LoginService, "get_payload")
    def test_get_valid_user_name_success_01(
        self,
        _get_payload
    ):
        """
        正常系： 認証トークンからログイン中の有効なユーザー名取得できなく`None`になる
        """
        # データセット
        _get_payload.return_value = None
        login = LoginService()
        self.assertIsNone(login.get_valid_user_name("token"))

    @patch.object(LoginService, "get_payload")
    def test_get_valid_user_name_success_02(
        self,
        _get_payload
    ):
        """
        正常系： 認証トークンからログイン中の有効なユーザー名取得
        """
        # データセット
        _get_payload.return_value = {"sub": "テスト"}
        login = LoginService()
        get_valid_user_name = login.get_valid_user_name("token")
        self.assertEqual(get_valid_user_name, "テスト")
