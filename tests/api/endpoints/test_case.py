import json
from typing import List
from deepdiff.diff import DeepDiff
import unittest
from fastapi.testclient import TestClient
from httpx import Response

from api.main import app
from api.databases.databases import DataBase
from tests.api.database.test_database import TestDataBase


class TestBase(unittest.TestCase):
    """
    unittestを継承したテストクラス
    """

    def params_error_check(
        self,
        code: str,
        msg: str,
        params_list: List[str],
        input_data_list: List[str],
        res_data: dict
    ) -> None:
        """
            パラメータのエラーチェック

            引数:
                code (str): Httpステータスコード
                msg (str): エラーメッセージ
                params_list (List[str]): クエリーパラメータのリスト
                input_data_list (List[str]): リクエストデータのリスト
                res_data (dict): レスポンスデータ
            戻り値:
                なし、アサートでテスト検証

            例: res = [
                {
                    "code": ErrorCode.INT_VAILD.value,
                    "detail": f"{loc[1]} パラメータは整数値のみです。",
                    "input": input_msg
                }
            ]
        """
        self.assertEqual(len(res_data), len(params_list))
        for res, param, input_data in zip(
            res_data,
            params_list,
            input_data_list
        ):
            self.assertEqual(res.get("code"), code)
            self.assertEqual(res.get("detail"), f"{param} {msg}")
            self.assertEqual(res.get("input"), input_data)

    def response_body_error_check(
        self,
        res_data: List[dict],
        res_expected_data: List[dict]
    ) -> None:
        """
            レスポンスのエラーチェック

            引数:
                res_data (List[dict]): レスポンス結果
                res_expected_data (List[dict]): 期待するレスポンス結果
            戻り値:
                なし、アサートでテスト検証

            例: res = {
                "result": [
                    {
                        "field": "user_id",
                        "message": "user_idは必須です。"
                    }
                ]
            }
        """
        self.assertEqual(len(res_data), len(res_expected_data))
        self.assertEqual(
            DeepDiff(
                res_data,
                res_expected_data,
                ignore_order=True
            ), {}
        )


class TestBaseAPI(TestBase):
    """
    TestBaseを継承しdbアクセス処理を追加したテストクラス
    """

    @classmethod
    def setUpClass(cls):
        """テスト用データベースの初期化"""
        TestDataBase.init_db()

    @classmethod
    def tearDownClass(cls):
        """テスト用データベースの削除"""
        TestDataBase.drop_db()

    def setUp(self):
        """セットアップ"""
        self.client = TestClient(app)
        self.db = next(TestDataBase.get_test_db())
        app.dependency_overrides[DataBase.get_db] = TestDataBase.get_test_db

    def tearDown(self):
        """テスト終了時処理"""
        self.db.rollback()
        self.db.close()

    def delete_client(
        self,
        url: str,
        data: dict
    ) -> Response:
        """
            削除クライアント

            - データベース内のデータを削除する際に使用

            引数:
                url (str): 対象のURL
                data (dict): リクエストボディデータ
            戻り値:
                Response: レスポンス型

        """
        response = self.client.request(
            method="DELETE",
            url=url,
            headers={"Content-Type": "application/json"},
            content=json.dumps(data)
        )
        return response
