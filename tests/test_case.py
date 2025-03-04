from typing import List, Optional
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

    # def params_error_check(
    #     self,
    #     code: str,
    #     msg: str,
    #     params_list: List[str],
    #     input_data_list: List[str],
    #     res_data: dict
    # ) -> None:
    #     """
    #         パラメータのエラーチェック

    #         引数:
    #             code (str): Httpステータスコード
    #             msg (str): エラーメッセージ
    #             params_list (List[str]): クエリーパラメータのリスト
    #             input_data_list (List[str]): リクエストデータのリスト
    #             res_data (dict): レスポンスデータ
    #         戻り値:
    #             なし、アサートでテスト検証

    #         例: res = [
    #             {
    #                 "code": ErrorCode.INT_VAILD.value,
    #                 "detail": f"{loc[1]} パラメータは整数値のみです。",
    #                 "input": input_msg
    #             }
    #         ]
    #     """
    #     self.assertEqual(len(res_data), len(params_list))
    #     for res, param, input_data in zip(
    #         res_data,
    #         params_list,
    #         input_data_list
    #     ):
    #         self.assertEqual(res.get("code"), code)
    #         self.assertEqual(res.get("detail"), f"{param} {msg}")
    #         self.assertEqual(res.get("input"), input_data)

    def response_body_check(
        self,
        res_data: List[dict],
        res_expected_data: List[dict]
    ) -> None:
        """
            レスポンスチェック
            - 機能：リストの辞書型で要素の順序問わず一致してるか確認

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
        super().setUpClass()
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

    def get_stock_price_path(self, endpoint: str) -> str:
        """
            株価予測APIのパス

            引数:
                endpoint (str): 各エンドポイント
            戻り値:
                str: urlパス
        """
        return f'/api/stock_price/{endpoint}'

    def get_login_path(self, endpoint: str) -> str:
        """
            認証APIのパス

            引数:
                endpoint (str): 各エンドポイント
            戻り値:
                str: urlパス
        """
        return f'/api/login/{endpoint}'

    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: dict = {"Content-Type": "application/json"}
    ) -> Response:
        """
            GETリクエストを送信するクライアントメソッド

            - API に対して GET リクエストを送信し、レスポンスを取得する
            - データの取得やリソースの作成に使用

            引数:
                url (str): リクエストを送信するエンドポイントのURL
                params (Optional[dict]): クエリーパラメータ デフォルトは None
                headers (dict, optional): リクエストヘッダ
                （デフォルト: "Content-Type: application/json"）

            戻り値:
                Response: FastAPIのレスポンスオブジェクト
        """
        return self.client.get(
            url,
            params=params,
            headers=headers
        )

    def post(
        self,
        url: str,
        data: dict,
        headers: dict = {"Content-Type": "application/json"}
    ) -> Response:
        """
            POSTリクエストを送信するクライアントメソッド

            - API に対して POST リクエストを送信し、レスポンスを取得する
            - 新規データの登録やリソースの作成に使用

            引数:
                url (str): リクエストを送信するエンドポイントのURL
                data (dict): リクエストボディデータ
                headers (dict, optional): リクエストヘッダ
                （デフォルト: "Content-Type: application/json"）

            戻り値:
                Response: FastAPIのレスポンスオブジェクト
        """
        return self.client.post(
            url,
            json=data,
            headers=headers
        )

    def put(
        self,
        url: str,
        data: dict,
        headers: dict = {"Content-Type": "application/json"}
    ) -> Response:
        """
            PUTリクエストを送信するクライアントメソッド

            - API に対して PUT リクエストを送信し、レスポンスを取得する
            - データの更新やリソースの作成に使用

            引数:
                url (str): リクエストを送信するエンドポイントのURL
                data (dict): リクエストボディデータ
                headers (dict, optional): リクエストヘッダ
                （デフォルト: "Content-Type: application/json"）

            戻り値:
                Response: FastAPIのレスポンスオブジェクト
        """
        return self.client.put(
            url,
            json=data,
            headers=headers
        )

    def delete(
        self,
        url: str,
        params: Optional[str] = None,
        headers: dict = {"Content-Type": "application/json"}
    ) -> Response:
        """
            DELETEリクエストを送信するクライアントメソッド

            - API に対して DELETE リクエストを送信し、レスポンスを取得する
            - データの削除やリソースの作成に使用

            引数:
                url (str): リクエストを送信するエンドポイントのURL
                params (str): リクエストパラメータ デフォルト None
                headers (dict, optional): リクエストヘッダ
                （デフォルト: "Content-Type: application/json"）

            戻り値:
                Response: FastAPIのレスポンスオブジェクト
        """
        return self.client.delete(
            url,
            params=params,
            headers=headers
        )
