from typing import List
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from api.common.exceptions import NotFoundException
from api.models.models import BrandModel, BrandInfoModel, PredictionResultModel
from api.schemas.response import Content
from api.schemas.stock_price import (
    BrandInfoListResponse,
    BrandInfoResponse,
    CreateBrandInfoRequest,
    PredictionResultResponse,
    UpdateBrandInfoRequest
)
from api.schemas.validation import ValidatonModel
from api.usercase.stock_price import StockPriceService
from const.const import HttpStatusCode

from tests.test_case import TestBaseAPI

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


class TestGetPredictionData(TestEndpointBase):

    def get_path(self, user_id: int, brand_code: int) -> str:
        """
        予測データ取得APIのパス

        引数:
            user_id (int): ユーザーid
            brand_code (int): 銘柄コード

        戻り値:
            str: urlパス
        """
        return self.get_stock_price_path(
            f'get_prediction_data/{user_id}/{brand_code}'
        )

    def test_get_prediction_data_valid_error_01(self):
        """
        正常系: バリデーションエラー
        - user_id必須チェック
        """
        # API実行
        response = self.client.get(self.get_path(0, 1111))
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="user_id",
                    message="user_idは必須です。"
                )
            ]
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    def test_get_prediction_data_valid_error_02(self):
        """
        正常系: バリデーションエラー
        - brand_code必須チェック
        """
        # API実行
        response = self.client.get(self.get_path(1, 0))
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="brand_code",
                    message="brand_codeは必須です。"
                )
            ]
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    def test_get_prediction_data_valid_error_03(self):
        """
        正常系: バリデーションエラー
        - brand_code範囲チェック
        """
        # データセット
        path_list = [
            (1, 999),
            (1, 10000)
        ]

        for user_id, brand_code in path_list:
            # API実行
            response = self.client.get(self.get_path(user_id, brand_code))
            self.assertEqual(
                response.status_code,
                HttpStatusCode.BADREQUEST.value
            )
            expected_response = Content[list[ValidatonModel]](
                result=[
                    ValidatonModel(
                        field="brand_code",
                        message="1000 から 9999 の範囲で入力してください。"
                    )
                ]
            )
            self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(StockPriceService, "get_prediction_info")
    def test_get_prediction_data_error_01(self, _get_prediction_info):
        """
        異常系: 登録されてない予測データを取得しようとした場合
        """
        # API実行
        _get_prediction_info.return_value = None
        response = self.client.get(self.get_path(1, 1234))

        expected_response = Content[str](
            result="登録されてない予測データです。"
        )

        self.assertEqual(response.status_code, HttpStatusCode.NOT_FOUND.value)
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(StockPriceService, "get_prediction_info")
    def test_get_prediction_data_error_02(self, _get_prediction_info):
        """
        異常系: 予期せぬエラー
        """
        # エラーモック
        _get_prediction_info.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.client.get(self.get_path(1, 1234))
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(response.status_code,
                         HttpStatusCode.SERVER_ERROR.value)
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(StockPriceService, "get_prediction_info")
    def test_get_prediction_data_success_01(self, _get_prediction_info):
        """
        正常系: 登録されてる予測データを取得できる
        """
        # API実行
        _get_prediction_info.return_value = PredictionResultModel(
            future_predictions="[1, 2, 3]",
            days_list="[2025-01-20, 2025-01-21, 2025-01-22]",
            brand_code=1234,
            user_id=1
        )
        response = self.client.get(self.get_path(1, 1234))

        expected_response = Content[PredictionResultResponse](
            result=PredictionResultResponse(
                future_predictions=[1, 2, 3],
                days_list=["2025-01-20", "2025-01-21", "2025-01-22"],
                brand_code=1234,
                user_id=1
            )
        )

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())


class TestBrandInfoList(TestEndpointBase):

    def get_path(self, user_id: int) -> str:
        """
        対象ユーザーの学習ずみ銘柄情報取得APIのパス

        引数:
            user_id (int): ユーザーid

        戻り値:
            str: urlパス
        """
        return self.get_stock_price_path(
            f'brand_info_list/{user_id}'
        )

    @patch.object(StockPriceService, "get_brand_list")
    def test_brand_info_list_error_01(self, _get_brand_list):
        """
        異常系: 予期せぬエラー
        """
        # エラーモック
        _get_brand_list.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.client.get(self.get_path(1))
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    def test_brand_info_list_valid_error_01(self):
        """
        正常系: バリデーションエラー
        - user_id必須チェック
        """
        # API実行
        response = self.client.get(self.get_path(0))
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="user_id",
                    message="user_idは必須です。"
                )
            ]
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(StockPriceService, "get_brand_list")
    def test_brand_info_list_success_01(self, _get_brand_list):
        """
        正常系: 対象ユーザーの学習ずみ銘柄情報取得できる(0件)
        """
        # API実行
        _get_brand_list.return_value = []
        response = self.client.get(self.get_path(1))

        expected_response = Content[List[BrandInfoListResponse]](
            result=[]
        )

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(StockPriceService, "get_brand_list")
    def test_brand_info_list_success_02(self, _get_brand_list):
        """
        正常系: 対象ユーザーの学習ずみ銘柄情報取得できる(2件)
        """
        # API実行
        _get_brand_list.return_value = [
            BrandInfoModel(
                brand_info_id=1,
                brand_name="hoge",
                brand_code=1234,
                learned_model_name="test1.pth",
                user_id=1
            ),
            BrandInfoModel(
                brand_info_id=2,
                brand_name="fuga",
                brand_code=5678,
                learned_model_name="test2.pth",
                user_id=1
            ),
        ]
        response = self.client.get(self.get_path(1))
        expected_response = Content[List[BrandInfoListResponse]](
            result=[
                BrandInfoListResponse(
                    brand_info_id=1,
                    brand_name="hoge",
                    brand_code=1234,
                    learned_model_name="test1.pth",
                    user_id=1
                ),
                BrandInfoListResponse(
                    brand_info_id=2,
                    brand_name="fuga",
                    brand_code=5678,
                    learned_model_name="test2.pth",
                    user_id=1
                )
            ]
        )

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())


class TestBrand(TestEndpointBase):

    def get_path(self) -> str:
        """
       全ての銘柄取得APIのパス

        引数:
            なし

        戻り値:
            str: urlパス
        """
        return self.get_stock_price_path(
            'brand'
        )

    @patch.object(StockPriceService, "get_brand_all_list")
    def test_brand_error_01(self, _get_brand_all_list):
        """
        異常系: 予期せぬエラー
        """
        # エラーモック
        _get_brand_all_list.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.client.get(self.get_path())
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(StockPriceService, "get_brand_all_list")
    def test_brand_success_01(self, _get_brand_all_list):
        """
        正常系: 銘柄取得できる(0件)
        """
        # API実行
        _get_brand_all_list.return_value = []
        response = self.client.get(self.get_path())

        expected_response = Content[List[BrandInfoResponse]](
            result=[]
        )

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(StockPriceService, "get_brand_all_list")
    def test_brand_success_02(self, _get_brand_all_list):
        """
        正常系: 全ての銘柄取得できる(2件)
        """
        # API実行
        _get_brand_all_list.return_value = [
            BrandModel(
                brand_name="hoge",
                brand_code=1234
            ),
            BrandModel(
                brand_name="fuga",
                brand_code=5678
            ),
        ]
        response = self.client.get(self.get_path())
        expected_response = Content[List[BrandInfoResponse]](
            result=[
                BrandInfoResponse(
                    brand_name="hoge",
                    brand_code=1234,
                ),
                BrandInfoResponse(
                    brand_name="fuga",
                    brand_code=5678,
                )
            ]
        )

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.response_body_check(
            response.json(), expected_response.model_dump())


class TestCreateStockPrice(TestEndpointBase):

    def get_path(self) -> str:
        """
        予測データ登録APIのパス

        引数:
            なし
        戻り値:
            str: urlパス
        """
        return self.get_stock_price_path(
            'create_stock_price'
        )

    def test_create_stock_price_valid_error_01(self):
        """
        正常系: バリデーションエラー
        - 以下項目の須チェック
            - brand_name
            - brand_code
            - user_id
            - create_by
            - update_by
        """
        # データセット
        data = CreateBrandInfoRequest(
            brand_name="",
            brand_code=0,
            user_id=0,
            create_by="",
            update_by="",
            is_valid=True
        ).model_dump()
        # API実行
        response = self.post(self.get_path(), data)
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="brand_name",
                    message="brand_nameは必須です。"
                ),
                ValidatonModel(
                    field="brand_code",
                    message="brand_codeは必須です。"
                ),
                ValidatonModel(
                    field="user_id",
                    message="user_idは必須です。"
                ),
                ValidatonModel(
                    field="create_by",
                    message="create_byは必須です。"
                ),
                ValidatonModel(
                    field="update_by",
                    message="update_byは必須です。"
                ),
            ]
        )
        self.response_body_check(
            response.json(), expected_response.model_dump())

    def test_create_stock_price_valid_error_02(self):
        """
        正常系: バリデーションエラー
        - brand_code範囲チェック
        """
        # データセット
        data_list = [
            CreateBrandInfoRequest(
                brand_name="test",
                brand_code=999,
                user_id=1,
                create_by="test",
                update_by="test",
                is_valid=True
            ).model_dump(),
            CreateBrandInfoRequest(
                brand_name="test",
                brand_code=10000,
                user_id=1,
                create_by="test",
                update_by="test",
                is_valid=True
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
                        field="brand_code",
                        message="1000 から 9999 の範囲で入力してください。"
                    )
                ]
            )
            self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(StockPriceService, "_create")
    def test_create_stock_price_error_01(self, _create):
        """
        異常系: 予期せぬエラー
        """
        # データセット
        data = CreateBrandInfoRequest(
            brand_name="test",
            brand_code=1234,
            user_id=1,
            create_by="test",
            update_by="test",
            is_valid=True
        ).model_dump()
        # エラーモック
        _create.side_effect = HTTPException(
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

    @patch.object(StockPriceService, "_create")
    def test_create_stock_price_success_01(self, _create):
        """
        異常系: 登録成功
        """
        # データセット
        data = {
            "brand_name": "test",
            "brand_code": 1234,
            "user_id": 1,
            "create_by": "test",
            "update_by": "test",
            "is_valid": True
        }
        _create.return_value = None
        # API実行
        response = self.post(self.get_path(), data)

        expected_response = Content[str](
            result="予測データの登録成功"
        )

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())


class TestUpdateStockPrice(TestEndpointBase):

    def get_path(self, user_id: int) -> str:
        """
        予測データ更新APIのパス

        引数:
            user_id (int): ユーザーid

        戻り値:
            str: urlパス
        """
        return self.get_stock_price_path(
            f'upadte_stock_price/{user_id}'
        )

    def test_upadte_stock_price_valid_error_01(self):
        """
        正常系: バリデーションエラー
        - 以下項目の須チェック
            - user_id
            - brand_name
            - brand_code
            - update_by
        """
        # データセット
        data = UpdateBrandInfoRequest(
            brand_name="",
            brand_code=0,
            update_by=""
        ).model_dump()
        # API実行
        response = self.put(self.get_path(0), data)
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="brand_name",
                    message="brand_nameは必須です。"
                ),
                ValidatonModel(
                    field="brand_code",
                    message="brand_codeは必須です。"
                ),
                ValidatonModel(
                    field="user_id",
                    message="user_idは必須です。"
                ),
                ValidatonModel(
                    field="update_by",
                    message="update_byは必須です。"
                ),
            ]
        )
        self.response_body_check(
            response.json(),
            expected_response.model_dump()
        )

    def test_upadte_stock_price_valid_error_02(self):
        """
        正常系: バリデーションエラー
        - brand_code範囲チェック
        """
        # データセット
        data_list = [
            UpdateBrandInfoRequest(
                brand_name="test",
                brand_code=999,
                update_by="test",
            ).model_dump(),
            UpdateBrandInfoRequest(
                brand_name="test",
                brand_code=10000,
                update_by="test",
            ).model_dump(),
        ]
        # API実行
        for data in data_list:
            response = self.put(self.get_path(1), data)
            self.assertEqual(
                response.status_code,
                HttpStatusCode.BADREQUEST.value
            )
            expected_response = Content[list[ValidatonModel]](
                result=[
                    ValidatonModel(
                        field="brand_code",
                        message="1000 から 9999 の範囲で入力してください。"
                    )
                ]
            )
            self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(StockPriceService, "_update")
    def test_upadte_stock_price_error_01(self, _update):
        """
        異常系: 予期せぬエラー
        """
        # データセット
        data = UpdateBrandInfoRequest(
            brand_name="test",
            brand_code=1234,
            update_by="test",
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

    @patch.object(StockPriceService, "_update")
    def test_upadte_stock_price_success_01(self, _update):
        """
        異常系: 更新成功
        """
        # データセット
        data = {
            "brand_name": "test",
            "brand_code": 1234,
            "update_by": "test",
        }
        _update.return_value = None
        # API実行
        response = self.put(self.get_path(1), data)

        expected_response = Content[str](
            result="予測データの更新成功"
        )

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())


class TestDeleteStockPrice(TestEndpointBase):

    def get_path(self, user_id: int, brand_code: int) -> str:
        """
        予測データ削除APIのパス

        引数:
            user_id (int): ユーザーid
            brand_code (int): 銘柄コード

        戻り値:
            str: urlパス
        """
        return self.get_stock_price_path(
            f'delete_stock_price/{user_id}/{brand_code}'
        )

    def test_delete_stock_price_valid_error_01(self):
        """
        正常系: バリデーションエラー
        - 以下項目の須チェック
            - user_id
            - brand_code
        """
        # API実行
        response = self.delete(self.get_path(0, 0))
        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        expected_response = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="user_id",
                    message="user_idは必須です。"
                ),
                ValidatonModel(
                    field="brand_code",
                    message="brand_codeは必須です。"
                )
            ]
        )
        self.response_body_check(
            response.json(),
            expected_response.model_dump()
        )

    def test_delete_stock_price_valid_error_02(self):
        """
        正常系: バリデーションエラー
        - brand_code範囲チェック
        """
        # データセット
        path_list = [
            (1, 999),
            (1, 10000)
        ]

        for user_id, brand_code in path_list:
            # API実行
            response = self.delete(self.get_path(user_id, brand_code))
            self.assertEqual(
                response.status_code,
                HttpStatusCode.BADREQUEST.value
            )
            expected_response = Content[list[ValidatonModel]](
                result=[
                    ValidatonModel(
                        field="brand_code",
                        message="1000 から 9999 の範囲で入力してください。"
                    )
                ]
            )
            self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(
        StockPriceService,
        "_brand_info_and_prediction_result_delete"
    )
    def test_delete_stock_price_error_01(self, _biaprd):
        """
        異常系: 予期せぬエラー
        """
        # エラーモック
        _biaprd.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.delete(self.get_path(1, 1234))
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(
        StockPriceService,
        "_brand_info_and_prediction_result_delete"
    )
    def test_delete_stock_price_error_02(self, _biaprd):
        """
        異常系: 削除対象の銘柄又は予測結果データが見つからなかった場合
        """
        # エラーモック
        _biaprd.side_effect = NotFoundException(
            "削除対象の銘柄情報が見つかりません。"
        )
        response = self.delete(self.get_path(1, 1234))
        expected_response = Content[str](
            result="削除対象の銘柄情報が見つかりません。"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.NOT_FOUND.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(
        StockPriceService,
        "_brand_info_and_prediction_result_delete"
    )
    def test_delete_stock_price_success_01(
        self,
        _biaprd
    ):
        """
        異常系: 削除成功
        """
        _biaprd.return_value = None
        # API実行
        response = self.delete(self.get_path(1, 1234))

        expected_response = Content[str](
            result="予測データの削除成功"
        )

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), expected_response.model_dump())
