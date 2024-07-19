import json
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient # type: ignore

from api.main import app
from api.models.models import BrandModel, BrandInfoModel, PredictionResultModel
from api.endpoints.stock_price import StockPriceBase, StockPriceService
from api.databases.databases import get_db
from const.const import HttpStatusCode, ErrorCode, PredictionResultConst, BrandInfoModelConst
from tests.api.database.test_database import get_test_db, init_db, drop_db

# モック
CHECK_BRAND_INFO = 'prediction.train.train.PredictionTrain.check_brand_info'
TRAIN_MAIN = 'prediction.train.train.PredictionTrain.main'
TEST_MAIN = 'prediction.test.test.PredictionTest.main'
STOCK_PRICE_BASE_PREDICTION = 'api.endpoints.stock_price.StockPriceBase.prediction'
GET_TEST_DB = 'tests.api.database.test_database.get_test_db'
GET_DB = 'api.databases.databases.get_db'

GET_STOCK_PRICE_PATH = "/get_stock_price"
GET_BRAND_INFO_LIST_PATH = "/brand_info_list"
GET_BRAND_PATH = "/brand"
CREATE_STOCK_PRICE_PATH = "/create_stock_price"
UPDATE_STOCK_PRICE_PATH = "/upadte_stock_price"
DELETE_BRAND_INFO_PATH = "/delete_stock_price"

# テストデータ
data = [100.0, 200.0, 300.0]
days = ["2023-07-01", "2023-07-02", "2023-07-03"]

# テスト項目のこり
# request_bodyのチェック、prediction
# GETにtimeout処理追加してテストする



def raise_db_error(*args, **kwargs):
    raise Exception("Database connection error")


class TestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # テスト用データベースの初期化
        init_db()

    @classmethod
    def tearDownClass(cls):
        # テスト用データベースの削除
        drop_db()

    def setUp(self):
        self.client = TestClient(app)
        self.db = next(get_test_db())
        app.dependency_overrides[get_db] = get_test_db

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    def params_error_check(self, code, msg, params_list, input_data_list, res_data):
        """
        パラメータのエラーチェック

        例: res = [
            {
                "code": ErrorCode.INT_VAILD.value,
                "detail": f"{loc[1]} パラメータは整数値のみです。",
                "input": input_msg
            }
        ]
        """
        self.assertEqual(len(res_data), len(params_list))
        for res, param, input_data in zip(res_data, params_list, input_data_list):
            self.assertEqual(res.get("code"), code)
            self.assertEqual(res.get("detail"), f"{param} {msg}")
            self.assertEqual(res.get("input"), input_data)

    def request_body_error_check(self, code, msg, res_data):
        """
        リクエストボディーのエラーチェック

        例: res = {
            "detail": [
                {
                    "code": 10,
                    "message": "既に登録されています"
                }
            ]
        }
        """
        detail = res_data.get("detail")
        self.assertEqual(detail[0].get("code"), code)
        self.assertEqual(detail[0].get("message"), msg)

    def delete_client(self, url, data):
        response = self.client.request(
            method="DELETE",
            url=url,
            headers={"Content-Type": "application/json"},
            content=json.dumps(data)
        )
        return response


class TestGetStockPrice(TestBase):

    def test_get_stock_price_success_01(self):
        """正常系: 予測データ取得API 取得データを返す"""
        # データセット
        params = {
            PredictionResultConst.BRAND_CODE.value: 1234,
            PredictionResultConst.USER_ID.value: 1
        }
        add_db_data_list = []
        data_list = [
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 1234, 1, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 3212, 2, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 1111, 1, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2345, 1, False],
        ]
        for i in data_list:
            add_db_data_list.append(PredictionResultModel(
                future_predictions=i[0],
                days_list=i[1],
                brand_code=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list)
        self.db.commit()

        # API実行
        response = self.client.get(GET_STOCK_PRICE_PATH, params=params)

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        expected_response = {
            "future_predictions": [100.0, 101.0, 102.0],
            "days_list": ["2024-07-01", "2024-07-02", "2024-07-03"],
            "brand_code": 1234,
            "user_id": 1
        }
        self.assertEqual(response.json(), expected_response)

    def test_get_stock_price_key_error(self):
        """異常系: 予測データ取得API 404 エラーチェック 存在しない銘柄"""
        params = {
            PredictionResultConst.BRAND_CODE.value: 9999,
            PredictionResultConst.USER_ID.value: 1
        }
        response = self.client.get(GET_STOCK_PRICE_PATH, params=params)
        self.assertEqual(response.status_code, HttpStatusCode.NOT_FOUND.value)
        self.assertEqual(response.json(), {
            "detail": [
                {
                    "code": ErrorCode.CHECK_EXIST.value,
                    "message": "登録されてない予測データです"
                }
            ]
        })

    def test_get_stock_price_validation_error_01(self):
        """異常系: 予測データ取得API 銘柄コード バリデーションエラー"""
        params = {
            PredictionResultConst.BRAND_CODE.value: "123.4",
            PredictionResultConst.USER_ID.value: 1
        }

        response = self.client.get(GET_STOCK_PRICE_PATH, params=params)
        self.assertEqual(response.status_code, HttpStatusCode.VALIDATION.value)
        self.params_error_check(
            ErrorCode.INT_VAILD.value,
            "パラメータは整数値のみです。",
            [
                PredictionResultConst.BRAND_CODE.value
            ],
            [
                "123.4"
            ],
            response.json()
        )

    def test_get_stock_price_validation_error_02(self):
        """異常系: 予測データ取得API ユーザーid バリデーションエラー"""
        params = {
            PredictionResultConst.BRAND_CODE.value: 1234,
            PredictionResultConst.USER_ID.value: "hoge"
        }

        response = self.client.get(GET_STOCK_PRICE_PATH, params=params)
        self.assertEqual(response.status_code, HttpStatusCode.VALIDATION.value)
        self.params_error_check(
            ErrorCode.INT_VAILD.value,
            "パラメータは整数値のみです。",
            [
                PredictionResultConst.USER_ID.value
            ],
            [
                "hoge"
            ],
            response.json()
        )

    def test_get_stock_price_validation_error_03(self):
        """異常系: 予測データ取得API brannd_codeとユーザーid バリデーションエラー"""
        params = {
            PredictionResultConst.BRAND_CODE.value: "45hoge",
            PredictionResultConst.USER_ID.value: "fuga"
        }

        response = self.client.get(GET_STOCK_PRICE_PATH, params=params)
        self.assertEqual(response.status_code, HttpStatusCode.VALIDATION.value)
        self.params_error_check(
            ErrorCode.INT_VAILD.value,
            "パラメータは整数値のみです。",
            [
                PredictionResultConst.BRAND_CODE.value,
                PredictionResultConst.USER_ID.value
            ],
            [
                "45hoge",
                "fuga"
            ],
            response.json()
        )


class TestBrandList(TestBase):

    def test_get_brand_list_success_01(self):
        """正常系: 対象ユーザーの学習ずみ銘柄情報取得API 取得データを返す"""
        # データセット
        params = {
            PredictionResultConst.USER_ID.value: 1
        }
        add_db_data_list = []
        data_list = [
            ["test1", 1111, "test1.pth", 1, True],
            ["test2", 2222, "test2.pth", 2, True],
            ["test3", 5436, "test3.pth", 1, True],
            ["test4", 2234, "test4.pth", 1, False],
        ]
        for i in data_list:
            add_db_data_list.append(BrandInfoModel(
                brand_name=i[0],
                brand_code=i[1],
                learned_model_name=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list)
        self.db.commit()

        # API実行
        response = self.client.get(GET_BRAND_INFO_LIST_PATH, params=params)

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        expected_response = [
            {
                "brand_info_id": 1,
                "brand_name": "test1",
                "brand_code": 1111,
                "learned_model_name": "test1.pth",
                "user_id": 1
            },
            {
                "brand_info_id": 3,
                "brand_name": "test3",
                "brand_code": 5436,
                "learned_model_name": "test3.pth",
                "user_id": 1
            },
        ]
        self.assertEqual(response.json(), expected_response)

    def test_get_brand_list_success_02(self):
        """正常系: 対象ユーザーの学習ずみ銘柄情報取得API 404 存在しないユーザーidは空で返す"""
        params = {
            PredictionResultConst.USER_ID.value: 6
        }
        response = self.client.get(GET_BRAND_INFO_LIST_PATH, params=params)
        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), [])

    def test_get_brand_list_validation_error_01(self):
        """異常系: 対象ユーザーの学習ずみ銘柄情報取得API バリデーションエラー"""
        params = {
            PredictionResultConst.USER_ID.value: "hoge"
        }

        response = self.client.get(GET_BRAND_INFO_LIST_PATH, params=params)
        self.assertEqual(response.status_code, HttpStatusCode.VALIDATION.value)
        self.params_error_check(
            ErrorCode.INT_VAILD.value,
            "パラメータは整数値のみです。",
            [
                PredictionResultConst.USER_ID.value
            ],
            [
                "hoge"
            ],
            response.json()
        )


class TestBrand(TestBase):

    def test_get_brand_success_01(self):
        """正常系: 全ての銘柄取得API 取得データを返す"""
        # データセット
        add_db_data_list = []
        data_list = [
            ["test1", 1111],
            ["test2", 2222],
            ["test3", 5436],
            ["test4", 2234],
        ]
        for i in data_list:
            add_db_data_list.append(BrandModel(
                brand_name=i[0],
                brand_code=i[1],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=True,
            ))
        self.db.add_all(add_db_data_list)
        self.db.commit()

        # API実行
        response = self.client.get(GET_BRAND_PATH)

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        expected_response = [
            {
                "brand_name": "test1",
                "brand_code": 1111
            },
            {
                "brand_name": "test2",
                "brand_code": 2222
            },
            {
                "brand_name": "test3",
                "brand_code": 5436
            },
            {
                "brand_name": "test4",
                "brand_code": 2234
            },
        ]
        self.assertEqual(response.json(), expected_response)


class TestCreateStockPrice(TestBase):

    @patch(STOCK_PRICE_BASE_PREDICTION)
    def test_create_stock_price_success_01(self, _stock_price_base_prediction):
        """正常系: 予測データ登録API 正しく登録できる"""
        # データセット
        data = {
            "brand_name": "日本ハム",
            "brand_code": 2282,
            "user_id": 3,
            "create_by": "test",
            "update_by": "test",
            "is_valid": True
        }

        _stock_price_base_prediction.return_value = (
            "['100.1', '200.2', '300.6']",
            "['2024-07-16', '2024-07-17', '2024-07-18']",
            "test.pth"
        )
        # API実行
        response = self.client.post(CREATE_STOCK_PRICE_PATH, json=data)

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)

        result_db_1 = self.db.query(BrandInfoModel).filter(
            BrandInfoModel.brand_code == data.get(BrandInfoModelConst.BRAND_CODE.value),
            BrandInfoModel.user_id == data.get(BrandInfoModelConst.USER_ID.value),
            BrandInfoModel.is_valid
        ).first()
        result_db_count_1 = self.db.query(BrandInfoModel).filter(
            BrandInfoModel.brand_code == data.get(BrandInfoModelConst.BRAND_CODE.value),
            BrandInfoModel.user_id == data.get(BrandInfoModelConst.USER_ID.value),
            BrandInfoModel.is_valid
        ).count()

        result_response_1 = {
            BrandInfoModelConst.BRAND_NAME.value: result_db_1.brand_name,
            BrandInfoModelConst.BRAND_CODE.value: result_db_1.brand_code,
            BrandInfoModelConst.LEARNED_MODEL_NAME.value: result_db_1.learned_model_name,
            BrandInfoModelConst.USER_ID.value: result_db_1.user_id
        }
        expected_response_1 = {
            BrandInfoModelConst.BRAND_NAME.value: "日本ハム",
            BrandInfoModelConst.BRAND_CODE.value: 2282,
            BrandInfoModelConst.LEARNED_MODEL_NAME.value: "test.pth",
            BrandInfoModelConst.USER_ID.value: 3
        }

        result_db_2 = self.db.query(PredictionResultModel).filter(
            PredictionResultModel.brand_code == data.get(PredictionResultConst.BRAND_CODE.value),
            PredictionResultModel.user_id == data.get(PredictionResultConst.USER_ID.value),
            PredictionResultModel.is_valid
        ).first()
        result_db_count_2 = self.db.query(PredictionResultModel).filter(
            PredictionResultModel.brand_code == data.get(PredictionResultConst.BRAND_CODE.value),
            PredictionResultModel.user_id == data.get(PredictionResultConst.USER_ID.value),
            PredictionResultModel.is_valid
        ).count()

        result_response_2 = {
            PredictionResultConst.FUTURE_PREDICTIONS.value: result_db_2.future_predictions,
            PredictionResultConst.DAYS_LIST.value: result_db_2.days_list,
            PredictionResultConst.BRAND_CODE.value: result_db_2.brand_code,
            PredictionResultConst.USER_ID.value: result_db_2.user_id
        }
        expected_response_2 = {
            PredictionResultConst.FUTURE_PREDICTIONS.value: "['100.1', '200.2', '300.6']",
            PredictionResultConst.DAYS_LIST.value: "['2024-07-16', '2024-07-17', '2024-07-18']",
            PredictionResultConst.BRAND_CODE.value: 2282,
            PredictionResultConst.USER_ID.value: 3
        }

        self.assertEqual(result_db_count_1, 1)
        self.assertEqual(result_db_count_2, 1)
        self.assertEqual(result_response_1, expected_response_1)
        self.assertEqual(result_response_2, expected_response_2)

    def test_create_stock_price_failed_duplication_check_01(self):
        """異常系: 重複した銘柄情報がある場合、409エラーを返す"""
        # 先にデータを登録する
        add_db_data_list = []
        data_list = [
            ["test1", 2768, "test1.pth", 1, True],
            ["test2", 2229, "test2.pth", 2, True],
            ["test3", 5469, "test3.pth", 1, True],
            ["test4", 2221, "test4.pth", 1, False],
        ]
        for i in data_list:
            add_db_data_list.append(BrandInfoModel(
                brand_name=i[0],
                brand_code=i[1],
                learned_model_name=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list)
        self.db.commit()

        # データセット
        data = {
            "brand_name": "test1",
            "brand_code": 2768,
            "user_id": 1,
            "create_by": "test",
            "update_by": "test",
            "is_valid": True
        }

        # API実行
        response = self.client.post(CREATE_STOCK_PRICE_PATH, json=data)

        self.assertEqual(response.status_code, HttpStatusCode.CONFLICT.value)
        self.request_body_error_check(
            ErrorCode.CHECK_EXIST.value,
            "銘柄情報は既に登録済みです。",
            response.json()
        )

    @patch(STOCK_PRICE_BASE_PREDICTION)
    def test_create_stock_price_failed_duplication_check_02(self, _stock_price_base_prediction):
        """異常系: 重複した予測結果データがある場合、409エラーを返す"""
        # あらかじめデータを登録する
        add_db_data_list = []
        data_list = [
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 1234, 1, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2413, 3, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 1111, 1, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2345, 1, False],
        ]
        for i in data_list:
            add_db_data_list.append(PredictionResultModel(
                future_predictions=i[0],
                days_list=i[1],
                brand_code=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list)
        self.db.commit()

        # データセット
        data = {
            "brand_name": "test1",
            "brand_code": 2413,
            "user_id": 3,
            "create_by": "test",
            "update_by": "test",
            "is_valid": True
        }
        _stock_price_base_prediction.return_value = (
            "['100.1', '200.2', '300.6']",
            "['2024-07-16', '2024-07-17', '2024-07-18']",
            "test.pth"
        )

        # API実行
        response = self.client.post(CREATE_STOCK_PRICE_PATH, json=data)

        self.assertEqual(response.status_code, HttpStatusCode.CONFLICT.value)
        self.request_body_error_check(
            ErrorCode.CHECK_EXIST.value,
            "予測結果データは既に登録済みです。",
            response.json()
        )


class TestUpdateStockPrice(TestBase):

    @patch(STOCK_PRICE_BASE_PREDICTION)
    def test_update_stock_price_success_01(self, _stock_price_base_prediction):
        """正常系: 予測データ更新API 正しく更新できる"""
        # 先にデータを登録する
        add_db_data_list_1 = []
        data_list_1 = [
            ["test1", 3401, "test1.pth", 3, True],
            ["test2", 2229, "test2.pth", 2, True],
            ["test3", 5469, "test3.pth", 1, True],
            ["test4", 2221, "test4.pth", 1, False],
        ]
        for i in data_list_1:
            add_db_data_list_1.append(BrandInfoModel(
                brand_name=i[0],
                brand_code=i[1],
                learned_model_name=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list_1)
        self.db.commit()

        add_db_data_list_2 = []
        data_list_2 = [
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 3401, 3, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2229, 2, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 5469, 1, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2221, 1, False],
        ]
        for i in data_list_2:
            add_db_data_list_2.append(PredictionResultModel(
                future_predictions=i[0],
                days_list=i[1],
                brand_code=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list_2)
        self.db.commit()

        # データセット
        data = {
            "brand_name": "test1",
            "brand_code": 3401,
            "update_by": "test_update",
            "user_id": 3,
        }

        _stock_price_base_prediction.return_value = (
            "['101.1', '202.2', '303.6']",
            "['2024-07-20', '2024-07-21', '2024-07-22']",
            "test_update.pth"
        )
        # API実行
        response = self.client.put(UPDATE_STOCK_PRICE_PATH, json=data)

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)

        result_db_1 = self.db.query(BrandInfoModel).filter(
            BrandInfoModel.brand_code == data.get(BrandInfoModelConst.BRAND_CODE.value),
            BrandInfoModel.user_id == data.get(BrandInfoModelConst.USER_ID.value),
            BrandInfoModel.is_valid
        ).first()
        result_db_count_1 = self.db.query(BrandInfoModel).filter(
            BrandInfoModel.brand_code == data.get(BrandInfoModelConst.BRAND_CODE.value),
            BrandInfoModel.user_id == data.get(BrandInfoModelConst.USER_ID.value),
            BrandInfoModel.is_valid
        ).count()

        result_response_1 = {
            BrandInfoModelConst.BRAND_NAME.value: result_db_1.brand_name,
            BrandInfoModelConst.BRAND_CODE.value: result_db_1.brand_code,
            BrandInfoModelConst.LEARNED_MODEL_NAME.value: result_db_1.learned_model_name,
            BrandInfoModelConst.USER_ID.value: result_db_1.user_id
        }
        expected_response_1 = {
            BrandInfoModelConst.BRAND_NAME.value: "test1",
            BrandInfoModelConst.BRAND_CODE.value: 3401,
            BrandInfoModelConst.LEARNED_MODEL_NAME.value: "test_update.pth",
            BrandInfoModelConst.USER_ID.value: 3
        }

        result_db_2 = self.db.query(PredictionResultModel).filter(
            PredictionResultModel.brand_code == data.get(PredictionResultConst.BRAND_CODE.value),
            PredictionResultModel.user_id == data.get(PredictionResultConst.USER_ID.value),
            PredictionResultModel.is_valid
        ).first()
        result_db_count_2 = self.db.query(PredictionResultModel).filter(
            PredictionResultModel.brand_code == data.get(PredictionResultConst.BRAND_CODE.value),
            PredictionResultModel.user_id == data.get(PredictionResultConst.USER_ID.value),
            PredictionResultModel.is_valid
        ).count()

        result_response_2 = {
            PredictionResultConst.FUTURE_PREDICTIONS.value: result_db_2.future_predictions,
            PredictionResultConst.DAYS_LIST.value: result_db_2.days_list,
            PredictionResultConst.BRAND_CODE.value: result_db_2.brand_code,
            PredictionResultConst.USER_ID.value: result_db_2.user_id
        }
        expected_response_2 = {
            PredictionResultConst.FUTURE_PREDICTIONS.value: "['101.1', '202.2', '303.6']",
            PredictionResultConst.DAYS_LIST.value: "['2024-07-20', '2024-07-21', '2024-07-22']",
            PredictionResultConst.BRAND_CODE.value: 3401,
            PredictionResultConst.USER_ID.value: 3
        }

        self.assertEqual(result_db_count_1, 1)
        self.assertEqual(result_db_count_2, 1)
        self.assertEqual(result_response_1, expected_response_1)
        self.assertEqual(result_response_2, expected_response_2)

    def test_update_stock_price_failed_not_exists_check_01(self):
        """異常系: 更新対象の銘柄データが存在しない場合、404エラーを返す"""
        # 先にデータを登録する
        add_db_data_list = []
        data_list = [
            ["test1", 2768, "test1.pth", 1, True],
            ["test2", 2229, "test2.pth", 2, True],
            ["test3", 5469, "test3.pth", 1, True],
            ["test4", 2221, "test4.pth", 1, False],
        ]
        for i in data_list:
            add_db_data_list.append(BrandInfoModel(
                brand_name=i[0],
                brand_code=i[1],
                learned_model_name=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list)
        self.db.commit()

        # データセット
        data_1 = {
            "brand_name": "test1",
            "brand_code": 9999,
            "user_id": 1,
            "update_by": "test_update",
        }
        data_2 = {
            "brand_name": "test4",
            "brand_code": 2221,
            "user_id": 1,
            "update_by": "test_update",
        }
        data_3 = {
            "brand_name": "test1",
            "brand_code": 2768,
            "user_id": 2,
            "update_by": "test_update",
        }

        data_list = [
            data_1,
            data_2,
            data_3,
        ]

        for data in data_list:
            # API実行
            response = self.client.put(UPDATE_STOCK_PRICE_PATH, json=data)

            # brand_codeが存在しないかつis_vaild = True
            self.assertEqual(response.status_code, HttpStatusCode.NOT_FOUND.value)
            self.request_body_error_check(
                ErrorCode.NOT_DATA.value,
                "更新対象の銘柄データが存在しません。",
                response.json()
            )

    @patch(STOCK_PRICE_BASE_PREDICTION)
    def test_update_stock_price_failed_not_exists_check_02(self, _stock_price_base_prediction):
        """異常系: 更新対象の予測結果データが存在しない場合、404エラーを返す"""
        # あらかじめデータを登録する
        add_db_data_list_1 = []
        data_list_1 = [
            ["test1", 2768, "test1.pth", 1, True],
            ["test2", 2229, "test2.pth", 3, True],
            ["test3", 1111, "test3.pth", 1, True],
            ["test4", 2221, "test4.pth", 1, False],
        ]
        for i in data_list_1:
            add_db_data_list_1.append(BrandInfoModel(
                brand_name=i[0],
                brand_code=i[1],
                learned_model_name=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list_1)
        self.db.commit()

        add_db_data_list_2 = []
        data_list_2 = [
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 1234, 1, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2229, 4, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 1111, 1, False],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2345, 1, False],
        ]
        for i in data_list_2:
            add_db_data_list_2.append(PredictionResultModel(
                future_predictions=i[0],
                days_list=i[1],
                brand_code=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list_2)
        self.db.commit()

        # データセット
        data_1 = {
            "brand_name": "test1",
            "brand_code": 2768,
            "user_id": 1,
            "update_by": "test_update"
        }
        data_2 = {
            "brand_name": "test1",
            "brand_code": 2229,
            "user_id": 3,
            "update_by": "test_update"
        }
        data_3 = {
            "brand_name": "test1",
            "brand_code": 1111,
            "user_id": 1,
            "update_by": "test_update"
        }

        data_list = [
            data_1,
            data_2,
            data_3,
        ]

        for data in data_list:
            _stock_price_base_prediction.return_value = (
                "['100.1', '200.2', '300.6']",
                "['2024-07-16', '2024-07-17', '2024-07-18']",
                "test.pth"
            )
            # API実行
            response = self.client.put(UPDATE_STOCK_PRICE_PATH, json=data)

            self.assertEqual(response.status_code, HttpStatusCode.NOT_FOUND.value)
            self.request_body_error_check(
                ErrorCode.NOT_DATA.value,
                "更新対象の予測結果データが存在しません。",
                response.json()
            )


class TestDeleteStockPrice(TestBase):

    def test_delete_stock_price_success_01(self):
        """正常系: 予測データ削除API 正しく登録できる"""
        # 先にデータを登録する
        add_db_data_list_1 = []
        data_list_1 = [
            ["test1", 3401, "test1.pth", 3, True],
            ["test2", 2229, "test2.pth", 2, True],
            ["test3", 8989, "test3.pth", 10, True],
            ["test4", 2221, "test4.pth", 1, False],
        ]
        for i in data_list_1:
            add_db_data_list_1.append(BrandInfoModel(
                brand_name=i[0],
                brand_code=i[1],
                learned_model_name=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list_1)
        self.db.commit()

        add_db_data_list_2 = []
        data_list_2 = [
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 3401, 3, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2229, 2, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 8989, 10, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2221, 1, False],
        ]
        for i in data_list_2:
            add_db_data_list_2.append(PredictionResultModel(
                future_predictions=i[0],
                days_list=i[1],
                brand_code=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list_2)
        self.db.commit()

        # データセット
        data = {
            "brand_code": 8989,
            "user_id": 10
        }

        # API実行
        response = self.delete_client(DELETE_BRAND_INFO_PATH, data)

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)

        result_db_count_1 = self.db.query(BrandInfoModel).filter(
            BrandInfoModel.brand_code == data.get(BrandInfoModelConst.BRAND_CODE.value),
            BrandInfoModel.user_id == data.get(BrandInfoModelConst.USER_ID.value),
            BrandInfoModel.is_valid
        ).count()

        result_db_count_2 = self.db.query(PredictionResultModel).filter(
            PredictionResultModel.brand_code == data.get(PredictionResultConst.BRAND_CODE.value),
            PredictionResultModel.user_id == data.get(PredictionResultConst.USER_ID.value),
            PredictionResultModel.is_valid
        ).count()

        self.assertEqual(result_db_count_1, 0)
        self.assertEqual(result_db_count_2, 0)

    def test_delete_stock_price_failed_not_exists_check_01(self):
        """異常系: 削除対象の銘柄情報が見つからない場合、404エラーを返す"""
        # 先にデータを登録する
        add_db_data_list_1 = []
        data_list_1 = [
            ["test1", 3401, "test1.pth", 3, True],
            ["test2", 2229, "test2.pth", 2, True],
            ["test3", 5469, "test3.pth", 1, True],
            ["test4", 2221, "test4.pth", 1, False],
        ]
        for i in data_list_1:
            add_db_data_list_1.append(BrandInfoModel(
                brand_name=i[0],
                brand_code=i[1],
                learned_model_name=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list_1)
        self.db.commit()

        add_db_data_list_2 = []
        data_list_2 = [
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 3401, 3, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2229, 2, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 5469, 1, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 2221, 1, False],
        ]
        for i in data_list_2:
            add_db_data_list_2.append(PredictionResultModel(
                future_predictions=i[0],
                days_list=i[1],
                brand_code=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list_2)
        self.db.commit()

        # データセット
        data_1 = {
            "brand_code": 9999,
            "user_id": 1
        }
        data_2 = {
            "brand_code": 2221,
            "user_id": 1
        }

        data_list = [
            data_1,
            data_2
        ]

        for data in data_list:
            # API実行
            response = self.delete_client(DELETE_BRAND_INFO_PATH, data)

            self.assertEqual(response.status_code, HttpStatusCode.NOT_FOUND.value)
            self.request_body_error_check(
                ErrorCode.NOT_DATA.value,
                "削除対象の銘柄情報が見つかりません。",
                response.json()
            )

    def test_delete_stock_price_failed_not_exists_check_02(self):
        """異常系: 削除対象の予測結果データが見つからない場合、404エラーを返す"""
        # 先にデータを登録する
        add_db_data_list_1 = []
        data_list_1 = [
            ["test1", 3410, "test1.pth", 11, True],
            ["test2", 9269, "test2.pth", 2, True],
            ["test3", 3421, "test3.pth", 1, True],
            ["test4", 1123, "test4.pth", 1, False],
        ]
        for i in data_list_1:
            add_db_data_list_1.append(BrandInfoModel(
                brand_name=i[0],
                brand_code=i[1],
                learned_model_name=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list_1)
        self.db.commit()

        add_db_data_list_2 = []
        data_list_2 = [
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 3411, 11, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 9269, 2, True],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 3421, 1, False],
            ["[100.0,101.0,102.0]", "[2024-07-01,2024-07-02,2024-07-03]", 1123, 1, False],
        ]
        for i in data_list_2:
            add_db_data_list_2.append(PredictionResultModel(
                future_predictions=i[0],
                days_list=i[1],
                brand_code=i[2],
                user_id=i[3],
                create_at=StockPriceBase.get_jst_now(),
                create_by="test_user",
                update_at=StockPriceBase.get_jst_now(),
                update_by="test_user",
                is_valid=i[4],
            ))
        self.db.add_all(add_db_data_list_2)
        self.db.commit()

        # データセット
        data_1 = {
            "brand_code": 3410,
            "user_id": 11
        }
        data_2 = {
            "brand_code": 3421,
            "user_id": 1
        }

        data_list = [
            data_1,
            data_2
        ]

        for data in data_list:
            # API実行
            response = self.delete_client(DELETE_BRAND_INFO_PATH, data)

            self.assertEqual(response.status_code, HttpStatusCode.NOT_FOUND.value)
            self.request_body_error_check(
                ErrorCode.NOT_DATA.value,
                "削除対象の予測結果データが見つかりません。",
                response.json()
            )
