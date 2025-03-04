import datetime
from typing import List, Literal

from api.schemas.response import Content
from api.schemas.stock_price import GetPredictionDataValidationModel
from api.schemas.test import TestValidationNodel
from api.schemas.validation import ValidatonModel
from api.validation.stock_price import GetPredictionDataValidation
from api.validation.validation import AbstractValidation, ValidationError
from tests.test_case import TestBase


class TestValidationError(TestBase):

    def _expected_result(self):
        """
        期待結果
        """
        return [
            ValidatonModel(
                field="test1",
                message="test1必須"
            ),
            ValidatonModel(
                field="test2",
                message="test2必須"
            ),
            ValidatonModel(
                field="test3",
                message="test3必須"
            )
        ]

    def test_valid_result_success_01(self):
        """
        正常系: バリデーション結果01
        """

        data = [
            ValidatonModel(
                field="test1",
                message="test1必須"
            ),
            ValidatonModel(
                field="test2",
                message="test2必須"
            ),
            True,
            ValidatonModel(
                field="test3",
                message="test3必須"
            ),
        ]

        self.assertEqual(
            ValidationError.valid_result(data),
            self._expected_result()
        )

    def test_valid_result_success_02(self):
        """
        正常系: バリデーション結果02
        """

        data = [
            [
                ValidatonModel(
                    field="test1",
                    message="test1必須"
                ),
                ValidatonModel(
                    field="test2",
                    message="test2必須"
                )
            ],
            True,
            ValidatonModel(
                field="test3",
                message="test3必須"
            ),
        ]

        self.assertEqual(
            ValidationError.valid_result(data),
            self._expected_result()
        )


class TestValidation(
    AbstractValidation[TestValidationNodel]
):
    """
    テスト用のバリデーション
    - カバレッジが通ってないやつのみ検証
    """

    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.test_float_01(),
            self.test_float_02(),
            self.test_float_03(),
            self.test_float_04(),
            self.test_str_01(),
            self.test_str_02(),
            self.test_datetime_01(),
            self.test_datetime_02(),
            self.test_datetime_03(),
            self.test_datetime_04(),
            self.test_date_01(),
            self.test_date_02(),
            self.test_date_03(),
            self.test_date_04(),
            self.test_time_01(),
            self.test_time_02(),
            self.test_time_03(),
            self.test_time_04(),
            self.test_bool_01(),
        ])

    ########################################
    # オーバーライドメソッド（バリデーション処理）#
    ########################################
    def test_float_01(self) -> ValidatonModel | Literal[True]:
        return self.validate_float(
            0.0,
            "float1"
        )

    def test_float_02(self) -> ValidatonModel | Literal[True]:
        return self.validate_float(
            9.99,
            "float2",
            10,
            20
        )

    def test_float_03(self) -> ValidatonModel | Literal[True]:
        return self.validate_float(
            20.1,
            "float3",
            10,
            20,
            False
        )

    def test_float_04(self) -> ValidatonModel | Literal[True]:
        return self.validate_float(
            19.9,
            "float4",
            10,
            20
        )

    def test_str_01(self) -> ValidatonModel | Literal[True]:
        return self.validate_str(
            "h",
            "str1",
            2,
            4
        )

    def test_str_02(self) -> ValidatonModel | Literal[True]:
        return self.validate_str(
            "hhhhh",
            "str2",
            2,
            4
        )

    def test_datetime_01(self) -> ValidatonModel | Literal[True]:
        return self.validate_datetime(
            datetime.datetime(2024, 1, 1, 23, 59, 59),
            "datetime1",
            datetime.datetime(2024, 1, 2, 0, 0, 0),
            datetime.datetime(2024, 2, 2, 0, 0, 0),
        )

    def test_datetime_02(self) -> ValidatonModel | Literal[True]:
        return self.validate_datetime(
            datetime.datetime(2024, 2, 2, 0, 0, 1),
            "datetime2",
            datetime.datetime(2024, 1, 2, 0, 0, 0),
            datetime.datetime(2024, 2, 2, 0, 0, 0),
            False
        )

    def test_datetime_03(self) -> ValidatonModel | Literal[True]:
        # テスト検証のため、あえて指定型定義以外の型を渡す
        return self.validate_datetime(
            None,  # type: ignore
            "datetime3"
        )

    def test_datetime_04(self) -> ValidatonModel | Literal[True]:
        return self.validate_datetime(
            datetime.datetime(2024, 2, 2, 0, 0, 0),
            "datetime4",
            datetime.datetime(2024, 1, 2, 0, 0, 0),
            datetime.datetime(2024, 2, 2, 0, 0, 0),
        )

    def test_date_01(self) -> ValidatonModel | Literal[True]:
        return self.validate_date(
            datetime.date(2023, 12, 31),
            "date1",
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 2),
        )

    def test_date_02(self) -> ValidatonModel | Literal[True]:
        return self.validate_date(
            datetime.date(2024, 1, 3),
            "date2",
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 2),
            False
        )

    def test_date_03(self) -> ValidatonModel | Literal[True]:
        # テスト検証のため、あえて指定型定義以外の型を渡す
        return self.validate_date(
            None,  # type: ignore
            "date3"
        )

    def test_date_04(self) -> ValidatonModel | Literal[True]:
        return self.validate_date(
            datetime.date(2024, 1, 2),
            "date4",
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 3),
        )

    def test_time_01(self) -> ValidatonModel | Literal[True]:
        return self.validate_time(
            datetime.time(0, 0, 0),
            "time1",
            datetime.time(0, 0, 1),
            datetime.time(23, 59, 58),
        )

    def test_time_02(self) -> ValidatonModel | Literal[True]:
        return self.validate_time(
            datetime.time(23, 59, 59),
            "time2",
            datetime.time(0, 0, 1),
            datetime.time(23, 59, 58),
            False
        )

    def test_time_03(self) -> ValidatonModel | Literal[True]:
        # テスト検証のため、あえて指定型定義以外の型を渡す
        return self.validate_time(
            None,  # type: ignore
            "time3"
        )

    def test_time_04(self) -> ValidatonModel | Literal[True]:
        return self.validate_time(
            datetime.time(23, 59, 57),
            "time4",
            datetime.time(0, 0, 1),
            datetime.time(23, 59, 58),
        )

    def test_bool_01(self) -> ValidatonModel | Literal[True]:
        # テスト検証のため、あえて指定型定義以外の型を渡す
        return self.validate_bool(
            None,  # type: ignore
            "bool"
        )


class TestAbstractValidation(TestBase):

    def _data(self) -> TestValidationNodel:
        return TestValidationNodel(
            v1=1,
            v2=1.0,
            v3="test",
            v4=datetime.datetime(2024, 3, 2, 15, 30, 0),
            v5=datetime.date(2024, 1, 1),
            v6=datetime.time(1, 0, 0),
            v7=False
        )

    def test__len__check_01(self):
        """
        正常系: 要素取得数2件
        """
        valid = GetPredictionDataValidation(
            GetPredictionDataValidationModel(
                user_id=0,
                brand_code=0
            )
        )

        self.assertEqual(len(valid), 2)

    def test__len__check_02(self):
        """
        正常系: 要素取得数0件
        """
        valid = GetPredictionDataValidation(
            GetPredictionDataValidationModel(
                user_id=1,
                brand_code=1234
            )
        )

        self.assertEqual(len(valid), 0)

    def test_validate_method_check_success_01(self):
        """
        正常系: 各バリデーションチェック
        """

        expected_valid = Content[list[ValidatonModel]](
            result=[
                ValidatonModel(
                    field="float1",
                    message="float1は必須です。"
                ),
                ValidatonModel(
                    field="float2",
                    message="10 から 20 の範囲で入力してください。"
                ),
                ValidatonModel(
                    field="float3",
                    message="10 から 20 の範囲で入力してください。"
                ),
                ValidatonModel(
                    field="str1",
                    message="2 文字以上 4 文字以下で入力してください。",
                ),
                ValidatonModel(
                    field="str2",
                    message="2 文字以上 4 文字以下で入力してください。",
                ),
                ValidatonModel(
                    field="datetime1",
                    message=(
                        "2024年1月2日 0時0分0秒 から "
                        "2024年2月2日 0時0分0秒 の範囲で入力してください。"
                    ),
                ),
                ValidatonModel(
                    field="datetime2",
                    message=(
                        "2024年1月2日 0時0分0秒 から "
                        "2024年2月2日 0時0分0秒 の範囲で入力してください。"
                    ),
                ),
                ValidatonModel(
                    field="datetime3",
                    message="datetime3の形式が正しくありません。",
                ),
                ValidatonModel(
                    field="date1",
                    message="2024年1月1日 から 2024年1月2日 の範囲で入力してください。",
                ),
                ValidatonModel(
                    field="date2",
                    message="2024年1月1日 から 2024年1月2日 の範囲で入力してください。",
                ),
                ValidatonModel(
                    field="date3",
                    message="date3の形式が正しくありません。",
                ),
                ValidatonModel(
                    field="time1",
                    message="0時0分1秒 から 23時59分58秒 の範囲で入力してください。",
                ),
                ValidatonModel(
                    field="time2",
                    message="0時0分1秒 から 23時59分58秒 の範囲で入力してください。",
                ),
                ValidatonModel(
                    field="time3",
                    message="time3の形式が正しくありません。",
                ),
                ValidatonModel(
                    field="bool",
                    message="boolの形式が正しくありません。",
                ),
            ]
        )

        valid = TestValidation(self._data())
        self.assertEqual(len(valid), len(
            expected_valid.result))  # type: ignore
        for v, e in zip(valid, expected_valid.result):  # type: ignore
            self.assertEqual(v, e)
