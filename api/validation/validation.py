from abc import ABCMeta, abstractmethod
import datetime
from itertools import chain
from typing import Any, Generic, List, TypeVar

from pydantic import BaseModel

from api.schemas.validation import ValidatonModel


# Pydanticモデルの型を汎用的に扱えるように TypeVar を定義
T = TypeVar("T", bound=BaseModel)


#######################
# エラーオグジェクト生成 #
#######################
class ValidationError:
    """バリデーションエラーハンドラー"""

    @staticmethod
    def generater(field: str, message: str) -> ValidatonModel:
        """バリデーションエラーモデルを生成"""
        return ValidatonModel(
            field=field,
            message=message
        )

    @staticmethod
    def valid_result(valid_list: List[Any]) -> List[ValidatonModel]:
        """
        バリデーション結果をリスト化して返却する
        - `True` は正常値とみなしリストには含めない
        - リスト内にリストがある場合、平坦化して返却
        """
        result = []
        flag = False

        for v in valid_list:
            if not isinstance(v, bool):
                if isinstance(v, list):
                    flag = True
                result.append(v)

        if flag:
            return list(chain.from_iterable(result))
        else:
            return result


#############
# 抽象クラス #
#############

class AbstractValidation(Generic[T], metaclass=ABCMeta):
    """
        バリデーション抽象クラス

        この抽象クラスの説明
        - @abstractmethodはつけない
        理由：@abstractmethodを明示してしまうと必ず子クラスで実装しなければならなく
        余計な空メソッドが増えてしまうので必要なもの以外は@abstractmethodはつけない
    """

    ###############################
    # 特殊メソッド（コンストラクタなど）#
    ###############################
    def __new__(
        cls,
        data: T
    ) -> List[ValidatonModel]:
        instance = super().__new__(cls)
        instance.__init__(data)
        return instance.result()

    def __init__(self, data: T) -> None:
        """共通の初期化処理"""
        self.data = data

    def __len__(self) -> int:
        """バリデーション結果の数を返す"""
        return len(self.result())

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    @abstractmethod
    def result(self) -> List[ValidatonModel]:
        """バリデーション結果を返す（子クラスで実装する）"""
        pass

    ##########################
    # バリデーションチェック #
    ##########################

    def validate_int(
        self,
        min_value: int | None = None,
        max_value: int | None = None
    ):
        """整数値バリデーション"""
        pass

    def validate_float(
        self,
        min_value: float | None = None,
        max_value: float | None = None
    ):
        """小数点バリデーション"""
        pass

    def validate_str(
        self,
        min_value: str | None = None,
        max_value: str | None = None
    ):
        """文字列バリデーション"""
        pass

    def validate_datetime(
        self,
        min_value: datetime.datetime | None = None,
        max_value: datetime.datetime | None = None
    ):
        """年月日+時間バリデーション"""
        pass

    def validate_date(
        self,
        min_value: datetime.datetime | None = None,
        max_value: datetime.datetime | None = None
    ):
        """年月日バリデーション"""
        pass

    def validate_time(
        self,
        min_value: datetime.datetime | None = None,
        max_value: datetime.datetime | None = None
    ):
        """時間バリデーション"""
        pass

    def validate_exist(
        self,
    ):
        """
            存在バリデーション

            機能
            - Noneチェック
            - ブーリアンチェック
            などの値が存在又は真であるかのチェック
        """
        pass

    #################################
    # 特殊ケースのバリデーションチェック #
    #################################

    def validate_email(
        self,
    ):
        """Eメールアドレスバリデーション"""
        pass

    def validate_password(
        self,
    ):
        """パスワードバリデーション"""
        pass

    def validate_regex(
        self,
    ):
        """
            特定の文字列でのバリデーション

            機能
            - 特定の文字であるかのチェックを行う場合、正規表現で行う

        """
        pass
