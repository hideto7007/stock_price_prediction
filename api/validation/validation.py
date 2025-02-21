from abc import ABCMeta, abstractmethod
import datetime
from itertools import chain
import re
from typing import Any, Final, Generic, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel

from api.schemas.validation import ValidatonModel


# Pydanticモデルの型を汎用的に扱えるように TypeVar を定義
T = TypeVar("T", bound=BaseModel)


REGEX_EMAIL: Final[str] = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
REGEX_PASSWORD: Final[str] = (
    r'^(?=.*[A-Z])(?=.*[.!?/-])[a-zA-Z0-9.!?/-]{8,24}$'
)


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
        val: int,
        field: Any,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> Union[ValidatonModel, Literal[True]]:
        """
        整数値バリデーション

        - 必須チェック (`val` が整数であり、0以上か)
        - `min_value`, `max_value` が指定されている場合は範囲チェックを行う

        引数:
            val (int): バリデーション対象の整数値
            field (Any): メッセージで出力するキー
            min_value (Optional[int]): 最小値（指定しない場合は None）
            max_value (Optional[int]): 最大値（指定しない場合は None）

        戻り値:
            Union[ValidatonModel, Literal[True]]:
                - `True` の場合: バリデーション通過
                - `ValidatonModel` の場合: バリデーションエラー
        """
        if not isinstance(val, int) or val <= 0:
            return ValidationError.generater(
                field,
                f"{field}は必須です。"
            )

        if (min_value is not None and val < min_value) or \
                (max_value is not None and val > max_value):
            return ValidationError.generater(
                field,
                f"{min_value} から {max_value} の範囲で入力してください。"
            )

        return True

    def validate_float(
        self,
        val: int,
        field: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> ValidatonModel | Literal[True]:
        """
        小数点バリデーション

        - 必須チェック (`val` が浮動小数であり、0以上か)
        - `min_value`, `max_value` が指定されている場合は範囲チェックを行う

        引数:
            val (int): バリデーション対象の浮動小数
            field (Any): メッセージで出力するキー
            min_value (Optional[float]): 最小値（指定しない場合は None）
            max_value (Optional[float]): 最大値（指定しない場合は None）

        戻り値:
            Union[ValidatonModel, Literal[True]]:
                - `True` の場合: バリデーション通過
                - `ValidatonModel` の場合: バリデーションエラー
        """
        if not isinstance(val, float) or val <= 0:
            return ValidationError.generater(
                field,
                f"{field}は必須です。"
            )

        if (min_value is not None and val < min_value) or \
                (max_value is not None and val > max_value):
            return ValidationError.generater(
                field,
                f"{min_value} から {max_value} の範囲で入力してください。"
            )

        return True

    def validate_str(
        self,
        val: str,
        field: Any,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> Union[ValidatonModel, Literal[True]]:
        """
        文字列バリデーション

        - 空文字チェック
        - `min_length` / `max_length` が指定されている場合、長さの範囲チェック

        引数:
            val (str): バリデーション対象の文字列
            field (Any): メッセージで出力するキー
            min_length (Optional[int]): 文字列の最小長（指定なしの場合は `None`）
            max_length (Optional[int]): 文字列の最大長（指定なしの場合は `None`）

        戻り値:
            Union[ValidatonModel, Literal[True]]:
                - `True` の場合: バリデーション通過
                - `ValidatonModel` の場合: バリデーションエラー
        """
        if not isinstance(val, str) or val.strip() == "":
            return ValidationError.generater(field, f"{field}は必須です。")

        if (min_length is not None and len(val) < min_length) or \
           (max_length is not None and len(val) > max_length):
            return ValidationError.generater(
                field,
                f"{min_length} 文字以上 {max_length} 文字以下で入力してください。"
            )

        return True

    def validate_datetime(
        self,
        val: datetime.datetime,
        field: Any,
        min_value: Optional[datetime.datetime] = None,
        max_value: Optional[datetime.datetime] = None
    ) -> Union[ValidatonModel, Literal[True]]:
        """
        年月日 + 時間バリデーション

        - `min_value` / `max_value` が指定されている場合、範囲チェック

        引数:
            val (datetime.datetime): バリデーション対象の日時
            field (Any): メッセージで出力するキー
            min_value (Optional[datetime.datetime]): 最小許容日時（デフォルト `None`）
            max_value (Optional[datetime.datetime]): 最大許容日時（デフォルト `None`）

        戻り値:
            Union[ValidatonModel, Literal[True]]:
                - `True` の場合: バリデーション通過
                - `ValidatonModel` の場合: バリデーションエラー
        """
        if not isinstance(val, datetime.datetime):
            return ValidationError.generater(
                field,
                f"{field}の形式が正しくありません。"
            )

        if (min_value is not None and val < min_value) or \
                (max_value is not None and val > max_value):
            return ValidationError.generater(
                field,
                f"{min_value} から {max_value} の範囲で入力してください。"
            )

        return True

    def validate_date(
        self,
        val: datetime.date,
        field: Any,
        min_value: Optional[datetime.date] = None,
        max_value: Optional[datetime.date] = None
    ) -> Union[ValidatonModel, Literal[True]]:
        """
        年月日バリデーション

        - `min_value` / `max_value` が指定されている場合、範囲チェック

        引数:
            val (datetime.date): バリデーション対象の日付
            field (Any): メッセージで出力するキー
            min_value (Optional[datetime.date]): 最小許容日（デフォルト `None`）
            max_value (Optional[datetime.date]): 最大許容日（デフォルト `None`）

        戻り値:
            Union[ValidatonModel, Literal[True]]:
                - `True` の場合: バリデーション通過
                - `ValidatonModel` の場合: バリデーションエラー
        """
        if not isinstance(val, datetime.date):
            return ValidationError.generater(
                field,
                f"{field}の形式が正しくありません。"
            )

        if (min_value is not None and val < min_value) or \
                (max_value is not None and val > max_value):
            return ValidationError.generater(
                field,
                f"{min_value} から {max_value} の範囲で入力してください。"
            )

        return True

    def validate_time(
        self,
        val: datetime.time,
        field: Any,
        min_value: Optional[datetime.time] = None,
        max_value: Optional[datetime.time] = None
    ) -> Union[ValidatonModel, Literal[True]]:
        """
        時間バリデーション

        - `min_value` / `max_value` が指定されている場合、範囲チェック

        引数:
            val (datetime.time): バリデーション対象の時間
            field (Any): メッセージで出力するキー
            min_value (Optional[datetime.time]): 最小許容時間（デフォルト `None`）
            max_value (Optional[datetime.time]): 最大許容時間（デフォルト `None`）

        戻り値:
            Union[ValidatonModel, Literal[True]]:
                - `True` の場合: バリデーション通過
                - `ValidatonModel` の場合: バリデーションエラー
        """
        if not isinstance(val, datetime.time):
            return ValidationError.generater(
                field,
                f"{field}の形式が正しくありません。"
            )

        if (min_value is not None and val < min_value) or \
                (max_value is not None and val > max_value):
            return ValidationError.generater(
                field,
                f"{min_value} から {max_value} の範囲で入力してください。"
            )

        return True

    def validate_bool(
        self,
        val: bool,
        field: Any,
    ) -> Union[ValidatonModel, Literal[True]]:
        """
        真偽値バリデーション


        引数:
            val (datetime.time): バリデーション対象の真偽値
            field (Any): メッセージで出力するキー

        戻り値:
            Union[ValidatonModel, Literal[True]]:
                - `True` の場合: バリデーション通過
                - `ValidatonModel` の場合: バリデーションエラー
        """
        if not isinstance(val, bool):
            return ValidationError.generater(
                field,
                f"{field}の形式が正しくありません。"
            )

        return True

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
        val: str,
        field: Any
    ):
        """
        Eメールアドレスバリデーション

        引数:
            val (datetime.time): バリデーション対象の文字列
            field (Any): メッセージで出力するキー

        戻り値:
            Union[ValidatonModel, Literal[True]]:
                - `True` の場合: バリデーション通過
                - `ValidatonModel` の場合: バリデーションエラー
        """
        if not isinstance(val, str) or val.strip() == "":
            return ValidationError.generater(
                field,
                f"{field}は必須です。"
            )
        elif re.match(REGEX_EMAIL, val) is None:
            return ValidationError.generater(
                field,
                f"{field}の形式が間違っています。"
            )
        return True

    def validate_password(
        self,
        val: str,
        field: Any
    ):
        """
        パスワードバリデーション

        引数:
            val (datetime.time): バリデーション対象の文字列
            field (Any): メッセージで出力するキー

        戻り値:
            Union[ValidatonModel, Literal[True]]:
                - `True` の場合: バリデーション通過
                - `ValidatonModel` の場合: バリデーションエラー
        """
        if not isinstance(val, str) or val.strip() == "":
            return ValidationError.generater(
                field,
                f"{field}は必須です。"
            )
        elif re.match(REGEX_PASSWORD, val) is None:
            return ValidationError.generater(
                field,
                f"{field}は8文字以上24文字以下、"
                f"大文字、記号(ビックリマーク(!)、ピリオド(.)、スラッシュ(/)、"
                f"クエスチョンマーク(?)、ハイフン(-))を含めてください"
            )
        return True

    def validate_regex(
        self,
    ):
        """
            特定の文字列でのバリデーション

            機能
            - 特定の文字であるかのチェックを行う場合、正規表現で行う

        """
        pass
