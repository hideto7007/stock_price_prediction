

import re
from typing import Final, List, Literal
from api.schemas.login import (
    ReadUsersMeRequest, CreateUserRequest, LoginUserRequest, UserIdRequest
)
from api.schemas.validation import ValidatonModel
from api.validation.validation import AbstractValidation, ValidationError
from const.const import LoginFieldConst as LFC


REGEX_EMAIL: Final[str] = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
REGEX_PASSWORD: Final[str] = (
    r'^(?=.*[A-Z])(?=.*[.!?/-])[a-zA-Z0-9.!?/-]{8,24}$'
)


class RegisterUserValidation(AbstractValidation[CreateUserRequest]):
    """ログイン情報登録バリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.validate_str(),
            self.validate_email(),
            self.validate_password(),
        ])

    ########################################
    # オーバーライドメソッド（バリデーション処理）#
    ########################################

    def validate_str(self) -> ValidatonModel | Literal[True]:
        if len(self.data.user_name) > 0:
            return True
        return ValidationError.generater(
            LFC.USER_NAME.value,
            f"{LFC.USER_NAME.value}は必須です。"
        )

    def validate_email(self) -> ValidatonModel | Literal[True]:
        if len(self.data.user_email) == 0:
            return ValidationError.generater(
                LFC.USER_EMAIL.value,
                f"{LFC.USER_EMAIL.value}は必須です。"
            )
        elif re.match(REGEX_EMAIL, self.data.user_email) is None:
            return ValidationError.generater(
                LFC.USER_EMAIL.value,
                f"{LFC.USER_EMAIL.value}の形式が間違っています。"
            )
        return True

    def validate_password(self) -> ValidatonModel | Literal[True]:
        if len(self.data.user_password) == 0:
            return ValidationError.generater(
                LFC.USER_PASSWORD.value,
                f"{LFC.USER_PASSWORD.value}は必須です。"
            )
        elif re.match(REGEX_PASSWORD, self.data.user_password) is None:
            return ValidationError.generater(
                LFC.USER_PASSWORD.value,
                f"{LFC.USER_PASSWORD.value}は8文字以上24文字以下、"
                f"大文字、記号(ビックリマーク(!)、ピリオド(.)、スラッシュ(/)、"
                f"クエスチョンマーク(?)、ハイフン(-))を含めてください"
            )
        return True


class LoginUserValidation(AbstractValidation[LoginUserRequest]):
    """ログイン情報取得バリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.validate_str(),
            self.validate_password(),
        ])

    ########################################
    # オーバーライドメソッド（バリデーション処理）#
    ########################################

    def validate_str(self) -> ValidatonModel | Literal[True]:
        if len(self.data.user_name) > 0:
            return True
        return ValidationError.generater(
            LFC.USER_NAME.value,
            f"{LFC.USER_NAME.value}は必須です。"
        )

    def validate_password(self) -> ValidatonModel | Literal[True]:
        if len(self.data.user_password) == 0:
            return ValidationError.generater(
                LFC.USER_PASSWORD.value,
                f"{LFC.USER_PASSWORD.value}は必須です。"
            )
        elif re.match(REGEX_PASSWORD, self.data.user_password) is None:
            return ValidationError.generater(
                LFC.USER_PASSWORD.value,
                f"{LFC.USER_PASSWORD.value}は8文字以上24文字以下、"
                f"大文字、記号(ビックリマーク(!)、ピリオド(.)、スラッシュ(/)、"
                f"クエスチョンマーク(?)、ハイフン(-))を含めてください"
            )
        return True


class ReadUsersMeValidation(AbstractValidation[ReadUsersMeRequest]):
    """ユーザー情報取得バリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.validate_str(),
        ])

    ########################################
    # オーバーライドメソッド（バリデーション処理）#
    ########################################

    def validate_str(self) -> ValidatonModel | Literal[True]:
        if len(self.data.access_token) > 0:
            return True
        return ValidationError.generater(
            LFC.ACCESS_TOKEN.value,
            f"{LFC.ACCESS_TOKEN.value}は必須です。"
        )


class UserIdValidation(AbstractValidation[UserIdRequest]):
    """ユーザーidチェックバリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.validate_int(),
        ])

    ########################################
    # オーバーライドメソッド（バリデーション処理）#
    ########################################

    def validate_int(self) -> ValidatonModel | Literal[True]:
        if isinstance(self.data.user_id, int):
            return True
        return ValidationError.generater(
            LFC.USER_ID.value,
            f"{LFC.USER_ID.value}は必須です。"
        )
