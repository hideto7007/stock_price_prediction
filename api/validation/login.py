

from typing import List, Literal
from api.schemas.login import (
    CreateUserRequest, LoginUserRequest, UserIdRequest
)
from api.schemas.validation import ValidatonModel
from api.validation.validation import AbstractValidation, ValidationError
from const.const import LoginFieldConst as LFC


class RegisterUserValidation(AbstractValidation[CreateUserRequest]):
    """ログイン情報登録バリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.user_name(),
            self.user_email(),
            self.user_password(),
        ])

    #######################
    # バリデーション呼び出し #
    #######################
    def user_name(self) -> ValidatonModel | Literal[True]:
        return self.validate_str(self.data.user_name, LFC.USER_NAME.value)

    def user_email(self) -> ValidatonModel | Literal[True]:
        return self.validate_email(self.data.user_email, LFC.USER_EMAIL.value)

    def user_password(self) -> ValidatonModel | Literal[True]:
        return self.validate_password(
            self.data.user_password,
            LFC.USER_PASSWORD.value
        )


class LoginUserValidation(AbstractValidation[LoginUserRequest]):
    """ログイン情報取得バリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.user_name(),
            self.user_password(),
        ])

    #######################
    # バリデーション呼び出し #
    #######################
    def user_name(self) -> ValidatonModel | Literal[True]:
        return self.validate_str(self.data.user_name, LFC.USER_NAME.value)

    def user_password(self) -> ValidatonModel | Literal[True]:
        return self.validate_password(
            self.data.user_password,
            LFC.USER_PASSWORD.value
        )

# TODO:未使用
# class ReadUsersMeValidation(AbstractValidation[ReadUsersMeRequest]):
#     """ユーザー情報取得バリデーション"""

#     ##############################
#     # プライベートメソッド（内部処理）#
#     ##############################
#     def result(self) -> List[ValidatonModel]:
#         return ValidationError.valid_result([
#             self.access_token(),
#         ])

#     #######################
#     # バリデーション呼び出し #
#     #######################
#     def access_token(self) -> ValidatonModel | Literal[True]:
#         return self.validate_str(
#             self.data.access_token,
#             LFC.ACCESS_TOKEN.value
#         )


class UserIdValidation(AbstractValidation[UserIdRequest]):
    """ユーザーidチェックバリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.user_id(),
        ])

    #######################
    # バリデーション呼び出し #
    #######################
    def user_id(self) -> ValidatonModel | Literal[True]:
        return self.validate_int(
            self.data.user_id,
            LFC.USER_ID.value
        )
