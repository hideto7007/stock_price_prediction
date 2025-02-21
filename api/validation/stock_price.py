

from typing import List, Literal
from api.schemas.stock_price import (
    BrandInfoListValidationModel,
    CreateBrandInfoRequest, DeleteBrandInfoValidationModel,
    GetPredictionDataValidationModel, UpdateBrandInfoValidationModel
)
from api.schemas.validation import ValidatonModel
from api.validation.validation import AbstractValidation, ValidationError
from const.const import (
    StockPriceFieldConst as SPFC,
    LoginFieldConst as LFC
)


class GetPredictionDataValidation(
    AbstractValidation[GetPredictionDataValidationModel]
):
    """予測データ取得バリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.user_id(),
            self.brand_code(),
        ])

    ########################################
    # オーバーライドメソッド（バリデーション処理）#
    ########################################
    def user_id(self) -> ValidatonModel | Literal[True]:
        return self.validate_int(self.data.user_id, LFC.USER_ID.value)

    def brand_code(self) -> ValidatonModel | Literal[True]:
        return self.validate_int(
            self.data.brand_code,
            SPFC.BRAND_CODE.value,
            1000, 9999
        )


class BrandInfoListValidation(
    AbstractValidation[BrandInfoListValidationModel]
):
    """対象ユーザーの学習ずみ銘柄情報取得バリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.user_id(),
        ])

    ########################################
    # オーバーライドメソッド（バリデーション処理）#
    ########################################
    def user_id(self) -> ValidatonModel | Literal[True]:
        return self.validate_int(self.data.user_id, LFC.USER_ID.value)


class CreateBrandInfoRequestValidation(
    AbstractValidation[CreateBrandInfoRequest]
):
    """予測データ登録バリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.brand_name(),
            self.brand_code(),
            self.user_id(),
            self.create_by(),
            self.update_by(),
            self.is_valid(),
        ])

    ########################################
    # オーバーライドメソッド（バリデーション処理）#
    ########################################
    def brand_name(self) -> ValidatonModel | Literal[True]:
        return self.validate_str(
            self.data.brand_name,
            SPFC.BRAND_NAME.value
        )

    def brand_code(self) -> ValidatonModel | Literal[True]:
        return self.validate_int(
            self.data.brand_code,
            SPFC.BRAND_CODE.value,
            1000, 9999
        )

    def user_id(self) -> ValidatonModel | Literal[True]:
        return self.validate_int(self.data.user_id, LFC.USER_ID.value)

    def create_by(self) -> ValidatonModel | Literal[True]:
        return self.validate_str(
            self.data.create_by,
            SPFC.CREATE_BY.value
        )

    def update_by(self) -> ValidatonModel | Literal[True]:
        return self.validate_str(
            self.data.update_by,
            SPFC.UPDATE_BY.value
        )

    def is_valid(self) -> ValidatonModel | Literal[True]:
        return self.validate_bool(
            self.data.is_valid,
            SPFC.IS_VALID.value
        )


class UpdateBrandInfoRequestValidation(
    AbstractValidation[UpdateBrandInfoValidationModel]
):
    """予測データ更新バリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.brand_name(),
            self.brand_code(),
            self.user_id(),
            self.update_by(),
        ])

    ########################################
    # オーバーライドメソッド（バリデーション処理）#
    ########################################
    def brand_name(self) -> ValidatonModel | Literal[True]:
        return self.validate_str(
            self.data.brand_name,
            SPFC.BRAND_NAME.value
        )

    def brand_code(self) -> ValidatonModel | Literal[True]:
        return self.validate_int(
            self.data.brand_code,
            SPFC.BRAND_CODE.value,
            1000, 9999
        )

    def user_id(self) -> ValidatonModel | Literal[True]:
        return self.validate_int(self.data.user_id, LFC.USER_ID.value)

    def update_by(self) -> ValidatonModel | Literal[True]:
        return self.validate_str(
            self.data.update_by,
            SPFC.UPDATE_BY.value
        )


class DeleteBrandInfoRequestValidation(
    AbstractValidation[DeleteBrandInfoValidationModel]
):
    """予測データ削除バリデーション"""

    ##############################
    # プライベートメソッド（内部処理）#
    ##############################
    def result(self) -> List[ValidatonModel]:
        return ValidationError.valid_result([
            self.user_id(),
            self.brand_code()
        ])

    ########################################
    # オーバーライドメソッド（バリデーション処理）#
    ########################################
    def user_id(self) -> ValidatonModel | Literal[True]:
        return self.validate_int(self.data.user_id, LFC.USER_ID.value)

    def brand_code(self) -> ValidatonModel | Literal[True]:
        return self.validate_int(
            self.data.brand_code,
            SPFC.BRAND_CODE.value,
            1000, 9999
        )
