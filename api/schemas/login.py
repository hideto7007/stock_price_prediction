from typing_extensions import Final
from pydantic import BaseModel  # type: ignore
# from datetime import datetime


###########################
# ユーザーケースでも使うモデル #
###########################

class UserBaseModel(BaseModel):
    user_id: int
    user_name: str
    user_email: str
    user_password: str


class GetUserName(BaseModel):
    user_name: str


##################
# リクエストモデル #
##################
class UserCreateRequest(BaseModel):
    user_name: str
    user_email: str
    user_password: str


class UserAccessTokenRequest(BaseModel):
    user_name: str
    user_password: str


##################
# レスポンスモデル #
##################
class UserAccessTokenResponse(BaseModel):
    access_token: str
    token_type: Final[str] = "bearer"


class UserResponseModel(UserBaseModel):
    pass
