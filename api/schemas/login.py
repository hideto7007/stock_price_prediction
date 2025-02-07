from typing import Literal, Optional
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
    user_password: Optional[str] = None


class UserLoginModel(UserBaseModel):
    access_token: str


##################
# リクエストモデル #
##################
class UserCreateRequest(BaseModel):
    user_name: str
    user_email: str
    user_password: str


class UserLoginRequest(BaseModel):
    user_name: str
    user_password: str


##################
# レスポンスモデル #
##################
class UserLoginResponse(BaseModel):
    user_info: UserLoginModel
    token_type: Literal["bearer"] = "bearer"


class UserResponseModel(UserBaseModel):
    pass
