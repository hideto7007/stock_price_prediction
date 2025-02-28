from typing import Literal, Optional
from pydantic import BaseModel
# from datetime import datetime


###########################
# ユーザーケースでも使うモデル #
###########################

class BaseUserModel(BaseModel):
    user_id: int
    user_name: str
    user_email: str
    user_password: Optional[str] = None


class LoginUserModel(BaseUserModel):
    access_token: str


##################
# リクエストモデル #
##################
class CreateUserRequest(BaseModel):
    user_name: str
    user_email: str
    user_password: str


class UpdateUserRequest(BaseModel):
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    user_confirmation_password: Optional[str] = None
    user_password: Optional[str] = None


class RestoreUserRequest(CreateUserRequest):
    pass


class LoginUserRequest(BaseModel):
    user_name: str
    user_password: str

# TODO:未使用
# class ReadUsersMeRequest(BaseModel):
#     access_token: str


class UserIdRequest(BaseModel):
    user_id: int


##################
# レスポンスモデル #
##################
class LoginUserResponse(BaseModel):
    user_info: LoginUserModel
    token_type: Literal["bearer"] = "bearer"


class UserResponseModel(BaseUserModel):
    pass
