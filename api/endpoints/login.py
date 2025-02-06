
from datetime import timedelta

from fastapi.responses import JSONResponse

from api.common.env import Env
from api.common.exceptions import (
    ConflictException,
    HttpExceptionHandler
)
from api.schemas.login import GetUserName, UserCreateRequest
from api.usercase.login import Login
from const.const import HttpStatusCode
from sqlalchemy.orm import Session  # type: ignore
from fastapi import APIRouter, Depends, Request
from jose import jwt  # type: ignore

from api.databases.databases import get_db
from api.schemas.response import Content
from api.schemas.login import (
    UserAccessTokenRequest,
    UserAccessTokenResponse,
    UserResponseModel
)
from utils.utils import Swagger


env = Env.get_instance()
router = APIRouter()


@router.post(
    "/register_user",
    tags=["認証"],
    responses=Swagger.swagger_responses({
        200: {
            "user_id": 1,
            "user_email": "test@example.com",
            "user_name": "test",
            "user_password": "hash_password"
        },
        409: "既に登録済みのユーザーです",
        500: "予期せぬエラーが発生しました"
    })
)
async def register_user(
    request: Request,
    user: UserCreateRequest,
    db: Session = Depends(get_db)
):
    """
        ログイン情報登録API

        引数:
            user (UserCreateRequest): リクエストボディ
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    try:
        login = Login()
        db_user = login.get_user_name(db, user.user_name)
        if db_user:
            raise ConflictException("既に登録済みのユーザーです")
        user = login.create_user(db, user)
        context = Content[UserResponseModel](
            result=UserResponseModel(
                user_id=int(getattr(user, "user_id", 0)),
                user_email=str(user.user_email),
                user_name=str(user.user_name),
                user_password=str(user.user_password)
            )
        )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=context.model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.post(
    "/access_token",
    tags=["認証"],
    responses=Swagger.swagger_responses({
        200: {
            "access_token": "access_token",
            "token_type": "bearer"
        },
        401: "ユーザーIDまたはパスワードが間違っています。",
        500: "予期せぬエラーが発生しました"
    })
)
async def access_token(
    request: Request,
    data: UserAccessTokenRequest,
    db: Session = Depends(get_db)
):
    """
        アクセストークン取得API

        引数:
            user (UserAccessTokenRequest): リクエストボディ
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    try:
        login = Login()
        user = login.authenticate_user(db, data.user_name, data.user_password)
        if not user:
            content = Content[str](
                result="ユーザーIDまたはパスワードが間違っています。"
            )
            return JSONResponse(
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                content=content.model_dump()
            )
        user_data = GetUserName(**user)
        access_token_expires = timedelta(
            minutes=env.access_token_expire_minutes)
        access_token = login.create_access_token(
            {"sub": user_data.user_name},
            access_token_expires
        )
        context = Content[UserAccessTokenResponse](
            result=UserAccessTokenResponse(
                access_token=access_token
            )
        )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=context.model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.get(
    "/read_users_me/{token}",
    tags=["認証"],
    responses=Swagger.swagger_responses({
        200: {
            "access_token": "access_token",
            "token_type": "bearer"
        },
        401: "ユーザーIDまたはパスワードが間違っています。",
        500: "予期せぬエラーが発生しました"
    })
)
async def read_users_me(
    request: Request,
    token: str,
    db: Session = Depends(get_db)
):
    """
        ユーザー情報取得API

        引数:
            token (str): アクセストークン
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    try:
        login = Login()
        payload = jwt.decode(token, env.secret_key, algorithms=[env.algorithm])
        user_name = payload.get("sub")
        if user_name is None:
            return JSONResponse(
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                content=Content[str](
                    result="認証情報の有効期限が切れています。"
                )
            )
        user_info = login.get_user_name(db, user_name)
        if user_info is None:
            return JSONResponse(
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                content=Content[str](
                    result="対象のユーザー情報がありません。"
                )
            )
        user_info = UserResponseModel(**user_info)
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=user_info
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)
