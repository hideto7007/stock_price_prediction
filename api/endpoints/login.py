
from datetime import timedelta
from typing import cast

from fastapi.responses import JSONResponse

from api.common.env import Env
from api.common.exceptions import (
    HttpExceptionHandler
)
from api.schemas.login import UserCreateRequest, UserLoginModel
from api.usercase.login import Login
from const.const import HttpStatusCode
from sqlalchemy.orm import Session  # type: ignore
from fastapi import APIRouter, Depends, Request

from api.databases.databases import get_db
from api.schemas.response import Content
from api.schemas.login import (
    UserLoginRequest,
    UserLoginResponse,
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
            return JSONResponse(
                status_code=HttpStatusCode.CONFLICT.value,
                content=Content[str](
                    result="既に登録済みのユーザーです"
                ).model_dump()
            )
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
    "/login_user",
    tags=["認証"],
    responses=Swagger.swagger_responses({
        200: {
            "user_info": {
                "user_id": 1,
                "user_name": "test",
                "user_email": "test@example.com",
                "access_token": "token"
            },
            "token_type": "bearer"
        },
        401: "ユーザーIDまたはパスワードが間違っています。",
        500: "予期せぬエラーが発生しました"
    })
)
async def login_user(
    request: Request,
    data: UserLoginRequest,
    db: Session = Depends(get_db)
):
    """
        アクセストークン取得API

        引数:
            user (UserLoginRequest): リクエストボディ
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
        access_token_expires = timedelta(
            minutes=int(env.access_token_expire_minutes))
        access_token = login.create_access_token(
            {"sub": user.user_name},
            access_token_expires
        )
        context = Content[UserLoginResponse](
            result=UserLoginResponse(
                user_info=UserLoginModel(
                    user_id=cast(int, user.user_id),
                    user_name=str(user.user_name),
                    user_email=str(user.user_email),
                    access_token=access_token
                )
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
            "user_id": 17,
            "user_name": "hideto",
            "user_email": "test@example.com",
        },
        401: "認証情報の有効期限が切れています。or 対象のユーザー情報がありません。",
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
        user_name = login.get_valid_user_name(token)
        if user_name is None:
            return JSONResponse(
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                content=Content[str](
                    result="認証情報の有効期限が切れています。"
                ).model_dump()
            )
        user = login.get_user_name(db, user_name)
        if user is None:
            return JSONResponse(
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                content=Content[str](
                    result="対象のユーザー情報がありません。"
                ).model_dump()
            )
        user_res = UserResponseModel(
            user_id=cast(int, user.user_id),
            user_name=str(user.user_name),
            user_email=str(user.user_email),
        )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=user_res.model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)
