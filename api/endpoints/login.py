
from datetime import timedelta
from typing import cast

from fastapi.responses import JSONResponse

from api.common.env import Env
from api.common.exceptions import (
    HttpExceptionHandler
)
from api.common.response import ValidationResponse
from api.validation.login import (
    LoginUserValidation, ReadUsersMeValidation,
    RegisterUserValidation, UserIdValidation
)
from api.schemas.login import (
    ReadUsersMeRequest, CreateUserRequest,
    LoginUserModel, UpdateUserRequest, UserIdRequest
)
from api.schemas.validation import ValidatonModel
from api.usercase.login import Login
from const.const import HttpStatusCode
from sqlalchemy.orm import Session  # type: ignore
from fastapi import APIRouter, Depends, Request

from api.databases.databases import get_db
from api.schemas.response import Content
from api.schemas.login import (
    LoginUserRequest,
    LoginUserResponse,
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
    user: CreateUserRequest,
    db: Session = Depends(get_db)
):
    """
        ログイン情報登録API

        引数:
            user (CreateUserRequest): リクエストボディ
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    try:
        # バリデーションチェック
        valid = RegisterUserValidation(user)

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

        login = Login()
        db_user = login.get_user_info(db, user.user_name)
        if db_user:
            return JSONResponse(
                status_code=HttpStatusCode.CONFLICT.value,
                content=Content[str](
                    result="既に登録済みのユーザーです"
                ).model_dump()
            )
        user = login.save_user(db, user)
        context = Content[UserResponseModel](
            result=UserResponseModel(
                user_id=cast(int, user.user_id),
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
    data: LoginUserRequest,
    db: Session = Depends(get_db)
):
    """
        ログイン情報取得API

        引数:
            user (LoginUserRequest): リクエストボディ
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    try:
        # バリデーションチェック
        valid = LoginUserValidation(data)

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

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
        context = Content[LoginUserResponse](
            result=LoginUserResponse(
                user_info=LoginUserModel(
                    user_id=cast(int, user.user_id),
                    user_name=str(user.user_name),
                    user_email=str(user.user_email),
                    access_token=access_token
                )
            )
        )
        response = JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=context.model_dump()
        )

        # ログイン時にトークンセット
        response.set_cookie(
            key="auth_stock_price_token",
            value=access_token,
            httponly=True,
            secure=True,
            samesite="strict"
        )

        return response

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
        # バリデーションチェック
        valid = ReadUsersMeValidation(
            ReadUsersMeRequest(
                access_token=token
            )
        )

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

        login = Login()
        user_name = login.get_valid_user_name(token)
        if user_name is None:
            return JSONResponse(
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                content=Content[str](
                    result="認証情報の有効期限が切れています。"
                ).model_dump()
            )
        user = login.get_user_info(db, user_name)
        if user is None:
            return JSONResponse(
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                content=Content[str](
                    result="対象のユーザー情報がありません。"
                ).model_dump()
            )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=Content[UserResponseModel](
                result=UserResponseModel(
                    user_id=cast(int, user.user_id),
                    user_name=str(user.user_name),
                    user_email=str(user.user_email)
                )
            ).model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.get(
    "/user_info/{user_id}",
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
async def user_info(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db)
):
    """
        ユーザー情報取得API

        引数:
            user_id (int): ユーザーid
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    try:
        # バリデーションチェック
        valid = UserIdValidation(
            UserIdRequest(
                user_id=user_id
            )
        )

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

        login = Login()
        user = login.get_user_info(db, None, user_id)
        if user is None:
            return JSONResponse(
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                content=Content[str](
                    result="対象のユーザー情報がありません。"
                ).model_dump()
            )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=Content[UserResponseModel](
                result=UserResponseModel(
                    user_id=cast(int, user.user_id),
                    user_name=str(user.user_name),
                    user_email=str(user.user_email)
                )
            ).model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.put(
    "/users_update/{user_id}",
    tags=["認証"],
    responses=Swagger.swagger_responses({
        200: "ユーザー情報の更新成功",
        401: "認証情報の有効期限が切れています。or 対象のユーザー情報がありません。",
        500: "予期せぬエラーが発生しました"
    })
)
async def users_update(
    request: Request,
    user_id: int,
    data: UpdateUserRequest,
    db: Session = Depends(get_db)
):
    """
        ユーザー情報更新API

        引数:
            user_id (int): ユーザーid
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    try:
        # バリデーションチェック
        valid = UserIdValidation(
            UserIdRequest(
                user_id=user_id
            )
        )

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

        login = Login()
        login.update_user(db, user_id, data)
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=Content[str](
                result="ユーザー情報の更新成功"
            ).model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.delete(
    "/users_delete/{user_id}",
    tags=["認証"],
    responses=Swagger.swagger_responses({
        200: "ユーザー情報の削除成功",
        401: "認証情報の有効期限が切れています。or 対象のユーザー情報がありません。",
        500: "予期せぬエラーが発生しました"
    })
)
async def users_delete(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db)
):
    """
        ユーザー情報削除API

        引数:
            user_id (int): ユーザーid
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    try:
        # バリデーションチェック
        valid = UserIdValidation(
            UserIdRequest(
                user_id=user_id
            )
        )

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

        login = Login()
        login.delete_user(db, user_id)
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=Content[str](
                result="ユーザー情報の削除成功"
            ).model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)
