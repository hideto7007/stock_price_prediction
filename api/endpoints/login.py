
from datetime import timedelta
import os

from api.common.exceptions import BaseHTTPException, ConflictException, HttpExceptionHandler
from api.usercase.login import Login
from dotenv import load_dotenv  # type: ignore
from typing import Any
from api.models.models import UserModel
from const.const import HttpStatusCode, LoginConst
from sqlalchemy.orm import Session  # type: ignore
from fastapi import APIRouter, Depends, Request
from jose import JWTError, jwt  # type: ignore

from api.databases.databases import get_db
from api.schemas.schemas import (
    ErrorMsg,
    Response,
    Token,
    User,
    UserAccessToken,
    UserAccessTokenResponse,
    UserCreate
)
from utils.utils import Swagger


load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")


router = APIRouter()


@router.post(
    "/register_user",
    tags=["認証"],
    responses=Swagger.swagger_responses({
        200: "登録成功",
        400: {
            "message": "既に登録済みのユーザーです"
        },
        500: {
            "message": "予期せぬエラーが発生しました"
        }
    })
)
async def register_user(
    request: Request,
    user: UserCreate,
    db: Session = Depends(get_db)
):
    """
        ログイン情報登録API

        引数:
            user (UserCreate): リクエストボディ
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    try:
        login = Login()
        db_user = login.get_user_name(db, user.user_name)
        if db_user:
            raise ConflictException("既に登録済みのユーザーです")
        login.create_user(db, user)
        return Response[str](
            status_code=HttpStatusCode.SUCCESS,
            data="登録成功"
        )
    except BaseHTTPException as e:
        return await HttpExceptionHandler.main_handler(request, e)

        # HTTPException(
        #     status_code=HttpStatusCode.SERVER_ERROR.value,
        #     detail=error_msg.dict()
        # )


@router.post(
    "/access_token",
    tags=["認証"],
    response_model=Token
)
async def access_token(
    data: UserAccessToken,
    db: Session = Depends(get_db)
):
    """
        アクセストークン取得API

        引数:
            user (UserCreate): リクエストボディ
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    login = Login()
    user = login.authenticate_user(db, data.user_name, data.user_password)
    if not user:
        error_msg = [
            ErrorMsg(
                message="ユーザーIDまたはパスワードが間違っています。"
            )
        ]
        raise Response[Any](
            status_code=HttpStatusCode.UNAUTHORIZED.value,
            detail=error_msg,
            headers=LoginConst.HEADERS.value,
        )
    user_data = UserModel(**user)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = login.create_access_token(
        {"sub": user_data.user_name},
        access_token_expires
    )
    return Response[UserAccessTokenResponse](
        status_code=HttpStatusCode.SUCCESS.value,
        data=UserAccessTokenResponse(
            access_token=access_token,
            token_type="bearer"
        )
    )


@router.get(
    "/read_users_me",
    tags=["認証"],
    response_model=User
)
async def read_users_me(token: str, db: Session = Depends(get_db)):
    """
        ユーザー情報取得API

        引数:
            token (str): アクセストークン
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """
    login = Login()
    error_msg = [
        ErrorMsg(
            message="認証情報の有効期限が切れています。"
        )
    ]
    credentials_exception = Response[Any](
        status_code=HttpStatusCode.UNAUTHORIZED.value,
        detail=error_msg,
        headers=LoginConst.HEADERS.value,
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_name = payload.get("sub")
        if user_name is None:
            raise credentials_exception
        user_info = login.get_user_name(db, user_name)
        if user_info is None:
            error_msg = [
                ErrorMsg(
                    message="対象のユーザー情報がありません。"
                )
            ]
            raise Response[Any](
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                detail=error_msg,
                headers=LoginConst.HEADERS.value,
            )
        return user_info
    except JWTError:
        raise credentials_exception
