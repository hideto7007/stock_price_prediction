# middleware.py
import json
import uuid
from fastapi import Request  # type: ignore
from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore
from fastapi.security import OAuth2PasswordBearer  # type: ignore
from starlette.types import ASGIApp
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
from typing import Any

from api.common.exceptions import (
    ExpiredSignatureException,
    HttpExceptionHandler,
    TokenRequiredException
)
from api.usercase.login import Login
from common.logger import Logger
from const.const import HttpStatusCode
from api.schemas.response import Content


class OAuth2Middleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
    ):
        super().__init__(app)
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.exempt_paths = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
            "/api/login/login_user",
            "/api/login/register_user"
        ]

    async def dispatch(self, request: Request, call_next):
        # ユーザー登録とログインでは認証トークンチェックを省く
        for path in self.exempt_paths:
            if request.url.path in path:
                return await call_next(request)

        try:
            token = await self.oauth2_scheme(request)
            if token is None:
                raise TokenRequiredException("認証トークンが必要です。")
            if self.is_valid_token(token) is None:
                raise ExpiredSignatureException("トークンが無効です。")

        except TokenRequiredException as e:
            return await HttpExceptionHandler.main_handler(request, e)
        except ExpiredSignatureException as e:
            return await HttpExceptionHandler.main_handler(request, e)
        return await call_next(request)

    def is_valid_token(self, token: str) -> dict[str, Any] | None:
        return Login.get_payload(token)


class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, timeout: int = 30):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        timeout = self.timeout
        # 特定のパスでタイムアウトを変更
        if (request.url.path.startswith("/create") or
                request.url.path.startswith("/update")):
            timeout = 1800  # 30分
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except asyncio.TimeoutError:
            error_msg = Content[str](result="read time out.")
            return JSONResponse(
                status_code=HttpStatusCode.TIMEOUT.value,
                content=error_msg.model_dump()
            )


class RequestWritingLoggerMiddleware(BaseHTTPMiddleware):
    """リクエストごと結果をログに書き込むMiddleware"""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request_body = await self.get_request_body(request)
        response = await call_next(request)

        # ここではレスポンスデータをログに出力するためチャンクして取り出している
        # 後の処理でレスポンスを再生成する
        content = b""
        async for chunk in response.body_iterator:  # type: ignore
            content += chunk
        response.headers["X-Request-ID"] = request_id

        # リクエストのパスワードはマスクする
        request_body = self.password_mask(request_body)

        request_body_dump = json.dumps(request_body, ensure_ascii=False)
        content_decode = content.decode("utf-8")

        if response.status_code == HttpStatusCode.SUCCESS.value:
            Logger.info(request, request_body_dump, content_decode)
        else:
            Logger.error(request, request_body_dump, content_decode)

        new_response = StreamingResponse(
            iter([content]),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
        return new_response

    async def get_request_body(self, request: Request) -> Any | str:
        try:
            body = await request.json()
        except Exception:
            body = await request.body()
            body = body.decode("utf-8")

        return body

    def password_mask(self, request_body: Any | str) -> Any | str:

        if isinstance(request_body, dict) and \
                request_body["user_password"] is not None:
            request_body["user_password"] = None

        return request_body
