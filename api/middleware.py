# middleware.py
import uuid
from fastapi import Request, HTTPException  # type: ignore
from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore
from fastapi.security import OAuth2PasswordBearer  # type: ignore
from starlette.types import ASGIApp
from fastapi.responses import JSONResponse
import asyncio
from typing import List

from const.const import HttpStatusCode
from api.schemas.schemas import Content

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class OAuth2Middleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, oauth2_scheme: OAuth2PasswordBearer, exempt_paths: List[str] = None):
        super().__init__(app)
        self.oauth2_scheme = oauth2_scheme
        self.exempt_paths = exempt_paths or [
            "/docs", "/redoc", "/openapi.json", "/favicon.ico", "/login"]

    async def dispatch(self, request: Request, call_next):
        # リクエストのパスが認証を必要としない場合はスキップ
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        try:
            token = await self.oauth2_scheme(request)
            if not self.is_valid_token(token):
                error_msg = Content[str](result="トークンが無効です。")
                return JSONResponse(
                    status_code=HttpStatusCode.UNAUTHORIZED.value,
                    content=error_msg.model_dump()
                )
        except HTTPException as e:
            # トークンがない、または不正なトークンが渡された場合
            error_msg = Content[str](result="認証に失敗しました。")
            return JSONResponse(
                status_code=e.status_code,
                content=error_msg.model_dump()
            )
        return await call_next(request)

    def is_valid_token(self, token: str) -> bool:
        print("debug", token)
        # トークンの有効性をチェックするロジックを実装
        return True  # ここでは常に有効とする仮の実装


class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, timeout: int = 30):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        timeout = self.timeout
        # 特定のパスでタイムアウトを変更
        if request.url.path.startswith("/create") or request.url.path.startswith("/update"):
            timeout = 1800  # 30分
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except asyncio.TimeoutError:
            error_msg = Content[str](result="read time out.")
            return JSONResponse(
                status_code=HttpStatusCode.TIMEOUT.value,
                content=error_msg.model_dump()
            )


class RequestIDMiddleware(BaseHTTPMiddleware):
    """リクエストごとに一意の request_id を設定する Middleware"""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
