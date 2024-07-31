# middleware.py
from fastapi import Request, HTTPException # type: ignore
from starlette.middleware.base import BaseHTTPMiddleware # type: ignore
from fastapi.security import OAuth2PasswordBearer # type: ignore
from starlette.types import ASGIApp
from fastapi.responses import JSONResponse
import asyncio
from typing import List

from const.const import ErrorCode, HttpStatusCode
from api.schemas.schemas import ErrorMsg, Detail

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class OAuth2Middleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, oauth2_scheme: OAuth2PasswordBearer, exempt_paths: List[str] = None):
        super().__init__(app)
        self.oauth2_scheme = oauth2_scheme
        self.exempt_paths = exempt_paths or ["/docs", "/redoc", "/openapi.json", "/favicon.ico"]

    async def dispatch(self, request: Request, call_next):
        # リクエストのパスが認証を必要としない場合はスキップ
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        try:
            token = await self.oauth2_scheme(request)
            if not self.is_valid_token(token):
                error_msg = ErrorMsg(code=ErrorCode.UNAUTHORIZED.value, message="トークンが無効です。")
                detail = Detail(detail=[error_msg])
                return JSONResponse(
                    status_code=HttpStatusCode.UNAUTHORIZED.value,
                    content=detail.dict()
                )
        except HTTPException as e:
            # トークンがない、または不正なトークンが渡された場合
            error_msg = ErrorMsg(code=ErrorCode.UNAUTHORIZED.value, message="認証に失敗しました。")
            detail = Detail(detail=[error_msg])
            return JSONResponse(
                status_code=e.status_code,
                content=detail.dict()
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
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            error_msg = ErrorMsg(code=ErrorCode.TIME_OUT, message="read time out.")
            detail = Detail(detail=[error_msg])
            return JSONResponse(
                status_code=HttpStatusCode.TIME_OUT.value,
                content=detail.dict()
            )
