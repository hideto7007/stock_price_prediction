# middleware.py
from fastapi import Request, HTTPException # type: ignore
from starlette.middleware.base import BaseHTTPMiddleware # type: ignore
from starlette.status import HTTP_401_UNAUTHORIZED # type: ignore
from starlette.types import ASGIApp
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import asyncio

from const.const import ErrorCode, HttpStatusCode
from api.schemas.schemas import ErrorMsg, Detail


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        session_token = request.cookies.get("session_token")
        
        if not session_token or not self.is_valid_session(session_token):
            raise HTTPException(
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                detail=[ErrorMsg(code=ErrorCode.UNAUTHORIZED.value, message="セッションが無効です。").dict()]
            )
        
        response = await call_next(request)
        return response
    
    def is_valid_session(self, token: str) -> bool:
        # セッションの有効性をチェックするロジックを実装
        # 例えば、データベースやキャッシュからセッション情報を取得して検証する
        return True
    

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
