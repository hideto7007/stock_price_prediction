from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from const.const import HttpStatusCode


class Response(JSONResponse):
    """FastAPI の JSONResponse をラップするカスタムレスポンスクラス"""

    def __new__(
        cls,
        request: Request,
        status_code: int,
        content: Any
    ) -> JSONResponse:
        """インスタンス作成時に JSONResponse を直接返す"""
        return super().__new__(cls)

    def __init__(
            self,
            request: Request,
            status_code: int,
            content: Any
    ) -> None:
        super().__init__(status_code=status_code, content=content)
        self.request = request
        self.status_code = status_code
        self.content = content


class ValidationResponse(JSONResponse):
    """FastAPI の JSONResponse をラップするカスタムレスポンスクラス"""

    def __new__(
        cls,
        content: Any
    ) -> JSONResponse:
        """インスタンス作成時に JSONResponse を直接返す"""
        return super().__new__(cls)

    def __init__(
            self,
            content: Any
    ) -> None:
        super().__init__(
            status_code=HttpStatusCode.BADREQUEST.value,
            content=content
        )
        self.content = content
