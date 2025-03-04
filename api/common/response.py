from typing import Any

from fastapi.responses import JSONResponse

from const.const import HttpStatusCode


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
