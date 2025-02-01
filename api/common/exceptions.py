from typing import Any, Awaitable, Callable
from api.schemas.schemas import ErrorMsg
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from const.const import ErrorCode, HttpStatusCode


class BaseHttpException(HTTPException):
    """カスタムHTTP例外の基底クラス"""
    pass


class CustomBaseException(Exception):
    """カスタム例外の基底クラス"""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class ConflictException(CustomBaseException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=HttpStatusCode.BADREQUEST.value,
            detail=detail
        )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """
    リクエストバリデーションエラー時のハンドリング
    """
    errors = exc.errors()
    custom_errors = []
    for error in errors:
        loc = error.get("loc", [])
        input_msg = error.get("input", [])
        custom_errors.append(
            {
                "code": ErrorCode.INT_VAILD.value,
                "detail": f"{loc[1]} パラメータは整数値のみです。",
                "input": input_msg
            }
        )
    return JSONResponse(
        status_code=HttpStatusCode.VALIDATION.value,
        content=custom_errors,
    )


class HttpExceptionHandler(BaseException):

    @staticmethod
    def add(app: FastAPI):
        """
        HTTPException のカスタムエラーハンドリング

        引数:
            app (FastAPI): fastAPIにエラーハンドラー追加
        """

        handlers = {
            HTTPException: HttpExceptionHandler.server_error_handler,
            RequestValidationError: HttpExceptionHandler.valid_error_handler,
            Exception: HttpExceptionHandler.exception_handler,
            TypeError: HttpExceptionHandler.type_error_handler,
        }

        for exc_type, handler in handlers.items():
            app.add_exception_handler(exc_type, handler)

    @staticmethod
    async def main_handler(req: Request, exc: Any):
        """
        HTTPException のカスタムエラーハンドリング

        引数:
            request (Request): 受け取ったリクエスト
            exc (HTTPException): 発生した HTTP 例外

        戻り値:
            JSONResponse: カスタムエラーレスポンス
        """
        if isinstance(exc, HTTPException):
            return await HttpExceptionHandler.server_error_handler(req, exc)
        if isinstance(exc, RequestValidationError):
            return await HttpExceptionHandler.valid_error_handler(req, exc)
        if isinstance(exc, TypeError):
            return await HttpExceptionHandler.type_error_handler(req, exc)
        return await HttpExceptionHandler.exception_handler(req, exc)

    @staticmethod
    async def server_error_handler(
        req: Request,
        e: HTTPException
    ) -> JSONResponse:
        """
        HTTPException のカスタムエラーハンドリング

        引数:
            req (Request): 受け取ったリクエスト
            exc (HTTPException): 発生した HTTP 例外

        戻り値:
            JSONResponse: カスタムエラーレスポンス
        """
        context = ErrorMsg(
            code=e.status_code,
            message=str(e)
        )
        return JSONResponse(
            status_code=e.status_code,
            content=context.model_dump(),
        )

    @staticmethod
    async def valid_error_handler(
        req: Request,
        e: RequestValidationError
    ) -> JSONResponse:
        """
        RequestValidationError のカスタムエラーハンドリング

        引数:
            req (Request): 受け取ったリクエスト
            exc (RequestValidationError): 発生したバリデーションの例外

        戻り値:
            JSONResponse: カスタムエラーレスポンス
        """
        # TODO:ログ出力
        context = ErrorMsg(
            code=HttpStatusCode.VALIDATION.value,
            message=str(e)
        )
        return JSONResponse(
            status_code=HttpStatusCode.VALIDATION.value,
            content=context.model_dump(),
        )

    @staticmethod
    async def type_error_handler(
        req: Request,
        e: TypeError
    ) -> JSONResponse:
        """
        CustomBaseException のカスタムエラーハンドリング

        引数:
            request (Request): 受け取ったリクエスト
            exc (CustomBaseException): 発生した 例外

        戻り値:
            JSONResponse: カスタムエラーレスポンス
        """
        context = ErrorMsg(
            code=HttpStatusCode.BADREQUEST.value,
            message=str(e)
        )
        return JSONResponse(
            status_code=HttpStatusCode.BADREQUEST.value,
            content=context.model_dump(),
        )

    @staticmethod
    async def exception_handler(
        req: Request,
        e: CustomBaseException
    ) -> JSONResponse:
        """
        CustomBaseException のカスタムエラーハンドリング

        引数:
            request (Request): 受け取ったリクエスト
            exc (CustomBaseException): 発生した 例外

        戻り値:
            JSONResponse: カスタムエラーレスポンス
        """
        context = ErrorMsg(
            code=e.status_code,
            message=str(e.detail)
        )
        return JSONResponse(
            status_code=e.status_code,
            content=context.model_dump(),
        )
