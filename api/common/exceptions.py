from typing import Any
from api.schemas.response import Content
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from common.logger import Logger
from const.const import ErrorCode, HttpStatusCode


class BaseHttpException(HTTPException):
    """カスタムHTTP例外の基底クラス"""
    pass


class CustomBaseException(Exception):
    """カスタムHTTP例外の基底クラス"""
    pass


class ConflictException(CustomBaseException):
    pass


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
            AttributeError: HttpExceptionHandler.attribute_error_handler,
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
        await Logger.error(req, exc)
        if isinstance(exc, HTTPException):
            return await HttpExceptionHandler.server_error_handler(req, exc)
        if isinstance(exc, RequestValidationError):
            return await HttpExceptionHandler.valid_error_handler(req, exc)
        if isinstance(exc, TypeError):
            return await HttpExceptionHandler.type_error_handler(req, exc)
        if isinstance(exc, AttributeError):
            return await HttpExceptionHandler.attribute_error_handler(req, exc)
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
        context = Content[str](
            result=str(e)
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
        context = Content[str](
            result=str(e)
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
        TypeError のカスタムエラーハンドリング

        引数:
            request (Request): 受け取ったリクエスト
            exc (TypeError): 発生した 例外

        戻り値:
            JSONResponse: カスタムエラーレスポンス
        """
        context = Content[str](
            result=str(e)
        )
        return JSONResponse(
            status_code=HttpStatusCode.BADREQUEST.value,
            content=context.model_dump(),
        )

    @staticmethod
    async def attribute_error_handler(
        req: Request,
        e: AttributeError
    ) -> JSONResponse:
        """
        AttributeError のカスタムエラーハンドリング

        引数:
            request (Request): 受け取ったリクエスト
            exc (AttributeError): 発生した 例外

        戻り値:
            JSONResponse: カスタムエラーレスポンス
        """
        context = Content[str](
            result=str(e)
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
        context = Content[str](
            result=str(e)
        )
        return JSONResponse(
            status_code=HttpStatusCode.SERVER_ERROR.value,
            content=context.model_dump(),
        )
