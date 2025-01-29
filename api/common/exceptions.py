from api.schemas.schemas import ErrorMsg
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from const.const import ErrorCode, HttpStatusCode


class ConflictException(Exception):
    pass


class BaseHTTPException(HTTPException):
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

        app.add_exception_handler(
            HTTPException, HttpExceptionHandler.server_error_handler)
        app.add_exception_handler(
            RequestValidationError, HttpExceptionHandler.valid_error_handler)
        app.add_exception_handler(
            Exception, HttpExceptionHandler.exception_handler)

    @classmethod
    async def main_handler(cls, req: Request, exc: HTTPException):
        """
        HTTPException のカスタムエラーハンドリング

        引数:
            request (Request): 受け取ったリクエスト
            exc (HTTPException): 発生した HTTP 例外

        戻り値:
            JSONResponse: カスタムエラーレスポンス
        """
        print("check", type(exc))
        if isinstance(exc, HTTPException):
            return await cls.server_error_handler(req, exc)
        if isinstance(exc, RequestValidationError):
            return await cls.valid_error_handler(req, exc)
        return await cls.exception_handler(req, exc)

    @classmethod
    async def server_error_handler(cls, req: Request, e: HTTPException):
        """
        HTTPException のカスタムエラーハンドリング

        引数:
            req (Request): 受け取ったリクエスト
            exc (HTTPException): 発生した HTTP 例外

        戻り値:
            JSONResponse: カスタムエラーレスポンス
        """
        # TODO:ログ出力
        context = ErrorMsg(
            code=e.status_code,
            message=str(e)
        )
        return JSONResponse(
            status_code=e.status_code,
            content=context.model_dump(),
        )

    @classmethod
    async def valid_error_handler(
        cls,
        req: Request,
        e: RequestValidationError
    ):
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

    @classmethod
    async def exception_handler(cls, req: Request, e: Exception):
        """
        Exception のカスタムエラーハンドリング

        引数:
            request (Request): 受け取ったリクエスト
            exc (Exception): 発生した 例外

        戻り値:
            JSONResponse: カスタムエラーレスポンス
        """
        # TODO:ログ出力
        context = ErrorMsg(
            code=502,
            message=str(e)
        )
        return JSONResponse(
            status_code=502,
            content=context.model_dump(),
        )
