from fastapi import FastAPI, Request, HTTPException # type: ignore
from fastapi.exceptions import RequestValidationError # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from fastapi.security import OAuth2PasswordBearer # type: ignore

from api.router.router import api_router
from const.const import ErrorCode, HttpStatusCode
from api.middleware import TimeoutMiddleware, OAuth2Middleware


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
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


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# ミドルウェアの追加
# app.add_middleware(OAuth2Middleware, oauth2_scheme=oauth2_scheme)
# タイムアウトミドルウェアを追加（タイムアウト時間を指定）
app.add_middleware(TimeoutMiddleware, timeout=3)
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run("main:app", port=8000, host="0.0.0.0", reload=True)