from fastapi import FastAPI, Request # type: ignore
from fastapi.exceptions import RequestValidationError # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY # type: ignore

from api.router.router import api_router
from const.const import ErrorCode


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
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content=custom_errors,
    )

app.include_router(api_router)
