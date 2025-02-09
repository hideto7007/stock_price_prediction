from api.common.exceptions import HttpExceptionHandler
from fastapi import FastAPI
from api.router.router import api_router
from api.middleware import OAuth2Middleware, RequestWritingLoggerMiddleware
from utils.utils import Swagger

app = FastAPI()


# 例外ハンドラーを登録
HttpExceptionHandler.add(app)
# app.add_exception_handler(
#     RequestValidationError,
#     validation_exception_handler
# )

# ミドルウェアの追加
app.add_middleware(OAuth2Middleware)
# タイムアウトミドルウェアを追加（タイムアウト時間を指定）
# app.add_middleware(TimeoutMiddleware, timeout=5)
app.add_middleware(RequestWritingLoggerMiddleware)
# app.add_middleware(MiddlewareTest)
app.include_router(api_router)

app.openapi = lambda: Swagger.custom_openapi(app)
