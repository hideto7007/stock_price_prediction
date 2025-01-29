from api.common.exceptions import HttpExceptionHandler
from fastapi import FastAPI  # type: ignore
from fastapi.security import OAuth2PasswordBearer  # type: ignore
from api.router.router import api_router
from api.middleware import TimeoutMiddleware


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


# 例外ハンドラーを登録
HttpExceptionHandler.add(app)
# app.add_exception_handler(
#     RequestValidationError,
#     validation_exception_handler
# )

# ミドルウェアの追加
# app.add_middleware(OAuth2Middleware, oauth2_scheme=oauth2_scheme)
# タイムアウトミドルウェアを追加（タイムアウト時間を指定）
app.add_middleware(TimeoutMiddleware, timeout=3)
app.include_router(api_router)
