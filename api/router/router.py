from fastapi import APIRouter # type: ignore
from api.endpoints import stock_price, login


api_router = APIRouter()
api_router.include_router(login.router)
api_router.include_router(stock_price.router)
