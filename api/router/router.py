from fastapi import APIRouter
from api.endpoints import stock_price, login

api_dict = {
    "login": login.router,
    "stock_price": stock_price.router
}

api_router = APIRouter()


for prefix, router in api_dict.items():
    api_router.include_router(router, prefix=f"/api/{prefix}")
