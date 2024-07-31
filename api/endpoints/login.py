
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv # type: ignore
from typing import Union
from api.models.models import AccountModel
from const.const import HttpStatusCode
from sqlalchemy.orm import Session # type: ignore
from fastapi import APIRouter, HTTPException, Depends # type: ignore
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm # type: ignore
from jose import JWTError, jwt # type: ignore
from passlib.context import CryptContext # type: ignore

from api.databases.databases import get_db


from api.schemas.schemas import (
    ErrorMsg,
    Token,
    TokenData,
    Account,
    AccountInDB,
    CreateAccount
)


load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")


router = APIRouter()

# パスワードのハッシュ化に使用するコンテキスト
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2スキーマの定義
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    """パスワードの検証関数"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """パスワードのハッシュ化関数"""
    return pwd_context.hash(password)


def get_user(username: str, db: Session = Depends(get_db)):
    """ユーザー名からユーザー情報を取得する関数"""
    username = db.query(AccountModel).filter(AccountModel.username == username).first()
    if username:
        return username


def authenticate_user(username: str, password: str, db: Session = Depends(get_db)):
    """ユーザーの認証関数"""
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    """アクセストークンの生成関数"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """現在のユーザーを取得する関数"""
    credentials_exception = HTTPException(
        status_code=HttpStatusCode.UNAUTHORIZED.value,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(token_data.username, db)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: Account = Depends(get_current_user)):
    """現在のアクティブなユーザーを取得する関数"""
    if current_user.disabled:
        raise HTTPException(status_code=HttpStatusCode.BADREQUEST.value, detail="Inactive user")
    return current_user


@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> Token:
    """アクセストークンを取得するためのエンドポイント"""
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=HttpStatusCode.UNAUTHORIZED.value,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@router.get("/users/me/", response_model=Account)
async def read_users_me(current_user: Account = Depends(get_current_active_user)):
    """現在のユーザー情報を取得するエンドポイント"""
    return current_user


@router.get("/users/me/items/")
async def read_own_items(current_user: Account = Depends(get_current_active_user)):
    """現在のユーザーのアイテムを取得するエンドポイント"""
    return [{"item_id": "Foo", "owner": current_user.username}]
