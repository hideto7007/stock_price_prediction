from fastapi.security import OAuth2PasswordBearer  # type: ignore
from passlib.context import CryptContext  # type: ignore

# OAuth2スキーマの定義
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Authentication:
    """
    認証クラス
    """

    __pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    @classmethod
    def verify_password(
        cls,
        plain_password: str,
        hashed_password: str
    ) -> bool:
        """
            平文とハッシュパスワード検証

            引数:
                plain_password (str): 平文パスワード
                hashed_password (str): ハッシュパスワード

            戻り値:
                bool: 一致していたら、True、それ以外は、False
        """

        return Authentication.__pwd_context.verify(
            plain_password,
            hashed_password
        )

    @classmethod
    def hash_password(
        cls,
        plain_password: str,
    ) -> str:
        """
            平文をハッシュパスワードに変換

            引数:
                plain_password (str): 平文パスワード

            戻り値:
                str: ハッシュ化したパスワード
        """

        return Authentication.__pwd_context.hash(
            plain_password
        )
