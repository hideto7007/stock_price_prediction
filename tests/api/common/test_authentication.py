from api.common.authentication import Authentication
from tests.test_case import TestBase


class TestAuthentication(TestBase):

    def test_verify_password_success(self):
        """正常系: 平文とハッシュ化パスワードが一致する場合"""
        plain_password = "TestPassword123!"
        hashed_password = Authentication.hash_password(plain_password)
        self.assertTrue(
            Authentication.verify_password(
                plain_password, hashed_password
            )
        )

    def test_verify_password_failure(self):
        """異常系: 平文とハッシュ化パスワードが異なる場合"""
        plain_password = "TestPassword123!"
        wrong_password = "WrongPassword!"
        hashed_password = Authentication.hash_password(plain_password)
        self.assertFalse(
            Authentication.verify_password(
                wrong_password,
                hashed_password
            )
        )
