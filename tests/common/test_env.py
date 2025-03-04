

from common.env import Env
from tests.test_case import TestBase


class TestEnv(TestBase):

    def test_env_instance_success_01(self):
        """
        正常系: `Env`のインスタンス化成功
        """

        env = Env.get_instance()

        self.assertIsInstance(env, Env)

    def test_env_instance_error_01(self):
        """
        異常系: `Env`のインスタンス化失敗
        """

        with self.assertRaisesRegex(
            ValueError,
            "get_instanceメソッドを経由して作成してください。"
        ):
            Env()
