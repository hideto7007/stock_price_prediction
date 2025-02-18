import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from common.logger import Logger

logger = Logger()

load_dotenv()

TEST_DATABASE_PATH = os.getenv("TEST_DATABASE_PATH")
TEST_CREATE_PATH = os.getenv("TEST_CREATE_PATH")
TEST_DELETE_PATH = os.getenv("TEST_DELETE_PATH")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, TEST_DATABASE_PATH)}"


engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# テスト用のデータベースセッションの取得
def get_test_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def execute_sql_file(engine, filepath):
    with engine.begin() as connection:  # Use begin() to ensure commit/rollback
        with open(filepath, 'r') as f:
            sql_statements = f.read().split(';')
            for statement in sql_statements:
                statement = statement.strip()
                if statement:
                    connection.execute(text(statement + ";"))


# テスト用のテーブル初期化
def init_db(CREATE_SQL=os.path.join(BASE_DIR, TEST_CREATE_PATH)):
    execute_sql_file(engine, CREATE_SQL)


# テスト用のテーブル削除
def drop_db(DELETE_SQL=os.path.join(BASE_DIR, TEST_DELETE_PATH)):
    execute_sql_file(engine, DELETE_SQL)
