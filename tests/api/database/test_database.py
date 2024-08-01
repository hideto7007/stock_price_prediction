import os
from dotenv import load_dotenv # type: ignore
from sqlalchemy import create_engine, text # type: ignore
from sqlalchemy.ext.declarative import declarative_base # type: ignore
from sqlalchemy.orm import sessionmaker # type: ignore

load_dotenv()

TEST_DATABASE_PATH = os.getenv("TEST_DATABASE_PATH")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
TEST_DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, TEST_DATABASE_PATH)}"
print(TEST_DATABASE_PATH)
print(TEST_DATABASE_URL)


engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
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


# テスト用のデータベース初期化
def init_db(CREATE_SQL=os.path.join(BASE_DIR, '../sql/create.sql')):
    execute_sql_file(engine, CREATE_SQL)


# テスト用のデータベース削除
def drop_db(DELETE_SQL=os.path.join(BASE_DIR, '../sql/delete.sql')):
    execute_sql_file(engine, DELETE_SQL)
