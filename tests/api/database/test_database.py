import os
from dotenv import load_dotenv # type: ignore
from sqlalchemy import create_engine, text # type: ignore
from sqlalchemy.ext.declarative import declarative_base # type: ignore
from sqlalchemy.orm import sessionmaker # type: ignore

load_dotenv()

TEST_DATABASE_PATH = os.getenv("TEST_DATABASE_PATH")
TEST_CREATE_PATH = os.getenv("TEST_CREATE_PATH")
TEST_DELETE_PATH = os.getenv("TEST_DELETE_PATH")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, TEST_DATABASE_PATH)}"


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
def init_db(CREATE_SQL=os.path.join(BASE_DIR, TEST_CREATE_PATH)):
    print("CREATE_SQL", CREATE_SQL)
    execute_sql_file(engine, CREATE_SQL)


# テスト用のデータベース削除
def drop_db(DELETE_SQL=os.path.join(BASE_DIR, TEST_DELETE_PATH)):
    execute_sql_file(engine, DELETE_SQL)
