import os
from dotenv import load_dotenv # type: ignore
from sqlalchemy import create_engine # type: ignore
from sqlalchemy.ext.declarative import declarative_base # type: ignore
from sqlalchemy.orm import sessionmaker # type: ignore

load_dotenv()

DATABASE_PATH = os.getenv("DATABASE_PATH")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
print(DATABASE_PATH)
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, DATABASE_PATH)}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# データベースセッションの取得
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
