from fastapi import FastAPI, Depends
from typing import Union
from sqlalchemy.orm import Session

from API.database import SessionLocal, Base, engine

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def read_root(db: Session= Depends(get_db)):
    return {"message": "데이터베이스 연결에 성공하였습니다."}