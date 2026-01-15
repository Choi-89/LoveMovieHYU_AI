from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from typing import Union
from sqlalchemy.orm import Session

from API.database import SessionLocal, Base, engine

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    from models.Genre import train_genre_model
    from models.Dataset import create_dataset
    create_dataset()
    train_genre_model()

    yield

    print("서버 종료")

app = FastAPI(lifespan=lifespan)
@app.get("/")
async def read_root(db: Session= Depends(get_db)):
    return {"message": "데이터베이스 연결, 학습 성공하였습니다."}

@app.get("/genre/{genre}")
async def validate(genre: str):
    from models.Validation import validate_model
    genre = validate_model(genre)
    return {"recommended_genres": genre}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
