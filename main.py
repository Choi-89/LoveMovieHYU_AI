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
    from models.TrainingDataset import make_training_dataset
    create_dataset()
    make_training_dataset()
    train_genre_model()

    yield

    print("서버 종료")

app = FastAPI(lifespan=lifespan)
@app.get("/")
async def read_root(db: Session= Depends(get_db)):
    return {"message": "데이터베이스 연결, 학습 성공하였습니다."}

@app.get("/api/recommend/movie/{user_id}/{P}/{E}/{I}")
async def validate(user_id: int, P: float, E: float, I: float):
    from models.Validation import validate_model
    genre = validate_model(user_id, P, E, I)
    return {"recommended_movies": genre}

@app.get("/api/recommend/{theme}")
async def recommend_theme(theme: str):
    return {"message": f"{theme} 개발 중"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
