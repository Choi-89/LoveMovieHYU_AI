from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from typing import Union
from sqlalchemy.orm import Session

from API.database import SessionLocal, Base, engine

class Add_Movie:
    def __init__(self, user_id: int, movie_id: int, P: float, E: float, I: float):
        self.user_id = str(user_id)
        self.movie_id = str(movie_id)
        self.P = P
        self.E = E
        self.I = I

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "movie_id": self.movie_id,
            "P": self.P,
            "E": self.E,
            "I": self.I
        }

Add_Movies = []

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
async def validate(user_id: str, P: float, E: float, I: float):
    from models.Validation import validate_model
    genre = validate_model(user_id, P, E, I)
    return genre

@app.get("/api/dataset/{user_id}/{movie_id}/{P}/{E}/{I}")
async def get_dataset_entry(user_id: str, movie_id: int, P: float, E: float, I: float):
    Add_Movies.append(Add_Movie(user_id, movie_id, P, E, I).to_dict())
    return {400: "success"}

@app.get("/add/movie/")
async def add_movie():
    from models.Add_dataset import add_dataset
    add_dataset(Add_Movies)
    return {"message": "데이터 추가 완료"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
