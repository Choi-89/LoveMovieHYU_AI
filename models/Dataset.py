import pandas as pd
import numpy as np
from API.database import SessionLocal
from API.models import Movie
from sqlalchemy.orm import joinedload
import tensorflow as tf
import random


def create_dataset(output_file="./movies.csv"):   
    # db 값 가져오기
    db = SessionLocal()
    movies_query = db.query(Movie).options(joinedload(Movie.movie_genre)).all()

    # Dataframe 변환
    movies_data = [
        {
            "id": movie.id,
            "title": movie.title,
            "vote_average": float(movie.vote_average),
            "genres": [mg.genre.name for mg in movie.movie_genre]
        }
        for movie in movies_query
    ]

    movies = pd.DataFrame(movies_data)

    # 전처리
    movies['movie_id'] = movies['id'].astype(str)

    ratings = movies.explode('genres')
    ratings = ratings[['movie_id', 'genres']]
    ratings = ratings.dropna()

    emotion = []
    # '로맨스', '액션', '자극', 'SF',
    # '공포', '애니메이션', '범죄', '코미디', 
    # '스릴러', '집중', '전쟁', '드라마', '가족', '판타지',
    # '모험', '서부', '역사', '미스터리', '음악', '다큐멘터리', 'TV 영화'

    for raw in ratings.itertuples():
        if raw.genres not in emotion:
            emotion.append(raw.genres)

    ratings.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(emotion)