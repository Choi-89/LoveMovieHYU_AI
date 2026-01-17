import pandas as pd
import numpy as np
from API.database import SessionLocal
from API.models import Movie
from sqlalchemy.orm import joinedload
import tensorflow as tf



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
    movies['id'] = movies['id'].astype(str)

    ratings = movies.explode('genres')
    ratings = ratings[['id', 'genres']]
    ratings = ratings.dropna()

    # genre -> feel 변환
    for raw in ratings.itertuples():
        genre = raw.genres
        if genre in ["드라마", "힐링", "가족"]:
            ratings.loc[raw.Index, 'genres'] = "편안해요"
        elif genre in ["코미디", "로코", "애니메이션"]:
            ratings.loc[raw.Index, 'genres'] = "신나"
        elif genre in ["멜로", "감정 드라마"]:
            ratings.loc[raw.Index, 'genres'] = "슬퍼"
        elif genre in ["드라마", "독립"]:
            ratings.loc[raw.Index, 'genres'] = "우울"
        elif genre in ["예술", "독립", "사회"]:
            ratings.loc[raw.Index, 'genres'] = "생각"
        elif genre in ["스릴러", "미스터리"]:
            ratings.loc[raw.Index, 'genres'] = "집중"
        elif genre in ["액션", "스포츠"]:
            ratings.loc[raw.Index, 'genres'] = "자극"
        elif genre in ["로드무비", "일상"]:
            ratings.loc[raw.Index, 'genres'] = "지쳐"
        else:
            ratings.loc[raw.Index, 'genres'] = "기타"

    ratings.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
