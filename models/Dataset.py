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

    ratings.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
