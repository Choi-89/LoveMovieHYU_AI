from API.database import SessionLocal
from API.models import Movie
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sqlalchemy.orm import joinedload

import matplotlib.pyplot as plt



def train_genre_model():
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

    unique_genres = ratings['genres'].unique()
    unique_movie_ids = ratings['id'].unique()

    dataset = tf.data.Dataset.from_tensor_slices({
        "movie_id": ratings['id'].values,
        "genre": ratings['genres'].values
    })

    train_dataset = dataset.shuffle(len(ratings)).batch(256).cache()

    embedding_dimension = 32

    genre_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_genres, mask_token=None
        ),
        tf.keras.layers.Embedding(len(unique_genres) + 1, embedding_dimension)
    ])

    movie_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_ids, mask_token=None
        ),
        tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension)
    ])

    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=tf.data.Dataset.from_tensor_slices(unique_movie_ids).batch(128).map(movie_model)
        )
    )

    class MovieGenreModel(tfrs.Model):
        def __init__(self, movie_model, task):
            super().__init__()
            self.model = genre_model
            self.movie_model = movie_model
            self.task = task

        def compute_loss(self, features, training=False):
            movie_embeddings = self.movie_model(features["movie_id"])
            genre_embeddings = self.model(features["genre"])
            return self.task(genre_embeddings, movie_embeddings) #손실계산
        
    model = MovieGenreModel(movie_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    history = model.fit(train_dataset, epochs=15)

    print("="*50)
    print("훈련 기록에 저장된 키(Keys) 목록:")
    print(history.history.keys())
    print("="*50)

    # 손실 및 정확도 그래프 확인
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1) # 1행 2열 중 첫 번째
    plt.plot(history.history['loss'], label='Training Loss')
    plt.legend()

    accuracy_key = 'factorized_top_k/top_10_categorical_accuracy'
    if 'factorized_top_k/top_10_accuracy' in history.history:
        accuracy_key = 'factorized_top_k/top_10_accuracy'
    elif 'factorized_top_k/top_5_accuracy' in history.history:
        accuracy_key = 'factorized_top_k/top_5_accuracy'

    plt.subplot(1, 2, 2)
    plt.plot(history.history[accuracy_key], label=f'Accuracy ({accuracy_key})')
    plt.title('Model Top-K Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    index = tfrs.layers.factorized_top_k.BruteForce(model.model)

    index.index_from_dataset(
        tf.data.Dataset.from_tensor_slices(unique_movie_ids).batch(128).map(
            lambda movie_id: (movie_id, model.movie_model(movie_id))
        )
    )

    query_genre = "액션"
    _, recommendations = index(tf.constant([query_genre]))
    print(f"\n---'{query_genre}' 장르 추천(10개)---")
    for movie_id in recommendations[0, :10].numpy():
        title = movies[movies['id'] == movie_id.decode('utf-8')]['title'].values[0]
        print(f"추천 영화: {title}")
    
    return [title.decode('utf-8') for title in recommendations[0, :10].numpy()]