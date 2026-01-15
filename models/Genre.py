import os
from API.database import SessionLocal
from API.models import Movie
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sqlalchemy.orm import joinedload

import matplotlib.pyplot as plt



def train_genre_model():
    data = "./movies.csv"
    dataframe = pd.read_csv(data)
    dataframe['id'] = dataframe['id'].astype(str)
    dataframe['genres'] = dataframe['genres'].astype(str)
    dataset = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    unique_genres = dataframe['genres'].unique()
    unique_movie_ids = dataframe['id'].unique()

    train_dataset = dataset.shuffle(len(dataset)).batch(256).cache()

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
            movie_embeddings = self.movie_model(features["id"])
            genre_embeddings = self.model(features["genres"])
            return self.task(genre_embeddings, movie_embeddings) #손실계산
        
    model = MovieGenreModel(movie_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    index = tfrs.layers.factorized_top_k.BruteForce(model.model)

    index.index_from_dataset(
        tf.data.Dataset.from_tensor_slices(unique_movie_ids).batch(128).map(
            lambda movie_id: (movie_id, model.movie_model(movie_id))
        )
    )
    _ = index(tf.constant(["액션"]))

    tf.saved_model.save(index, os.path.join(os.getcwd(), 'tfrs_model'))
    print("모델 저장 완료.")