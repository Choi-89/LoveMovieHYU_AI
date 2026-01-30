import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sqlalchemy.orm import joinedload

import matplotlib.pyplot as plt



def train_genre_model():
    # 데이터 불러오기
    df = pd.read_csv("emotional_dataset.csv")
    p_index = df['P'].astype(np.float32).values
    e_index = df['E'].astype(np.float32).values
    i_index = df['I'].astype(np.float32).values
    data_dict = {
        'user_id': df["user_id"].astype(str).values,
        'movie_id': df["movie_id"].astype(str).values,
        'P': p_index,
        'E': e_index,
        'I': i_index
    }
    # TF 데이터셋 생성
    dataset = tf.data.Dataset.from_tensor_slices(data_dict)

    # 유니크 vocabulary 생성
    user_ids = df["user_id"].unique()
    movie_ids = df["movie_id"].unique()

    # 학습 설정
    train = dataset.shuffle(len(df)).batch(256).cache()
    embedding_dimension = 32

    # 모델 정의
    class UserModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            safe_user_vocab = [str(x) for x in user_ids]
            self.user_embedding = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=safe_user_vocab, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(user_ids) + 1, embedding_dimension
                )
            ])

            self.bio_normalization = tf.keras.layers.Normalization(axis=-1)
            self.bio_dense = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(embedding_dimension)
            ])

        def adapt_normalization(self, data):
            self.bio_normalization.adapt(data)

        def call(self, inputs):
            user_vec = self.user_embedding(inputs['user_id'])

            p = tf.reshape(inputs['P'], [-1, 1])
            e = tf.reshape(inputs['E'], [-1, 1])
            i = tf.reshape(inputs['I'], [-1, 1])

            bio_comb = tf.concat([p, e, i], axis=1)
            bio_norm = self.bio_normalization(bio_comb)
            bio_vec = self.bio_dense(bio_norm)

            return tf.concat([user_vec, bio_vec], axis=1)
    
    class MovieModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            safe_movie_vocab = [str(x) for x in movie_ids]
            self.movie_embedding = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(), dtype=tf.string),
                tf.keras.layers.StringLookup(
                    vocabulary=safe_movie_vocab, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(movie_ids) + 1, embedding_dimension * 2
                )
            ])

        def call(self, inputs):
            if inputs.dtype != tf.string:
                inputs = tf.strings.as_string(inputs)
            return self.movie_embedding(inputs)
        
    user_model = UserModel()
    bio_comb = np.stack([p_index, e_index, i_index], axis=1)
    user_model.adapt_normalization(bio_comb)
    movie_model = MovieModel()

    class Recommender(tfrs.Model):
        def __init__(self, user_model, movie_model):
            super().__init__()
            self.user_model = user_model
            self.movie_model = movie_model
            self.task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=tf.data.Dataset.from_tensor_slices(movie_ids).batch(128).map(movie_model)
                )
            )

        def compute_loss(self, features, training=False):
            query_embeddings = self.user_model({
                'user_id': features['user_id'],
                'P': features['P'],
                'E': features['E'],
                'I': features['I']
            })
            movie_embeddings = self.movie_model(features["movie_id"])
            return self.task(query_embeddings, movie_embeddings)
        
    model = Recommender(user_model, movie_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
    else:
        print("Initializing from scratch.")
    model.fit(train, epochs=3)

    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        tf.data.Dataset.from_tensor_slices(movie_ids).batch(128).map(
            lambda id: (id, model.movie_model(id))
        )
    )

    save_path = os.path.join("tfrs_model")

    dummy_input = {
        'user_id': tf.constant(["0"]),
        'P': tf.constant([[0.0]]),
        'E': tf.constant([[0.0]]),
        'I': tf.constant([[0.0]])
    }
    _ = index(dummy_input)

    tf.saved_model.save(index, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_genre_model()